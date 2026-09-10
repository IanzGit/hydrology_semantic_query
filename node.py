from __future__ import annotations

import json
import logging
import time
from typing import Any

from langchain_core.messages import AIMessage, HumanMessage, RemoveMessage, SystemMessage
from langchain_core.runnables import RunnableConfig
from langgraph.graph.message import REMOVE_ALL_MESSAGES
from pydantic import BaseModel, ConfigDict, ValidationError, field_validator

from app.agents.messages import stringify_message_content
from app.agents.scenarios.cqccri_smart_query.subgraph.common_models import (
    SmartQueryResult as CqccriSmartQueryResult,
)
from app.agents.scenarios.cqccri_smart_query.subgraph.common_models import (
    input_query,
    resolve_smart_query_answer,
)
from app.agents.streaming import llm_stream_output

from .contracts import (
    ExecutionPlanRevision,
    FailureKind,
    MainAgentAction,
    MainAgentDecision,
    QueryOutcome,
    QueryTask,
    QueryTaskExecutionContext,
    ReportSectionRequirement,
    ReportTask,
    SemanticQueryError,
    SemanticQueryResult,
    StepStatus,
    TaskExecutionResult,
    TaskExecutionStatus,
)
from .prompts import build_messages, main_agent_system_prompt
from .query_child.runtime import (
    HydrologySemanticQueryServices,
    build_error,
    build_step,
    outcome_for_error,
    request_data,
    safe_response_excerpt,
    thought_output,
)
from .query_child.state import QueryAgentState
from .report_child.report import REPORT_FAILURE_WARNING
from .report_child.state import ReportAgentState
from .state import (
    HYDROLOGY_SEMANTIC_QUERY_SCENE_ID,
    HydrologySemanticQueryMemory,
    HydrologySemanticQueryState,
)

logger = logging.getLogger("uvicorn.error")

MAX_HYDROLOGY_HISTORY_MESSAGES = 24

_OUTCOME_ERRORS = {
    QueryOutcome.PLANNER_ERROR.value: (
        "当前问题暂时无法转换为有效的数据查询，请调整查询条件后重试。"
    ),
    QueryOutcome.EXECUTION_ERROR.value: "当前数据查询暂时未能完成，请稍后重试。",
    QueryOutcome.SYSTEM_ERROR.value: "当前查询暂时无法完成。",
}


class StandaloneQuestion(BaseModel):
    standalone_question: str

    model_config = ConfigDict(extra="forbid")

    @field_validator("standalone_question")
    @classmethod
    def validate_question(cls, value: str) -> str:
        question = value.strip()
        if not question:
            raise ValueError("standalone_question 不能为空")
        return question


def response_format() -> dict[str, Any]:
    return {
        "type": "json_schema",
        "json_schema": {
            "name": "hydrology_standalone_question",
            "strict": True,
            "schema": StandaloneQuestion.model_json_schema(),
        },
    }


def parse_standalone_question(text: str) -> str:
    return StandaloneQuestion.model_validate_json(text).standalone_question


def _reset_parent_state(services: HydrologySemanticQueryServices) -> dict[str, Any]:
    return {
        "answer": "",
        "standalone_question": None,
        "catalog_mode": None,
        "retrieval_trace": None,
        "selected_models": [],
        "semantic_query": None,
        "compiled_sql": None,
        "compiled_params": [],
        "columns": [],
        "rows": [],
        "steps": [],
        "warnings": list(services.startup_warnings),
        "attempts": 0,
        "stage": "initialize",
        "error": None,
        "outcome": None,
        "result": None,
        "query_history": [],
        "matched_playbook": None,
        "plan_revisions": [],
        "pending_tasks": [],
        "report_sections": [],
        "task_results": [],
        "current_task": None,
        "main_action": None,
        "direct_answer": "",
        "dispatch_count": 0,
        "querying_blocked": False,
        "stream_outputs": [],
    }


def _input_scene_id(smart_input: Any) -> str:
    if isinstance(smart_input, dict):
        return str(smart_input.get("scene_id", "") or "")
    return str(getattr(smart_input, "scene_id", "") or "")


def _input_current_time(smart_input: Any) -> Any:
    if isinstance(smart_input, dict):
        return smart_input.get("current_time")
    return getattr(smart_input, "current_time", None)


def make_cqccri_entry_node(services: HydrologySemanticQueryServices):
    """构造 CQCCRI 公共输入到原水文场景 State 的白名单适配节点。"""

    async def entry(state: HydrologySemanticQueryState) -> dict[str, Any]:
        smart_input = state.get("smart_query_input") or {}
        scene_id = str(state.get("sence_id", "") or "")
        input_scene_id = _input_scene_id(smart_input)
        query = input_query(smart_input).strip()
        valid_scene = (
            scene_id == HYDROLOGY_SEMANTIC_QUERY_SCENE_ID
            and input_scene_id == HYDROLOGY_SEMANTIC_QUERY_SCENE_ID
        )

        reset = _reset_parent_state(services)
        metadata = {
            "original_query": query,
            "current_time": _input_current_time(smart_input),
        }
        draft: dict[str, Any] = {}
        if not valid_scene:
            draft = {
                "_entry_rejected": True,
                "_entry_error": "水文语义查询场景 ID 未通过校验。",
            }
        elif not query:
            draft = {
                "_entry_rejected": True,
                "_entry_error": "水文语义查询问题不能为空。",
            }

        return {
            **reset,
            "query": query,
            "metadata": metadata,
            "draft": draft,
            "plan": {},
            "smart_query_output_result": CqccriSmartQueryResult(),
            "hydrology_semantic_query_memory": HydrologySemanticQueryMemory.from_dict(
                state.get("hydrology_semantic_query_memory")
            ),
        }

    return entry


async def _contextualize_question(
    state: HydrologySemanticQueryState,
    runtime,
    services: HydrologySemanticQueryServices,
) -> dict[str, Any]:
    request = request_data(state, services.settings)
    question = request["question"]
    if not request["conversation_context"]:
        return {"standalone_question": question}
    started = time.perf_counter()
    steps = list(state.get("steps", []))
    warnings = list(state.get("warnings", []))
    try:
        messages = build_messages(
            question=question,
            conversation_context=request["conversation_context"],
        )
        model = runtime.get_chat_model(streaming=False).bind(
            response_format=response_format(),
            extra_body={"enable_thinking": False},
        )
        response = await model.ainvoke(messages, config={"callbacks": []})
        standalone_question = parse_standalone_question(
            stringify_message_content(response.content)
        )
        steps.append(build_step(
            "question_contextualization",
            started,
            attempt=1,
            status=StepStatus.SUCCESS,
            metadata={"rewritten": standalone_question != question},
        ))
        return {
            "standalone_question": standalone_question,
            "steps": steps,
            "warnings": warnings,
            "stage": "question_contextualization",
            "stream_outputs": thought_output(
                "理解多轮问题",
                f"已将当前追问改写为独立问题：{standalone_question}",
            ),
        }
    except Exception as exc:
        warning = (
            "多轮问题改写失败，已使用原问题规划。"
            f"原因：{safe_response_excerpt(str(exc))[:200]}"
        )
        warnings.append(warning)
        steps.append(build_step(
            "question_contextualization",
            started,
            attempt=1,
            status=StepStatus.SKIPPED,
            summary=warning,
        ))
        return {
            "standalone_question": question,
            "steps": steps,
            "warnings": warnings,
            "stage": "question_contextualization",
        }


def make_initialize_node(runtime, services: HydrologySemanticQueryServices):
    async def initialize(state: HydrologySemanticQueryState) -> dict[str, Any]:
        started = time.perf_counter()
        reset = _reset_parent_state(services)
        metadata = dict(state.get("metadata") or {})
        try:
            request_data(state, services.settings)
        except Exception as exc:
            error = build_error(
                stage="initialize",
                code=exc.__class__.__name__,
                kind=FailureKind.VALIDATION,
                exc=exc,
            )
            step = build_step(
                "initialize",
                started,
                attempt=1,
                status=StepStatus.FAILED,
                summary=str(exc)[:1000],
            )
            return {
                **reset,
                "metadata": metadata,
                "steps": [step],
                "error": error,
                "outcome": QueryOutcome.PLANNER_ERROR,
            }
        step = build_step(
            "initialize",
            started,
            attempt=1,
            status=StepStatus.SUCCESS,
            metadata={
                "business_playbook_count": len(services.business_playbooks or ()),
            },
        )
        initialized: HydrologySemanticQueryState = {
            **state,
            **reset,
            "metadata": metadata,
            "steps": [step],
            "stream_outputs": thought_output(
                "准备执行计划",
                f"已加载 {len(services.business_playbooks or ())} 个典型场景知识文件",
            ),
        }
        contextualized = await _contextualize_question(
            initialized,
            runtime,
            services,
        )
        if contextualized.get("stream_outputs"):
            contextualized["stream_outputs"] = [
                *initialized["stream_outputs"],
                *contextualized["stream_outputs"],
            ]
        initialized.update(contextualized)
        return initialized

    return initialize


def _failure_outcome(
    state: HydrologySemanticQueryState,
) -> tuple[QueryOutcome, SemanticQueryError | None]:
    failures = [
        item
        for item in state.get("task_results", [])
        if item.status == TaskExecutionStatus.FAILED
    ]
    if failures:
        error = next(
            (item.error for item in reversed(failures) if item.error is not None),
            None,
        )
        return outcome_for_error(error), error
    if any(
        item.status == TaskExecutionStatus.NO_DATA
        for item in state.get("task_results", [])
    ):
        return QueryOutcome.NO_DATA, None
    error = state.get("error")
    return outcome_for_error(error), error


def make_finalize_node(services: HydrologySemanticQueryServices):
    async def finalize(state: HydrologySemanticQueryState) -> dict[str, Any]:
        started = time.perf_counter()
        steps = list(state.get("steps", []))
        warnings = list(state.get("warnings", []))
        task_results = list(state.get("task_results", []))
        direct_answer = state.get("direct_answer", "").strip()
        if (
            not task_results
            and state.get("outcome") is None
            and state.get("error") is None
        ):
            answer = direct_answer or "请提供需要查询的水文数据、范围或条件。"
            steps.append(build_step(
                "result_finalize",
                started,
                attempt=1,
                status=StepStatus.SUCCESS,
            ))
            return {
                "answer": answer,
                "result": None,
                "steps": steps,
                "warnings": warnings,
                "stream_outputs": [llm_stream_output(text=answer)],
            }
        successful = [
            item
            for item in task_results
            if item.status == TaskExecutionStatus.SUCCESS
            and item.query_record is not None
        ]
        if successful:
            incomplete_count = len(task_results) - len(successful)
            outcome = (
                QueryOutcome.PARTIAL_SUCCESS
                if incomplete_count
                else QueryOutcome.SUCCESS
            )
            error = next(
                (
                    item.error
                    for item in reversed(task_results)
                    if item.status == TaskExecutionStatus.FAILED
                    and item.error is not None
                ),
                None,
            )
            top = successful[-1].query_record
            answer = (
                f"查询计划部分完成，共 {len(task_results)} 个任务，"
                f"实际派发 {state.get('dispatch_count', 0)} 个，"
                f"获得 {len(successful)} 个有效结果，"
                f"另有 {incomplete_count} 个任务未成功。"
                if incomplete_count
                else (
                    f"查询计划已完成，共派发 {state.get('dispatch_count', 0)} 个任务，"
                    f"获得 {len(successful)} 个有效结果。"
                )
            )
        else:
            outcome, error = _failure_outcome(state)
            records = [
                item.query_record
                for item in task_results
                if item.query_record is not None
            ]
            top = records[-1] if records else None
            if outcome == QueryOutcome.NO_DATA:
                answer = "未查询到符合当前条件的数据。"
            elif outcome == QueryOutcome.PLANNER_ERROR:
                answer = "当前问题暂时无法转换为有效的数据查询，请调整查询条件后重试。"
            elif outcome == QueryOutcome.EXECUTION_ERROR:
                answer = "当前数据查询暂时未能完成，请稍后重试。"
            else:
                answer = "当前查询暂时无法完成。"
        steps.append(build_step(
            "result_finalize",
            started,
            attempt=max(1, state.get("attempts", 0)),
            status=StepStatus.SUCCESS,
        ))
        result = SemanticQueryResult(
            outcome=outcome,
            semantic_query=top.semantic_query if top else None,
            columns=top.columns if top else [],
            rows=top.rows if top else [],
            row_count=top.row_count if top else 0,
            attempts=state.get("attempts", 0),
            query_count=len(state.get("query_history", [])),
            query_history=state.get("query_history", []),
            matched_playbook=state.get("matched_playbook"),
            plan_revisions=state.get("plan_revisions", []),
            task_results=task_results,
            compiled_sql=top.compiled_sql if top else None,
            compiled_params=top.compiled_params if top else [],
            catalog_mode=state.get("catalog_mode"),
            query_mode=top.semantic_query.query_mode if top else None,
            selected_models=top.selected_models if top else [],
            retrieval_trace=state.get("retrieval_trace"),
            warnings=warnings,
            steps=steps,
            error=error,
        )
        metadata = dict(state.get("metadata") or {})
        metadata["hydrology_semantic_query_result"] = result.model_dump(
            mode="json",
            by_alias=True,
            exclude_none=True,
        )
        outputs = []
        if outcome not in {QueryOutcome.SUCCESS, QueryOutcome.PARTIAL_SUCCESS}:
            if state.get("report_sections"):
                outputs.extend(thought_output(
                    "执行报告任务",
                    _task_checklist_detail(
                        "查询任务未获得有效结果，已跳过报告生成。",
                        _execution_plan(state),
                        task_results,
                        state.get("report_sections", []),
                        TaskExecutionStatus.SKIPPED,
                    ),
                ))
            outputs.append(llm_stream_output(text=answer))
        return {
            "answer": answer,
            "result": result,
            "steps": steps,
            "warnings": warnings,
            "metadata": metadata,
            "stream_outputs": outputs,
        }

    return finalize


def make_report_dispatch_node(compiled_report_graph):
    async def dispatch(
        state: HydrologySemanticQueryState,
        config: RunnableConfig,
    ) -> dict[str, Any]:
        started = time.perf_counter()
        result = state.get("result")
        if result is None or result.outcome not in {
            QueryOutcome.SUCCESS,
            QueryOutcome.PARTIAL_SUCCESS,
        }:
            answer = state.get("answer", "")
            outputs = []
            if state.get("report_sections"):
                outputs.extend(thought_output(
                    "执行报告任务",
                    _task_checklist_detail(
                        "查询任务未获得有效结果，已跳过报告生成。",
                        _execution_plan(state),
                        list(state.get("task_results", [])),
                        state.get("report_sections", []),
                        TaskExecutionStatus.SKIPPED,
                    ),
                ))
            if answer:
                outputs.append(llm_stream_output(text=answer))
            return {"stream_outputs": outputs}
        report_task = ReportTask(
            original_question=request_question(state),
            sections=state.get("report_sections", []),
            task_results=state.get("task_results", []),
        )
        report_state: ReportAgentState = {
            "report_task": report_task,
            "fallback_answer": state.get("answer", ""),
            "steps": [],
        }
        try:
            output = await compiled_report_graph.ainvoke(report_state, config=config)
        except Exception as exc:
            logger.warning("hydrology report generation failed: %s", str(exc)[:1000])
            warnings = list(state.get("warnings", []))
            if REPORT_FAILURE_WARNING not in warnings:
                warnings.append(REPORT_FAILURE_WARNING)
            steps = list(state.get("steps", []))
            steps.append(build_step(
                "report_agent",
                started,
                attempt=1,
                status=StepStatus.FAILED,
                summary=str(exc)[:1000],
            ))
            result = result.model_copy(update={
                "warnings": warnings,
                "steps": steps,
            }, deep=True)
            metadata = dict(state.get("metadata") or {})
            metadata["hydrology_semantic_query_result"] = result.model_dump(
                mode="json",
                by_alias=True,
                exclude_none=True,
            )
            answer = state.get("answer", "")
            stream_outputs = thought_output(
                "执行报告任务",
                _task_checklist_detail(
                    "报告生成失败，已返回查询结果摘要。",
                    _execution_plan(state),
                    list(state.get("task_results", [])),
                    state.get("report_sections", []),
                    TaskExecutionStatus.FAILED,
                ),
            )
            if answer:
                stream_outputs.append(llm_stream_output(text=answer))
            return {
                "answer": answer,
                "result": result,
                "warnings": warnings,
                "steps": steps,
                "metadata": metadata,
                "stream_outputs": stream_outputs,
            }
        answer = output["answer"]
        if result.outcome == QueryOutcome.PARTIAL_SUCCESS:
            answer = f"> **注意：{state.get('answer', '')}**\n\n{answer}"
        warnings = list(state.get("warnings", []))
        for warning in output.get("warnings", []):
            if warning not in warnings:
                warnings.append(warning)
        steps = [*state.get("steps", []), *output.get("steps", [])]
        result = result.model_copy(update={
            "warnings": warnings,
            "steps": steps,
        }, deep=True)
        metadata = dict(state.get("metadata") or {})
        metadata["hydrology_semantic_query_result"] = result.model_dump(
            mode="json",
            by_alias=True,
            exclude_none=True,
        )
        return {
            "answer": answer,
            "result": result,
            "warnings": warnings,
            "steps": steps,
            "metadata": metadata,
            "stream_outputs": thought_output(
                "执行报告任务",
                _task_checklist_detail(
                    "报告生成完成。",
                    _execution_plan(state),
                    list(state.get("task_results", [])),
                    state.get("report_sections", []),
                    TaskExecutionStatus.SUCCESS,
                ),
            ) + output.get("outputs", []),
        }

    return dispatch


def _internal_outcome_value(result: Any, state: HydrologySemanticQueryState) -> str:
    if isinstance(result, dict):
        raw_outcome = result.get("outcome")
    else:
        raw_outcome = getattr(result, "outcome", None)
    if raw_outcome is None:
        raw_outcome = state.get("outcome")
    return str(getattr(raw_outcome, "value", raw_outcome) or "")


async def cqccri_result_node(state: HydrologySemanticQueryState) -> dict[str, Any]:
    """将水文场景内部结果统一映射为 ``SmartQueryResult``。"""
    draft = state.get("draft") or {}
    internal_result = state.get("result")
    internal_outcome = _internal_outcome_value(internal_result, state)
    answer = str(state.get("answer", "") or "").strip()

    if draft.get("_entry_rejected"):
        result = CqccriSmartQueryResult(
            scene_id=HYDROLOGY_SEMANTIC_QUERY_SCENE_ID,
            outcome="rejected",
            text2sql_error=str(
                draft.get("_entry_error") or "水文语义查询输入未通过校验。"
            ),
        )
    elif internal_outcome == QueryOutcome.SUCCESS.value:
        result = CqccriSmartQueryResult(
            scene_id=HYDROLOGY_SEMANTIC_QUERY_SCENE_ID,
            outcome="completed",
            text2sql_answer=answer,
        )
    elif internal_outcome == QueryOutcome.PARTIAL_SUCCESS.value:
        result = CqccriSmartQueryResult(
            scene_id=HYDROLOGY_SEMANTIC_QUERY_SCENE_ID,
            outcome="degraded",
            text2sql_answer=answer,
        )
    elif internal_outcome == QueryOutcome.NO_DATA.value:
        result = CqccriSmartQueryResult(
            scene_id=HYDROLOGY_SEMANTIC_QUERY_SCENE_ID,
            outcome="completed_empty",
            text2sql_answer=answer or "未查询到符合当前条件的数据。",
        )
    elif internal_outcome in _OUTCOME_ERRORS:
        result = CqccriSmartQueryResult(
            scene_id=HYDROLOGY_SEMANTIC_QUERY_SCENE_ID,
            outcome="failed",
            text2sql_error=answer or _OUTCOME_ERRORS[internal_outcome],
        )
    elif internal_result is None and answer:
        result = CqccriSmartQueryResult(
            scene_id=HYDROLOGY_SEMANTIC_QUERY_SCENE_ID,
            outcome="completed",
            text2sql_answer=answer,
        )
    else:
        result = CqccriSmartQueryResult(
            scene_id=HYDROLOGY_SEMANTIC_QUERY_SCENE_ID,
            outcome="failed",
            text2sql_error="当前查询暂时无法完成。",
        )

    visible_answer = resolve_smart_query_answer(result)
    return {
        "smart_query_output_result": result,
        "messages": [AIMessage(content=visible_answer)],
    }


async def clean_node(state: HydrologySemanticQueryState) -> dict[str, Any]:
    """清空本轮中间态，并将场景历史裁剪为最近 24 条最终对话。"""
    messages = list(state.get("messages", []) or [])
    durable_messages = [
        message for message in messages if isinstance(message, (HumanMessage, AIMessage))
    ][-MAX_HYDROLOGY_HISTORY_MESSAGES:]
    update: dict[str, Any] = {
        "draft": {},
        "plan": {},
        "stream_outputs": [],
    }
    if durable_messages != messages:
        update["messages"] = [
            RemoveMessage(id=REMOVE_ALL_MESSAGES),
            *durable_messages,
        ]
    return update


def request_question(state: HydrologySemanticQueryState) -> str:
    metadata = state.get("metadata") or {}
    return str(metadata.get("original_query") or state.get("query") or "").strip()


def main_response_format() -> dict[str, Any]:
    return {
        "type": "json_schema",
        "json_schema": {
            "name": "hydrology_main_agent_decision",
            "strict": True,
            "schema": MainAgentDecision.model_json_schema(),
        },
    }


def _decision_request(
    state: HydrologySemanticQueryState,
    services: HydrologySemanticQueryServices,
) -> dict[str, Any]:
    request = request_data(state, services.settings)
    return {
        "original_question": request["question"],
        "standalone_question": state.get("standalone_question") or request["question"],
        "task_budget": {
            "maximum": services.settings.max_query_rounds,
        },
    }


def _validate_decision(
    decision: MainAgentDecision,
    services: HydrologySemanticQueryServices,
) -> MainAgentDecision:
    known_playbooks = {
        playbook.name for playbook in services.business_playbooks or ()
    }
    if decision.matched_playbook is not None and decision.matched_playbook not in known_playbooks:
        raise ValueError("matched_playbook 不是已加载的业务知识文件")
    if len(decision.query_tasks) > services.settings.max_query_rounds:
        raise ValueError("查询任务数量超过任务预算")
    if decision.action == MainAgentAction.RESPOND:
        if decision.query_tasks or decision.report_sections:
            raise ValueError("直接回答不能包含查询任务或报告章节")
    planned_ids: list[str] = []
    for task in decision.query_tasks:
        if task.condition is not None:
            raise ValueError(f"任务 {task.task_id} 不支持 condition 条件分支")
        invalid_dependencies = set(task.depends_on) - set(planned_ids)
        if invalid_dependencies:
            raise ValueError(
                f"任务 {task.task_id} 引用了无效或尚未排在前面的依赖："
                f"{sorted(invalid_dependencies)}"
            )
        planned_ids.append(task.task_id)
    available_sources = set(planned_ids)
    for section in decision.report_sections:
        invalid_sources = set(section.source_task_ids) - available_sources
        if invalid_sources:
            raise ValueError(
                f"报告章节 {section.section_id} 引用了未知任务："
                f"{sorted(invalid_sources)}"
            )
    return decision


async def _invoke_decision(
    runtime,
    services: HydrologySemanticQueryServices,
    state: HydrologySemanticQueryState,
) -> MainAgentDecision:
    messages = [
        SystemMessage(content=main_agent_system_prompt(
            services.business_playbooks or ()
        )),
        HumanMessage(content=json.dumps(
            _decision_request(state, services),
            ensure_ascii=False,
            separators=(",", ":"),
            default=str,
        )),
    ]
    last_error: Exception | None = None
    for attempt in range(2):
        model = runtime.get_chat_model(streaming=False).bind(
            response_format=main_response_format(),
            extra_body={"enable_thinking": False},
            max_tokens=3200,
        )
        response = await model.ainvoke(messages, config={"callbacks": []})
        raw = stringify_message_content(response.content).strip()
        logger.info(
            "hydrology_semantic_query main decision response: attempt=%s response=%s",
            attempt + 1,
            safe_response_excerpt(raw),
        )
        try:
            decision = MainAgentDecision.model_validate_json(raw)
            return _validate_decision(
                decision,
                services,
            )
        except (ValidationError, ValueError) as exc:
            last_error = exc
            logger.warning(
                "hydrology_semantic_query main decision rejected: attempt=%s error=%s",
                attempt + 1,
                safe_response_excerpt(str(exc)),
            )
            if attempt == 0:
                messages.extend([
                    AIMessage(content=raw),
                    HumanMessage(content=(
                        "上一份决策未通过结构或上下文校验，请完整重写 JSON。"
                        f"校验错误：{safe_response_excerpt(str(exc))[:1000]}"
                    )),
                ])
    raise ValueError(f"主控 Agent 决策无效：{last_error}")


def _revision(
    decision: MainAgentDecision,
    number: int,
) -> ExecutionPlanRevision:
    return ExecutionPlanRevision(
        revision=number,
        action=decision.action,
        matched_playbook=decision.matched_playbook,
        query_tasks=decision.query_tasks,
        report_sections=decision.report_sections,
        summary=decision.summary,
    )


def _task_checklist_detail(
    summary: str,
    tasks: list[QueryTask],
    results: list[TaskExecutionResult],
    report_sections: list[ReportSectionRequirement] | None = None,
    report_status: TaskExecutionStatus | None = None,
) -> str:
    if not tasks and not report_sections:
        return summary
    statuses = {result.task.task_id: result.status for result in results}
    lines = []
    for task in tasks:
        status = statuses.get(task.task_id)
        if status == TaskExecutionStatus.SUCCESS:
            marker = "[ · ]"
        elif status is None:
            marker = "[  ]"
        else:
            marker = "[ × ]"
        lines.append(f"{marker} {task.objective}")
    if report_sections:
        if report_status == TaskExecutionStatus.SUCCESS:
            marker = "[ · ]"
        elif report_status is None:
            marker = "[  ]"
        else:
            marker = "[ × ]"
        if len(report_sections) == 1:
            title = report_sections[0].title
            objective = f"生成{title}" if title.endswith("报告") else f"生成{title}报告"
        else:
            objective = "生成水文语义查询综合分析报告"
        lines.append(f"{marker} {objective}")
    return f"{summary}\n\n{'\n'.join(lines)}"


def _decision_updates(
    state: HydrologySemanticQueryState,
    decision: MainAgentDecision,
    *,
    started: float,
    stage: str,
) -> dict[str, Any]:
    revisions = [_revision(decision, 1)]
    steps = list(state.get("steps", []))
    steps.append(build_step(
        stage,
        started,
        attempt=1,
        status=StepStatus.SUCCESS,
        metadata={
            "action": decision.action.value,
            "remaining_task_count": len(decision.query_tasks),
            "report_section_count": len(decision.report_sections),
            "matched_playbook": decision.matched_playbook,
        },
    ))
    return {
        "main_action": decision.action,
        "matched_playbook": decision.matched_playbook,
        "plan_revisions": revisions,
        "pending_tasks": decision.query_tasks,
        "report_sections": decision.report_sections,
        "direct_answer": decision.direct_answer or "",
        "steps": steps,
        "stage": stage,
        "stream_outputs": thought_output(
            "编排查询任务",
            _task_checklist_detail(
                decision.summary,
                decision.query_tasks,
                [],
                decision.report_sections,
            ),
        ),
    }


def make_main_plan_node(runtime, services: HydrologySemanticQueryServices):
    async def plan(state: HydrologySemanticQueryState) -> dict[str, Any]:
        started = time.perf_counter()
        try:
            decision = await _invoke_decision(
                runtime,
                services,
                state,
            )
            return _decision_updates(
                state,
                decision,
                started=started,
                stage="main_plan",
            )
        except Exception as exc:
            steps = list(state.get("steps", []))
            error = build_error(
                stage="main_plan",
                code=exc.__class__.__name__,
                kind=FailureKind.PLANNER,
                exc=exc,
            )
            steps.append(build_step(
                "main_plan",
                started,
                attempt=1,
                status=StepStatus.FAILED,
                summary=str(exc)[:1000],
            ))
            return {
                "main_action": MainAgentAction.RESPOND,
                "stage": "main_plan",
                "steps": steps,
                "error": error,
                "outcome": QueryOutcome.PLANNER_ERROR,
                "stream_outputs": thought_output(
                    "编排查询任务",
                    "主控 Agent 未能生成有效执行计划",
                ),
            }

    return plan


def _execution_plan(state: HydrologySemanticQueryState) -> list[QueryTask]:
    revisions = state.get("plan_revisions", [])
    return list(revisions[0].query_tasks) if revisions else []


def make_task_execute_node(
    compiled_query_graph,
    services: HydrologySemanticQueryServices,
):
    async def execute(
        state: HydrologySemanticQueryState,
        config: RunnableConfig,
    ) -> dict[str, Any]:
        pending = list(state.get("pending_tasks", []))
        if not pending:
            return {}
        started = time.perf_counter()
        task_results = list(state.get("task_results", []))
        steps = list(state.get("steps", []))
        if state.get("querying_blocked"):
            summary = "查询已被终止错误阻断，剩余任务不再执行。"
            task_results.extend(
                TaskExecutionResult(
                    task=task,
                    status=TaskExecutionStatus.SKIPPED,
                    summary=summary,
                )
                for task in pending
            )
            steps.append(build_step(
                "task_execute",
                started,
                attempt=max(1, state.get("dispatch_count", 0)),
                status=StepStatus.SKIPPED,
                summary=summary,
                metadata={"skipped_task_ids": [task.task_id for task in pending]},
            ))
            return {
                "pending_tasks": [],
                "current_task": pending[0],
                "task_results": task_results,
                "steps": steps,
                "stage": "task_execute",
                "stream_outputs": thought_output(
                    "执行查询任务",
                    _task_checklist_detail(
                        summary,
                        _execution_plan(state),
                        task_results,
                        state.get("report_sections", []),
                    ),
                ),
            }
        task = pending[0]
        result_by_id = {result.task.task_id: result for result in task_results}
        blocked_dependencies = [
            task_id
            for task_id in task.depends_on
            if task_id not in result_by_id
            or result_by_id[task_id].status != TaskExecutionStatus.SUCCESS
        ]
        if blocked_dependencies:
            summary = f"依赖任务未成功，已跳过：{', '.join(blocked_dependencies)}。"
            task_results.append(TaskExecutionResult(
                task=task,
                status=TaskExecutionStatus.SKIPPED,
                summary=summary,
            ))
            steps.append(build_step(
                "task_execute",
                started,
                attempt=max(1, state.get("dispatch_count", 0)),
                status=StepStatus.SKIPPED,
                summary=summary,
                metadata={
                    "task_id": task.task_id,
                    "blocked_dependencies": blocked_dependencies,
                },
            ))
            return {
                "pending_tasks": pending[1:],
                "current_task": task,
                "task_results": task_results,
                "steps": steps,
                "stage": "task_execute",
                "stream_outputs": thought_output(
                    "执行查询任务",
                    _task_checklist_detail(
                        f"{task.task_id}：{summary}",
                        _execution_plan(state),
                        task_results,
                        state.get("report_sections", []),
                    ),
                ),
            }
        parent_metadata = dict(state.get("metadata") or {})
        child_metadata = {
            "original_query": task.objective,
        }
        for key in ("catalog_mode", "catalog_metadata_filters"):
            if key in parent_metadata:
                child_metadata[key] = parent_metadata[key]
        child_state: QueryAgentState = {
            "query": task.objective,
            "standalone_question": task.objective,
            "current_task": task,
            "execution_context": QueryTaskExecutionContext(
                original_question=request_question(state),
                standalone_question=(
                    state.get("standalone_question") or request_question(state)
                ),
                plan=_execution_plan(state),
                current_task=task,
                completed_results=task_results,
            ),
            "metadata": child_metadata,
            "messages": [HumanMessage(content=task.objective)],
        }
        child_config: RunnableConfig = {
            **dict(config),
            "recursion_limit": 2 * services.settings.max_agent_iterations + 5,
        }
        child_output = await compiled_query_graph.ainvoke(
            child_state,
            config=child_config,
        )
        child_result = child_output.get("result")
        if child_result is not None and not hasattr(child_result, "outcome"):
            child_result = SemanticQueryResult.model_validate(child_result)
        record = (
            child_result.query_history[-1]
            if child_result and child_result.query_history
            else None
        )
        query_history = list(state.get("query_history", []))
        if record is not None:
            record = record.model_copy(update={
                "query_number": len(query_history) + 1,
                "task_id": task.task_id,
                "query_goal": task.objective,
            }, deep=True)
            query_history.append(record)
        if child_result is None:
            error = build_error(
                stage="query_dispatch",
                code="missing_query_agent_result",
                kind=FailureKind.SYSTEM,
                exc="查询子 Agent 未返回结果",
            )
            outcome = QueryOutcome.SYSTEM_ERROR
            attempts = 0
            child_warnings: list[str] = []
        else:
            error = child_result.error
            outcome = child_result.outcome
            attempts = child_result.attempts
            child_warnings = [
                warning
                for warning in child_result.warnings
                if warning not in services.startup_warnings
            ]
        if outcome == QueryOutcome.SUCCESS:
            status = TaskExecutionStatus.SUCCESS
            summary = f"任务成功，返回 {record.row_count if record else 0} 行数据。"
        elif outcome == QueryOutcome.NO_DATA:
            status = TaskExecutionStatus.NO_DATA
            summary = "任务执行成功，但未返回符合条件的数据。"
        else:
            status = TaskExecutionStatus.FAILED
            summary = f"任务失败：{error.code if error else outcome.value}。"
        task_result = TaskExecutionResult(
            task=task,
            status=status,
            outcome=outcome,
            query_record=record,
            error=error,
            warnings=child_warnings,
            attempts=attempts,
            summary=summary,
        )
        task_results.append(task_result)
        if child_result is not None:
            for step in child_result.steps:
                steps.append(step.model_copy(update={
                    "metadata": {**step.metadata, "task_id": task.task_id},
                }, deep=True))
        warnings = list(state.get("warnings", []))
        warnings.extend(
            f"任务 {task.task_id}：{warning}" for warning in child_warnings
        )
        if status == TaskExecutionStatus.FAILED:
            warnings.append(f"任务 {task.task_id} 未完成：{summary}")
        querying_blocked = bool(
            error
            and not error.retryable
            and error.kind in {FailureKind.EXECUTION, FailureKind.SYSTEM}
        )
        updates: dict[str, Any] = {
            "pending_tasks": pending[1:],
            "current_task": task,
            "task_results": task_results,
            "query_history": query_history,
            "dispatch_count": state.get("dispatch_count", 0) + 1,
            "attempts": state.get("attempts", 0) + attempts,
            "steps": steps,
            "warnings": warnings,
            "querying_blocked": state.get("querying_blocked", False) or querying_blocked,
            "stage": "task_execute",
            "stream_outputs": thought_output(
                "执行查询任务",
                _task_checklist_detail(
                    f"{task.task_id}：{summary}",
                    _execution_plan(state),
                    task_results,
                    state.get("report_sections", []),
                ),
            ),
        }
        if child_result is not None:
            updates.update({
                "semantic_query": child_result.semantic_query,
                "selected_models": child_result.selected_models,
                "compiled_sql": child_result.compiled_sql,
                "compiled_params": child_result.compiled_params,
                "catalog_mode": child_result.catalog_mode,
                "retrieval_trace": child_result.retrieval_trace,
                "error": error,
            })
            if outcome == QueryOutcome.SUCCESS and record is not None:
                updates["columns"] = record.columns
                updates["rows"] = record.rows
        return updates

    return execute


__all__ = [
    "StandaloneQuestion",
    "main_response_format",
    "make_finalize_node",
    "make_initialize_node",
    "make_main_plan_node",
    "make_report_dispatch_node",
    "make_task_execute_node",
    "parse_standalone_question",
    "request_question",
    "response_format",
]
