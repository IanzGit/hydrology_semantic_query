from __future__ import annotations

import json
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
    ReportAnalysisMethod,
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
            outcome = QueryOutcome.SUCCESS
            error = None
            top = successful[-1].query_record
            answer = (
                f"查询计划已完成，共派发 {state.get('dispatch_count', 0)} 个任务，"
                f"获得 {len(successful)} 个有效结果。"
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
            columns=top.columns if top and outcome == QueryOutcome.SUCCESS else [],
            rows=top.rows if top and outcome == QueryOutcome.SUCCESS else [],
            row_count=top.row_count if top and outcome == QueryOutcome.SUCCESS else 0,
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
        outputs = [] if outcome == QueryOutcome.SUCCESS else [llm_stream_output(text=answer)]
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
        if result is None or result.outcome != QueryOutcome.SUCCESS:
            answer = state.get("answer", "")
            return {
                "stream_outputs": [llm_stream_output(text=answer)] if answer else [],
            }
        report_task = ReportTask(
            original_question=request_question(state),
            sections=state.get("report_sections", []),
            task_results=state.get("task_results", []),
        )
        report_state: ReportAgentState = {
            "report_task": report_task,
            "result": result,
            "warnings": list(state.get("warnings", [])),
            "steps": [],
        }
        try:
            output = await compiled_report_graph.ainvoke(report_state, config=config)
        except Exception as exc:
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
            return {
                "answer": answer,
                "result": result,
                "warnings": warnings,
                "steps": steps,
                "metadata": metadata,
                "stream_outputs": (
                    [llm_stream_output(text=answer)] if answer else []
                ),
            }
        report = output["report"]
        warnings = list(output.get("warnings", []))
        steps = [*state.get("steps", []), *output.get("steps", [])]
        result = result.model_copy(update={
            "presentation": report,
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
            "answer": output["answer"],
            "result": result,
            "warnings": warnings,
            "steps": steps,
            "metadata": metadata,
            "stream_outputs": output["stream_outputs"],
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


def _task_result_payload(result: TaskExecutionResult) -> dict[str, Any]:
    record = result.query_record
    return {
        "task_id": result.task.task_id,
        "objective": result.task.objective,
        "status": result.status.value,
        "outcome": result.outcome.value if result.outcome else None,
        "summary": result.summary,
        "row_count": record.row_count if record else 0,
        "columns": (
            [column.model_dump(mode="json") for column in record.columns]
            if record
            else []
        ),
        "rows": record.rows if record else [],
        "error": (
            {
                "stage": result.error.stage,
                "code": result.error.code,
                "kind": result.error.kind.value,
                "retryable": result.error.retryable,
            }
            if result.error
            else None
        ),
    }


def _decision_request(
    state: HydrologySemanticQueryState,
    services: HydrologySemanticQueryServices,
    *,
    phase: str,
) -> dict[str, Any]:
    request = request_data(state, services.settings)
    return {
        "phase": phase,
        "original_question": request["question"],
        "standalone_question": state.get("standalone_question") or request["question"],
        "matched_playbook": state.get("matched_playbook"),
        "current_remaining_tasks": [
            task.model_dump(mode="json") for task in state.get("pending_tasks", [])
        ],
        "current_report_sections": [
            section.model_dump(mode="json")
            for section in state.get("report_sections", [])
        ],
        "completed_task_results": [
            _task_result_payload(result)
            for result in state.get("task_results", [])
        ],
        "querying_blocked": state.get("querying_blocked", False),
        "task_budget": {
            "maximum": services.settings.max_query_rounds,
            "dispatched": state.get("dispatch_count", 0),
            "remaining": max(
                0,
                services.settings.max_query_rounds - state.get("dispatch_count", 0),
            ),
        },
    }


def _validate_decision(
    decision: MainAgentDecision,
    state: HydrologySemanticQueryState,
    services: HydrologySemanticQueryServices,
    *,
    phase: str,
) -> MainAgentDecision:
    known_playbooks = {
        playbook.name for playbook in services.business_playbooks or ()
    }
    if decision.matched_playbook is not None and decision.matched_playbook not in known_playbooks:
        raise ValueError("matched_playbook 不是已加载的业务知识文件")
    revisions = state.get("plan_revisions", [])
    if phase == "replan" and revisions:
        if decision.matched_playbook != state.get("matched_playbook"):
            raise ValueError("Replan 不得改变已选业务知识文件")
    completed_ids = {result.task.task_id for result in state.get("task_results", [])}
    remaining_budget = max(
        0,
        services.settings.max_query_rounds - state.get("dispatch_count", 0),
    )
    if len(decision.query_tasks) > remaining_budget:
        raise ValueError("查询任务数量超过剩余任务预算")
    if decision.action == MainAgentAction.QUERY and state.get("querying_blocked"):
        raise ValueError("查询已被终止错误阻断，不能继续派发任务")
    if (
        phase == "initial"
        and decision.action == MainAgentAction.REPORT
        and not state.get("task_results")
    ):
        raise ValueError("初始规划不能在没有任务结果时直接生成报告")
    if decision.action == MainAgentAction.RESPOND:
        if state.get("task_results"):
            raise ValueError("已有查询任务结果时不能改为直接回答")
        if decision.query_tasks or decision.report_sections:
            raise ValueError("直接回答不能包含查询任务或报告章节")
    dependency_ids = {
        result.task.task_id
        for result in state.get("task_results", [])
        if result.status != TaskExecutionStatus.SKIPPED
    }
    planned_ids: list[str] = []
    for task in decision.query_tasks:
        if task.task_id in completed_ids:
            raise ValueError(f"任务 ID {task.task_id} 已执行，不能重复使用")
        allowed_dependencies = dependency_ids | set(planned_ids)
        invalid_dependencies = set(task.depends_on) - allowed_dependencies
        if invalid_dependencies:
            raise ValueError(
                f"任务 {task.task_id} 引用了无效或尚未排在前面的依赖："
                f"{sorted(invalid_dependencies)}"
            )
        planned_ids.append(task.task_id)
    available_sources = completed_ids | set(planned_ids)
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
    *,
    phase: str,
) -> MainAgentDecision:
    messages = [
        SystemMessage(content=main_agent_system_prompt(
            services.business_playbooks or ()
        )),
        HumanMessage(content=json.dumps(
            _decision_request(state, services, phase=phase),
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
        try:
            decision = MainAgentDecision.model_validate_json(raw)
            return _validate_decision(
                decision,
                state,
                services,
                phase=phase,
            )
        except (ValidationError, ValueError) as exc:
            last_error = exc
            if attempt == 0:
                messages.append(HumanMessage(content=(
                    "上一份决策未通过结构或上下文校验，请完整重写 JSON。"
                    f"校验错误：{safe_response_excerpt(str(exc))[:1000]}"
                )))
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


def _decision_updates(
    state: HydrologySemanticQueryState,
    decision: MainAgentDecision,
    *,
    started: float,
    stage: str,
) -> dict[str, Any]:
    revisions = list(state.get("plan_revisions", []))
    revisions.append(_revision(decision, len(revisions) + 1))
    steps = list(state.get("steps", []))
    steps.append(build_step(
        stage,
        started,
        attempt=len(revisions),
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
            "编排查询任务" if stage == "main_plan" else "调整执行计划",
            decision.summary,
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
                phase="initial",
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
                "main_action": MainAgentAction.REPORT,
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


def _fallback_report_sections(
    state: HydrologySemanticQueryState,
) -> list[ReportSectionRequirement]:
    existing = list(state.get("report_sections", []))
    if existing:
        return existing
    task_ids = [result.task.task_id for result in state.get("task_results", [])]
    return [
        ReportSectionRequirement(
            section_id="overview",
            title="总体情况",
            objective="概括查询范围、数据结果和主要事实。",
            source_task_ids=task_ids,
            analysis_methods=[ReportAnalysisMethod.OVERVIEW],
        ),
        ReportSectionRequirement(
            section_id="conclusion",
            title="综合结论",
            objective="综合已有证据并说明局限。",
            source_task_ids=task_ids,
            analysis_methods=[ReportAnalysisMethod.CONCLUSION],
        ),
    ]


def _fallback_replan_decision(
    state: HydrologySemanticQueryState,
) -> MainAgentDecision:
    pending = [] if state.get("querying_blocked") else list(state.get("pending_tasks", []))
    action = MainAgentAction.QUERY if pending else MainAgentAction.REPORT
    return MainAgentDecision(
        action=action,
        matched_playbook=state.get("matched_playbook"),
        query_tasks=pending,
        report_sections=_fallback_report_sections(state),
        summary=(
            "Replan 失败，继续执行原剩余计划。"
            if pending
            else "Replan 失败，使用当前结果进入报告阶段。"
        ),
    )


def _append_skipped_tasks(
    state: HydrologySemanticQueryState,
    decision: MainAgentDecision,
) -> list[TaskExecutionResult]:
    results = list(state.get("task_results", []))
    retained = {task.task_id for task in decision.query_tasks}
    for task in state.get("pending_tasks", []):
        if task.task_id not in retained:
            results.append(TaskExecutionResult(
                task=task,
                status=TaskExecutionStatus.SKIPPED,
                summary=f"Replan 已取消该任务：{decision.summary}",
            ))
    return results


def make_main_replan_node(runtime, services: HydrologySemanticQueryServices):
    async def replan(state: HydrologySemanticQueryState) -> dict[str, Any]:
        started = time.perf_counter()
        warnings = list(state.get("warnings", []))
        try:
            decision = await _invoke_decision(
                runtime,
                services,
                state,
                phase="replan",
            )
        except Exception as exc:
            decision = _fallback_replan_decision(state)
            warnings.append(
                "主控 Agent Replan 失败，已使用安全回退计划。"
                f"原因：{safe_response_excerpt(str(exc))[:200]}"
            )
        updates = _decision_updates(
            state,
            decision,
            started=started,
            stage="main_replan",
        )
        updates["task_results"] = _append_skipped_tasks(state, decision)
        updates["warnings"] = warnings
        return updates

    return replan


def make_query_dispatch_node(
    compiled_query_graph,
    services: HydrologySemanticQueryServices,
):
    async def dispatch(
        state: HydrologySemanticQueryState,
        config: RunnableConfig,
    ) -> dict[str, Any]:
        pending = list(state.get("pending_tasks", []))
        if not pending:
            return {
                "main_action": MainAgentAction.REPORT,
                "report_sections": _fallback_report_sections(state),
            }
        task = pending[0]
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
        task_results = [*state.get("task_results", []), task_result]
        steps = list(state.get("steps", []))
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
            "stage": "query_dispatch",
            "stream_outputs": thought_output(
                "执行查询任务",
                f"{task.task_id}：{summary}",
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

    return dispatch


__all__ = [
    "StandaloneQuestion",
    "main_response_format",
    "make_finalize_node",
    "make_initialize_node",
    "make_main_plan_node",
    "make_main_replan_node",
    "make_query_dispatch_node",
    "make_report_dispatch_node",
    "parse_standalone_question",
    "request_question",
    "response_format",
]
