from __future__ import annotations

import time
from collections.abc import Collection, Sequence
from contextlib import suppress
from typing import Any

from langchain_core.messages import (
    AIMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
)
from langchain_core.tools import BaseTool
from pydantic import BaseModel, ConfigDict, field_validator

from app.agents.messages import stringify_message_content
from app.agents.streaming import chain_of_thought_output, llm_stream_output

from .client import CubeClientError, catalog_from_meta
from .config import HYDROLOGY_SEMANTIC_QUERY_ID
from .models import FailureKind, QueryOutcome, SemanticQueryResult, StepStatus
from .prompts import build_messages, react_system_prompt
from .report import (
    REPORT_FAILURE_WARNING,
    build_result_outputs,
    generate_report,
)
from .runtime import (
    HydrologySemanticQueryServices,
    HydrologySemanticQueryState,
    RequestData,
    build_error,
    build_step,
    outcome_for_error,
    request_data,
    safe_response_excerpt,
    thought_output,
)
from .tool_call_parser import (
    contains_internal_protocol,
    parse_hydrology_tool_calls,
)

INTERNAL_PROTOCOL_WARNING = "模型返回了内部工具协议，已使用安全结果文案。"


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

def _reset() -> dict[str, Any]:
    return {
        "answer": "",
        "standalone_question": None,
        "catalog": None,
        "full_catalog": None,
        "catalog_mode": None,
        "semantic_context": None,
        "retrieval_trace": None,
        "selected_models": [],
        "semantic_query": None,
        "previous_query": None,
        "cube_query": None,
        "compiled_sql": None,
        "compiled_params": [],
        "cube_response": None,
        "columns": [],
        "rows": [],
        "steps": [],
        "warnings": [],
        "attempts": 0,
        "max_rows": 0,
        "stage": "catalog_prepare",
        "error": None,
        "outcome": None,
        "result": None,
        "agent_answer": "",
        "last_tool_terminal": False,
        "search_count": 0,
    }


def make_catalog_prepare_node(services: HydrologySemanticQueryServices):
    async def prepare_catalog(state: HydrologySemanticQueryState) -> dict[str, Any]:
        reset = _reset()
        started = time.perf_counter()
        try:
            request = request_data(state, services.settings)
            meta = await services.client.get_meta()
            full_catalog = catalog_from_meta(meta)
            step = build_step(
                "catalog_prepare",
                started,
                attempt=1,
                status=StepStatus.SUCCESS,
                metadata={
                    "total_model_count": len(full_catalog.models),
                    "view_count": sum(
                        model.model_type == "view"
                        for model in full_catalog.models.values()
                    ),
                    "cube_count": sum(
                        model.model_type == "cube"
                        for model in full_catalog.models.values()
                    ),
                },
            )
            return {
                **reset,
                "full_catalog": full_catalog,
                "steps": [step],
                "warnings": list(services.startup_warnings),
                "max_rows": request["max_rows"],
                "stream_outputs": thought_output(
                    "准备语义目录",
                    f"已加载 {len(full_catalog.models)} 个受治理的公开 View/Cube",
                ),
            }
        except Exception as exc:
            code = exc.code if isinstance(exc, CubeClientError) else exc.__class__.__name__
            status = exc.status_code if isinstance(exc, CubeClientError) else None
            step = build_step(
                "catalog_prepare",
                started,
                attempt=1,
                status=StepStatus.FAILED,
                summary=str(exc)[:1000],
            )
            return {
                **reset,
                "steps": [step],
                "stage": "catalog_prepare",
                "error": build_error(
                    stage="catalog_prepare",
                    code=code,
                    kind=FailureKind.SYSTEM,
                    exc=exc,
                    status_code=status,
                ),
                "outcome": QueryOutcome.SYSTEM_ERROR,
                "stream_outputs": thought_output("加载语义模型", "水文语义模型加载失败"),
            }

    return prepare_catalog


def make_question_contextualization_node(
    runtime,
    services: HydrologySemanticQueryServices,
):
    async def contextualize_question(
        state: HydrologySemanticQueryState,
    ) -> dict[str, Any]:
        request = request_data(state, services.settings)
        question = request["question"]
        if not request["conversation_context"]:
            return {"standalone_question": question}
        started = time.perf_counter()
        steps = list(state["steps"])
        warnings = list(state["warnings"])
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
                "多轮问题改写失败，已使用原问题检索。"
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

    return contextualize_question


def make_initialize_node(runtime, services: HydrologySemanticQueryServices):
    async def initialize(state: HydrologySemanticQueryState) -> dict[str, Any]:
        prepared = await make_catalog_prepare_node(services)(state)
        working: HydrologySemanticQueryState = dict(state)
        working.update(prepared)
        metadata = dict(state.get("metadata") or {})
        metadata["agent_iterations"] = 0
        initialized = {
            **prepared,
            "metadata": metadata,
            "agent_answer": "",
            "last_tool_terminal": False,
            "search_count": 0,
        }
        if prepared.get("outcome") is not None:
            return initialized
        contextualized = await make_question_contextualization_node(
            runtime, services
        )(working)
        initialized.update(contextualized)
        return initialized

    return initialize

def make_react_agent_node(
    runtime,
    services: HydrologySemanticQueryServices,
    tools: Sequence[BaseTool],
    tool_names: Collection[str],
):
    async def agent(state: HydrologySemanticQueryState) -> dict[str, Any]:
        metadata = dict(state.get("metadata") or {})
        iteration = int(metadata.get("agent_iterations", 0))
        messages = list(state.get("messages", []))
        tool_outputs = (
            list(state.get("stream_outputs", []))
            if messages and isinstance(messages[-1], ToolMessage)
            else []
        )
        final_only = bool(
            state.get("last_tool_terminal")
            or iteration >= services.settings.max_agent_iterations - 1
        )
        messages = [
            SystemMessage(content=react_system_prompt(
                state,
                services,
                final_only=final_only,
            )),
            *list(state.get("messages", [])),
        ]
        if not state.get("messages"):
            messages.append(
                HumanMessage(content=request_data(state, services.settings)["question"])
            )
        model = runtime.get_chat_model(streaming=False)
        if not final_only:
            bind_tools = getattr(model, "bind_tools", None)
            if callable(bind_tools):
                try:
                    model = bind_tools(tools, parallel_tool_calls=False)
                except (AttributeError, NotImplementedError, TypeError):
                    pass
        try:
            response = await model.ainvoke(messages, config={"callbacks": []})
        except Exception as exc:
            if state.get("outcome") in {QueryOutcome.SUCCESS, QueryOutcome.NO_DATA}:
                warnings = list(state.get("warnings", []))
                warnings.append("Agent 最终回答生成失败，已保留语义查询结果。")
                metadata["agent_iterations"] = iteration + 1
                return {
                    "messages": [AIMessage(content="")],
                    "metadata": metadata,
                    "agent_answer": "",
                    "warnings": warnings,
                    "last_tool_terminal": True,
                    "stream_outputs": tool_outputs + [chain_of_thought_output(
                        step_type="analysis",
                        text="水文语义查询",
                        detail="最终回答生成失败，已保留查询结果",
                        intent_id=HYDROLOGY_SEMANTIC_QUERY_ID,
                    )],
                }
            error = build_error(
                stage="agent_decision",
                code=exc.__class__.__name__,
                kind=FailureKind.SYSTEM,
                exc=exc,
            )
            metadata["agent_iterations"] = iteration + 1
            return {
                "messages": [AIMessage(content="")],
                "metadata": metadata,
                "agent_answer": "",
                "stage": "agent_decision",
                "error": error,
                "outcome": QueryOutcome.SYSTEM_ERROR,
                "last_tool_terminal": True,
                "stream_outputs": tool_outputs + [chain_of_thought_output(
                    step_type="analysis",
                    text="水文语义查询",
                    detail="Agent 决策模型调用失败",
                    intent_id=HYDROLOGY_SEMANTIC_QUERY_ID,
                )],
            }
        response_content = stringify_message_content(response.content).strip()
        allowed_tool_names = set(tool_names)
        invalid_native_tool_call = any(
            str(tool_call.get("name", "")) not in allowed_tool_names
            for tool_call in response.tool_calls
        )
        parsed = []
        if not response.tool_calls and response_content:
            parsed = parse_hydrology_tool_calls(response_content, allowed_tool_names)
        protocol_rejected = bool(
            (final_only and (response.tool_calls or parsed))
            or invalid_native_tool_call
            or (not parsed and contains_internal_protocol(response_content))
        )
        warnings = list(state.get("warnings", []))
        if protocol_rejected:
            if INTERNAL_PROTOCOL_WARNING not in warnings:
                warnings.append(INTERNAL_PROTOCOL_WARNING)
            response = AIMessage(content="")
        elif parsed:
            response.content = ""
            response.tool_calls = parsed
        if len(response.tool_calls) > 1:
            response.tool_calls = response.tool_calls[:1]
        metadata["agent_iterations"] = iteration + 1
        content = stringify_message_content(response.content).strip()
        answer = content if not response.tool_calls else ""
        return {
            "messages": [response],
            "metadata": metadata,
            "agent_answer": answer,
            "warnings": warnings,
            "stream_outputs": tool_outputs + [chain_of_thought_output(
                step_type="analysis",
                text="水文语义查询",
                detail=(
                    f"第 {iteration + 1} 轮已选择工具 {response.tool_calls[0]['name']}"
                    if response.tool_calls
                    else "已拦截内部工具协议，准备使用安全结果文案"
                    if protocol_rejected
                    else "已根据当前 Observation 形成最终回答"
                ),
                intent_id=HYDROLOGY_SEMANTIC_QUERY_ID,
            )],
        }

    return agent

def make_react_finalize_node(runtime, services: HydrologySemanticQueryServices):
    async def finalize_result(state: HydrologySemanticQueryState) -> dict[str, Any]:
        started = time.perf_counter()
        steps = list(state.get("steps", []))
        warnings = list(state.get("warnings", []))
        answer = state.get("agent_answer", "").strip()
        if contains_internal_protocol(answer):
            answer = ""
            if INTERNAL_PROTOCOL_WARNING not in warnings:
                warnings.append(INTERNAL_PROTOCOL_WARNING)
        has_query_result = bool(
            state.get("attempts", 0)
            or state.get("semantic_query") is not None
            or state.get("outcome") is not None
            or state.get("error") is not None
        )
        attempt = max(1, state.get("attempts", 0))
        request: RequestData | None = None
        with suppress(Exception):
            request = request_data(state, services.settings)
        if not has_query_result:
            if not answer:
                answer = "请提供需要查询的水文数据、范围或条件。"
            steps.append(build_step(
                "result_finalize",
                started,
                attempt=attempt,
                status=StepStatus.SUCCESS,
            ))
            return {
                "answer": answer,
                "result": None,
                "steps": steps,
                "warnings": warnings,
                "stream_outputs": [llm_stream_output(text=answer)],
            }
        outcome = state.get("outcome") or outcome_for_error(state.get("error"))
        success = outcome == QueryOutcome.SUCCESS
        semantic_query = state.get("semantic_query") or state.get("previous_query")
        if outcome == QueryOutcome.SUCCESS:
            answer = f"查询完成，共返回 {len(state.get('rows', []))} 行数据。"
        elif outcome == QueryOutcome.NO_DATA:
            answer = "未查询到符合当前条件的数据。"
        elif outcome == QueryOutcome.PLANNER_ERROR:
            answer = "当前问题暂时无法转换为有效的数据查询，请调整查询条件后重试。"
        elif outcome == QueryOutcome.EXECUTION_ERROR:
            answer = "当前数据查询暂时未能完成，请稍后重试。"
        else:
            answer = "当前查询暂时无法完成。"
        result = SemanticQueryResult(
            outcome=outcome,
            semantic_query=semantic_query,
            columns=state.get("columns", []) if success else [],
            rows=state.get("rows", []) if success else [],
            row_count=len(state.get("rows", [])) if success else 0,
            attempts=state.get("attempts", 0),
            compiled_sql=state.get("compiled_sql"),
            compiled_params=state.get("compiled_params", []),
            catalog_mode=state.get("catalog_mode"),
            query_mode=semantic_query.query_mode if semantic_query else None,
            selected_models=state.get("selected_models", []),
            retrieval_trace=state.get("retrieval_trace"),
            warnings=warnings,
            steps=steps,
            error=state.get("error"),
        )
        if success and request:
            try:
                answer = await generate_report(runtime, request["question"], result)
            except Exception:
                warnings.append(REPORT_FAILURE_WARNING)
                result.warnings = warnings
        steps.append(build_step(
            "result_finalize",
            started,
            attempt=attempt,
            status=StepStatus.SUCCESS,
        ))
        result.steps = steps
        outputs = build_result_outputs(
            result,
            answer=answer,
            question=request["question"] if request else str(state.get("query") or ""),
        )
        metadata = dict(state.get("metadata") or {})
        metadata["hydrology_semantic_query_result"] = result.model_dump(
            mode="json", by_alias=True, exclude_none=True
        )
        return {
            "answer": answer,
            "result": result,
            "steps": steps,
            "warnings": warnings,
            "metadata": metadata,
            "stream_outputs": outputs,
        }

    return finalize_result
