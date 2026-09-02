from __future__ import annotations

import time
from collections.abc import Collection, Sequence
from typing import Any

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.tools import BaseTool

from app.agents.messages import stringify_message_content
from app.agents.streaming import chain_of_thought_output

from ..contracts import FailureKind, QueryOutcome, SemanticQueryResult, StepStatus
from .client import CubeClientError, catalog_from_meta
from .config import HYDROLOGY_SEMANTIC_QUERY_ID
from .prompts import query_agent_system_prompt
from .runtime import (
    HydrologySemanticQueryServices,
    build_error,
    build_step,
    outcome_for_error,
    request_data,
    thought_output,
)
from .state import QueryAgentState
from .tool_call_parser import contains_internal_protocol, parse_hydrology_tool_calls

INTERNAL_PROTOCOL_WARNING = "模型返回了内部工具协议，已使用安全查询结果。"


def make_query_prepare_node(services: HydrologySemanticQueryServices):
    async def prepare(state: QueryAgentState) -> dict[str, Any]:
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
                metadata={"total_model_count": len(full_catalog.models)},
            )
            return {
                "full_catalog": full_catalog,
                "catalog": None,
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
                "steps": [step],
                "warnings": list(services.startup_warnings),
                "attempts": 0,
                "max_rows": request["max_rows"],
                "stage": "catalog_prepare",
                "error": None,
                "outcome": None,
                "result": None,
                "agent_answer": "",
                "last_tool_terminal": False,
                "search_count": 0,
                "query_count": 0,
                "query_history": [],
                "stream_outputs": thought_output(
                    "准备语义目录",
                    f"已加载 {len(full_catalog.models)} 个受治理的公开 View/Cube",
                ),
            }
        except Exception as exc:
            code = exc.code if isinstance(exc, CubeClientError) else exc.__class__.__name__
            status = exc.status_code if isinstance(exc, CubeClientError) else None
            error = build_error(
                stage="catalog_prepare",
                code=code,
                kind=FailureKind.SYSTEM,
                exc=exc,
                status_code=status,
            )
            step = build_step(
                "catalog_prepare",
                started,
                attempt=1,
                status=StepStatus.FAILED,
                summary=str(exc)[:1000],
            )
            return {
                "steps": [step],
                "warnings": list(services.startup_warnings),
                "stage": "catalog_prepare",
                "error": error,
                "outcome": QueryOutcome.SYSTEM_ERROR,
                "last_tool_terminal": True,
                "stream_outputs": thought_output(
                    "加载语义模型",
                    "水文语义模型加载失败",
                ),
            }

    return prepare


def make_query_react_node(
    runtime,
    services: HydrologySemanticQueryServices,
    tools: Sequence[BaseTool],
    tool_names: Collection[str],
):
    async def agent(state: QueryAgentState) -> dict[str, Any]:
        metadata = dict(state.get("metadata") or {})
        iteration = int(metadata.get("query_agent_iterations", 0))
        existing_messages = list(state.get("messages", []))
        tool_outputs = (
            list(state.get("stream_outputs", []))
            if existing_messages and isinstance(existing_messages[-1], ToolMessage)
            else []
        )
        final_only = bool(
            state.get("last_tool_terminal")
            or state.get("query_count", 0) >= 1
            or iteration >= services.settings.max_agent_iterations - 1
        )
        messages = [
            SystemMessage(content=query_agent_system_prompt(
                state,
                services,
                final_only=final_only,
            )),
            *existing_messages,
        ]
        if not existing_messages:
            task = state.get("current_task")
            messages.append(HumanMessage(content=task.objective if task else state["query"]))
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
            error = build_error(
                stage="query_agent_decision",
                code=exc.__class__.__name__,
                kind=FailureKind.SYSTEM,
                exc=exc,
            )
            metadata["query_agent_iterations"] = iteration + 1
            return {
                "messages": [AIMessage(content="")],
                "metadata": metadata,
                "agent_answer": "",
                "stage": "query_agent_decision",
                "error": error,
                "outcome": QueryOutcome.SYSTEM_ERROR,
                "last_tool_terminal": True,
                "stream_outputs": tool_outputs + [chain_of_thought_output(
                    step_type="analysis",
                    text="执行查询任务",
                    detail="查询子 Agent 决策模型调用失败",
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
        metadata["query_agent_iterations"] = iteration + 1
        answer = stringify_message_content(response.content).strip()
        return {
            "messages": [response],
            "metadata": metadata,
            "agent_answer": answer if not response.tool_calls else "",
            "warnings": warnings,
            "stream_outputs": tool_outputs + [chain_of_thought_output(
                step_type="analysis",
                text="执行查询任务",
                detail=(
                    f"第 {iteration + 1} 轮已选择工具 {response.tool_calls[0]['name']}"
                    if response.tool_calls
                    else "已结束查询任务"
                ),
                intent_id=HYDROLOGY_SEMANTIC_QUERY_ID,
            )],
        }

    return agent


def make_query_finalize_node():
    async def finalize(state: QueryAgentState) -> dict[str, Any]:
        started = time.perf_counter()
        steps = list(state.get("steps", []))
        warnings = list(state.get("warnings", []))
        outcome = state.get("outcome")
        error = state.get("error")
        if outcome is None:
            if error is None:
                error = build_error(
                    stage="query_agent_finalize",
                    code="query_not_executed",
                    kind=FailureKind.PLANNER,
                    exc="查询子 Agent 未形成可执行的 SemanticQuery",
                )
            outcome = outcome_for_error(error)
        success = outcome == QueryOutcome.SUCCESS
        history = list(state.get("query_history", []))
        top = history[-1] if history else None
        steps.append(build_step(
            "query_agent_finalize",
            started,
            attempt=max(1, state.get("attempts", 0)),
            status=StepStatus.SUCCESS,
        ))
        result = SemanticQueryResult(
            outcome=outcome,
            semantic_query=(
                top.semantic_query
                if top is not None
                else state.get("semantic_query") or state.get("previous_query")
            ),
            columns=state.get("columns", []) if success else [],
            rows=state.get("rows", []) if success else [],
            row_count=len(state.get("rows", [])) if success else 0,
            attempts=state.get("attempts", 0),
            query_count=len(history),
            query_history=history,
            compiled_sql=state.get("compiled_sql"),
            compiled_params=state.get("compiled_params", []),
            catalog_mode=state.get("catalog_mode"),
            query_mode=(
                state["semantic_query"].query_mode
                if state.get("semantic_query") is not None
                else None
            ),
            selected_models=state.get("selected_models", []),
            retrieval_trace=state.get("retrieval_trace"),
            warnings=warnings,
            steps=steps,
            error=error,
        )
        return {
            "result": result,
            "steps": steps,
            "warnings": warnings,
            "outcome": outcome,
            "error": error,
        }

    return finalize


__all__ = [
    "make_query_finalize_node",
    "make_query_prepare_node",
    "make_query_react_node",
]
