from __future__ import annotations

import json
from typing import Any

from langchain.tools import ToolRuntime
from langchain_core.messages import ToolMessage

from app.agents.scenarios.hydrology_semantic_query.models import SemanticQueryError
from app.agents.scenarios.hydrology_semantic_query.runtime import (
    outcome_for_error,
    safe_response_excerpt,
)


def dump_payload(payload: dict[str, Any]) -> str:
    return json.dumps(payload, ensure_ascii=False, separators=(",", ":"), default=str)


def tool_message(
    runtime: ToolRuntime,
    name: str,
    payload: dict[str, Any],
    *,
    error: bool = False,
) -> ToolMessage:
    return ToolMessage(
        content=dump_payload(payload),
        name=name,
        tool_call_id=runtime.tool_call_id or "call_0",
        status="error" if error else "success",
    )


def error_payload(error: SemanticQueryError, *, terminal: bool) -> dict[str, Any]:
    return {
        "ok": False,
        "kind": "error",
        "stage": error.stage,
        "outcome": outcome_for_error(error).value,
        "error": {
            "code": error.code,
            "message": safe_response_excerpt(error.internal_message),
            "details": error.internal_details,
            "retryable": error.retryable,
            "terminal": terminal,
        },
    }


def tool_input_error(exc: Exception) -> str:
    return dump_payload({
        "ok": False,
        "kind": "error",
        "stage": "tool_input_validation",
        "outcome": "planner_error",
        "error": {
            "code": exc.__class__.__name__,
            "message": safe_response_excerpt(str(exc)),
            "details": {},
            "retryable": True,
            "terminal": False,
        },
    })

