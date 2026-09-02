from __future__ import annotations

import json
import re
from collections.abc import Collection
from typing import Any

from app.agents.tools.parser import parse_tool_calls_from_text as parse_generic_tool_calls

_INVALID = object()
_TOOL_CALL_BLOCK = re.compile(
    r"<tool_call\b[^>]*>([\s\S]*?)</tool_call\s*>",
    re.IGNORECASE,
)
_FUNCTION_BLOCK = re.compile(
    r"<function\s*=\s*([a-zA-Z_][a-zA-Z0-9_]*)\s*>([\s\S]*?)</function\s*>",
    re.IGNORECASE,
)
_PARAMETER_BLOCK = re.compile(
    r"<parameter\s*=\s*([a-zA-Z_][a-zA-Z0-9_]*)\s*>([\s\S]*?)</parameter\s*>",
    re.IGNORECASE,
)
_ACTION_BLOCK = re.compile(
    r"Action:\s*(\w+)\s*\n\s*Action Input:\s*([\s\S]*?)"
    r"(?=\n\s*Action:\s*\w+\s*\n\s*Action Input:|\Z)",
    re.IGNORECASE,
)
_INTERNAL_PROTOCOL_MARKERS = re.compile(
    r"<\s*/?\s*tool_call\b|<\s*/?\s*function\s*=|"
    r"<\s*/?\s*parameter\s*=|(?im:^\s*Action\s*:)|"
    r"(?im:^\s*Action Input\s*:)|[\"']tool_calls[\"']\s*:|"
    r"(?im:^\s*Observation\s*:)|"
    r"[\"']semantic_query[\"']\s*:|[\"']query_mode[\"']\s*:|"
    r"[\"']rows_truncated[\"']\s*:|"
    r"[\"']kind[\"']\s*:\s*[\"'](?:semantic_query_result|error)[\"']",
    re.IGNORECASE,
)
_SQL_BLOCK = re.compile(
    r"(?is)(?:^|\n)\s*(?:```sql\s*)?(?:SELECT|WITH)\s+.+(?:```\s*)?$"
)


def _strip_fence(raw: str) -> str:
    payload = raw.strip()
    if not payload.startswith("```"):
        return payload
    payload = re.sub(r"^```(?:json)?\s*", "", payload, flags=re.IGNORECASE)
    return re.sub(r"\s*```$", "", payload).strip()


def _parse_json_value(raw: str) -> Any:
    payload = _strip_fence(raw)
    if not payload:
        return _INVALID
    try:
        return json.loads(payload)
    except json.JSONDecodeError:
        return _INVALID


def _parse_mapping(raw: str) -> dict[str, Any] | None:
    if not _strip_fence(raw):
        return {}
    value = _parse_json_value(raw)
    return value if isinstance(value, dict) else None


def _parse_xml_tool_calls(
    content: str,
    allowed_names: Collection[str],
) -> list[dict[str, Any]]:
    tool_calls: list[dict[str, Any]] = []
    for idx, tool_match in enumerate(_TOOL_CALL_BLOCK.finditer(content)):
        function_match = _FUNCTION_BLOCK.fullmatch(tool_match.group(1).strip())
        if function_match is None:
            continue
        name = function_match.group(1).strip()
        if name not in allowed_names:
            continue
        args: dict[str, Any] = {}
        function_body = function_match.group(2).strip()
        parameter_matches = list(_PARAMETER_BLOCK.finditer(function_body))
        if function_body and not parameter_matches:
            continue
        if _PARAMETER_BLOCK.sub("", function_body).strip():
            continue
        for parameter_match in parameter_matches:
            key = parameter_match.group(1).strip()
            if key in args:
                args = {}
                break
            raw_value = parameter_match.group(2).strip()
            value = _parse_json_value(raw_value)
            args[key] = raw_value if value is _INVALID else value
        if len(args) == len(parameter_matches):
            tool_calls.append({"name": name, "args": args, "id": f"call_xml_{idx}"})
    return tool_calls


def _parse_json_tool_calls(
    content: str,
    allowed_names: Collection[str],
) -> list[dict[str, Any]]:
    payload = _parse_mapping(content)
    if not payload or not isinstance(payload.get("tool_calls"), list):
        return []
    tool_calls: list[dict[str, Any]] = []
    for idx, raw_call in enumerate(payload["tool_calls"]):
        if not isinstance(raw_call, dict):
            continue
        raw_function = raw_call.get("function")
        function = raw_function if isinstance(raw_function, dict) else raw_call
        name = str(function.get("name", "")).strip()
        if not name or name not in allowed_names:
            continue
        raw_args = function.get("arguments", raw_call.get("arguments", {}))
        if isinstance(raw_args, str):
            args = _parse_mapping(raw_args)
        else:
            args = raw_args if isinstance(raw_args, dict) else None
        if args is None:
            continue
        tool_calls.append({
            "name": name,
            "args": args,
            "id": raw_call.get("id") or f"call_json_{idx}",
        })
    return tool_calls


def _parse_action_blocks(
    content: str,
    allowed_names: Collection[str],
) -> list[dict[str, Any]]:
    tool_calls: list[dict[str, Any]] = []
    for idx, match in enumerate(_ACTION_BLOCK.finditer(content)):
        name = match.group(1).strip()
        if name not in allowed_names:
            continue
        args = _parse_mapping(match.group(2))
        if args is None:
            continue
        tool_calls.append({"name": name, "args": args, "id": f"call_action_{idx}"})
    return tool_calls


def parse_hydrology_tool_calls(
    content: str,
    allowed_names: Collection[str],
) -> list[dict[str, Any]]:
    for parser in (
        _parse_xml_tool_calls,
        _parse_json_tool_calls,
        _parse_action_blocks,
    ):
        tool_calls = parser(content, allowed_names)
        if tool_calls:
            return tool_calls
    return parse_generic_tool_calls(content, set(allowed_names))


def contains_internal_protocol(content: str) -> bool:
    text = str(content or "").strip()
    if not text:
        return False
    return bool(
        _INTERNAL_PROTOCOL_MARKERS.search(text)
        or _SQL_BLOCK.search(text)
    )
