from __future__ import annotations

import math
from decimal import Decimal, InvalidOperation
from typing import Any

from app.agents.streaming import llm_stream_output

from .models import QueryOutcome, SemanticColumn, SemanticQueryResult
from .presentation_renderers import compose_structured_report, render_structured_report
from .reporting import REPORT_FAILURE_WARNING, generate_report, rows_to_markdown

_ANNOTATION_MEMBER_TYPES = {
    "dimensions": "dimension",
    "timeDimensions": "time_dimension",
    "measures": "measure",
}


def _annotation_map(payload: dict[str, Any]) -> dict[str, dict[str, Any]]:
    annotation = payload.get("annotation") or {}
    result: dict[str, dict[str, Any]] = {}
    if not isinstance(annotation, dict):
        return result
    for group, member_type in _ANNOTATION_MEMBER_TYPES.items():
        values = annotation.get(group) or {}
        if isinstance(values, dict):
            for name, item in values.items():
                metadata = dict(item) if isinstance(item, dict) else {}
                metadata["member_type"] = member_type
                result[str(name)] = metadata
    return result


def _number(value: Any) -> Any:
    if value is None or (
        isinstance(value, int | float) and not isinstance(value, bool)
    ):
        if isinstance(value, float) and not math.isfinite(value):
            return None
        return value
    try:
        number = Decimal(str(value))
    except (InvalidOperation, ValueError):
        return value
    if not number.is_finite():
        return None
    if number == number.to_integral_value():
        return int(number)
    converted = float(number)
    return converted if math.isfinite(converted) else None


def _typed(value: Any, data_type: str) -> Any:
    if value is None:
        return None
    if data_type == "number":
        return _number(value)
    if data_type == "boolean":
        if isinstance(value, bool):
            return value
        if isinstance(value, int | float) and value in {0, 1}:
            return bool(value)
        if isinstance(value, str):
            normalized = value.strip().lower()
            if normalized in {"1", "true", "yes"}:
                return True
            if normalized in {"0", "false", "no"}:
                return False
    return value


def _is_hidden_output_column(column: SemanticColumn) -> bool:
    short_name = column.name.rsplit(".", 1)[-1].lower()
    title = column.title.strip().lower()
    return (
        short_name in {"id", "relation_key", "code"}
        or short_name.endswith(("_id", "_ids", "_code"))
        or title.endswith(("id", "id列表", "编码"))
    )


def normalize_cube_response(
    payload: dict[str, Any],
) -> tuple[list[SemanticColumn], list[dict[str, Any]]]:
    if isinstance(payload.get("results"), list) and payload["results"]:
        first = payload["results"][0]
        if not isinstance(first, dict):
            raise ValueError("Cube /load 响应的 results 只能包含对象")
        payload = first
    raw_rows = payload.get("data") or []
    if not isinstance(raw_rows, list):
        raise ValueError("Cube /load 响应的 data 必须是列表")
    if any(not isinstance(row, dict) for row in raw_rows):
        raise ValueError("Cube /load 响应的 data 只能包含对象")
    rows = [dict(row) for row in raw_rows]
    annotations = _annotation_map(payload)
    names = list(annotations)
    for row in rows:
        for name in row:
            if name not in names:
                names.append(name)
    columns = [
        SemanticColumn(
            name=name,
            title=str(annotations.get(name, {}).get("title") or name),
            data_type=str(annotations.get(name, {}).get("type") or "string"),
            member_type=str(annotations.get(name, {}).get("member_type") or "unknown"),
        )
        for name in names
    ]
    columns = [column for column in columns if not _is_hidden_output_column(column)]
    visible_names = {column.name for column in columns}
    types = {column.name: column.data_type for column in columns}
    normalized_rows = [
        {
            name: _typed(value, types.get(name, "string"))
            for name, value in row.items()
            if name in visible_names
        }
        for row in rows
    ]
    return columns, normalized_rows


def build_result_outputs(
    result: SemanticQueryResult,
    *,
    answer: str,
    question: str,
) -> list[dict[str, Any]]:
    if result.outcome != QueryOutcome.SUCCESS:
        return [llm_stream_output(text=answer)] if answer else []
    report = compose_structured_report(
        result,
        answer=answer,
        question=question,
    )
    result.presentation = report
    return render_structured_report(report)


__all__ = [
    "REPORT_FAILURE_WARNING",
    "build_result_outputs",
    "generate_report",
    "normalize_cube_response",
    "rows_to_markdown",
]
