from __future__ import annotations

import json
import math
from decimal import Decimal, InvalidOperation
from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage

from app.agents.messages import stringify_message_content
from app.agents.streaming import (
    WorkflowOutputType,
    build_structured_output,
    llm_stream_output,
    table_output,
)

from .models import QueryOutcome, SemanticColumn, SemanticQueryResult

REPORT_FAILURE_WARNING = "Markdown 分析报告生成失败，已保留查询摘要。"


def _annotation_map(payload: dict[str, Any]) -> dict[str, dict[str, Any]]:
    annotation = payload.get("annotation") or {}
    result: dict[str, dict[str, Any]] = {}
    if not isinstance(annotation, dict):
        return result
    for group in ("dimensions", "timeDimensions", "measures"):
        values = annotation.get(group) or {}
        if isinstance(values, dict):
            for name, item in values.items():
                result[str(name)] = item if isinstance(item, dict) else {}
    return result


def _number(value: Any) -> Any:
    if value is None or isinstance(value, (int, float)) and not isinstance(value, bool):
        return value
    try:
        number = Decimal(str(value))
    except (InvalidOperation, ValueError):
        return value
    if number == number.to_integral_value():
        return int(number)
    converted = float(number)
    return converted if math.isfinite(converted) else None


def _typed(value: Any, data_type: str) -> Any:
    if value is None:
        return None
    if data_type == "number":
        return _number(value)
    if data_type == "boolean" and isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes"}
    return value


def normalize_cube_response(
    payload: dict[str, Any],
) -> tuple[list[SemanticColumn], list[dict[str, Any]]]:
    if isinstance(payload.get("results"), list) and payload["results"]:
        first = payload["results"][0]
        if isinstance(first, dict):
            payload = first
    raw_rows = payload.get("data") or []
    if not isinstance(raw_rows, list):
        raise ValueError("Cube /load 响应的 data 必须是列表")
    rows = [dict(row) for row in raw_rows if isinstance(row, dict)]
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
        )
        for name in names
    ]
    types = {column.name: column.data_type for column in columns}
    normalized_rows = [
        {name: _typed(value, types.get(name, "string")) for name, value in row.items()}
        for row in rows
    ]
    return columns, normalized_rows


def _table(result: SemanticQueryResult) -> dict[str, Any]:
    headers = [
        {"field": column.name, "title": column.title, "dataType": column.data_type}
        for column in result.columns
    ]
    return table_output(table_name="水文语义查询结果", headers=headers, rows=result.rows)


def _chart(result: SemanticQueryResult, chart_type: str) -> dict[str, Any] | None:
    if not result.rows or len(result.columns) < 2:
        return None
    numeric = [column for column in result.columns if column.data_type == "number"]
    if not numeric:
        return None
    label = next((item for item in result.columns if item.data_type != "number"), result.columns[0])
    if chart_type == "PIE":
        value = numeric[0]
        series = [{
            "name": value.title,
            "data": [
                {"name": str(row.get(label.name, "")), "value": row.get(value.name)}
                for row in result.rows
            ],
        }]
    else:
        series = [
            {
                "name": value.title,
                "data": [
                    {"name": str(row.get(label.name, "")), "value": row.get(value.name)}
                    for row in result.rows
                ],
            }
            for value in numeric
            if value.name != label.name
        ]
    if not series:
        return None
    return build_structured_output(
        output_type=WorkflowOutputType.CHART_OUTPUT,
        data={
            "mode": "inline",
            "taskId": None,
            "chartData": {
                "chartType": chart_type,
                "chartName": "水文语义查询结果",
                "hasData": True,
                "seriesData": series,
            },
        },
    )


def _scatter(result: SemanticQueryResult) -> dict[str, Any] | None:
    numeric = [column for column in result.columns if column.data_type == "number"]
    if len(numeric) < 2 or not result.rows:
        return None
    x_axis, y_axis = numeric[:2]
    option = {
        "title": {"text": "水文语义查询结果", "left": "center"},
        "tooltip": {"trigger": "item"},
        "xAxis": {"type": "value", "name": x_axis.title},
        "yAxis": {"type": "value", "name": y_axis.title},
        "series": [{
            "name": y_axis.title,
            "type": "scatter",
            "data": [
                [row.get(x_axis.name), row.get(y_axis.name)] for row in result.rows
            ],
        }],
    }
    return llm_stream_output(text="```echarts\n" + json.dumps(option, ensure_ascii=False) + "\n```")


def build_result_outputs(
    result: SemanticQueryResult,
    *,
    answer: str,
    question: str,
) -> list[dict[str, Any]]:
    outputs = [llm_stream_output(text=answer)] if answer else []
    if result.outcome != QueryOutcome.SUCCESS:
        return outputs
    chart_type = None
    for marker, candidate in (
        ("饼图", "PIE"),
        ("柱状图", "BAR"),
        ("折线图", "LINE"),
        ("趋势", "LINE"),
    ):
        if marker in question:
            chart_type = candidate
            break
    if "散点图" in question:
        scatter = _scatter(result)
        if scatter:
            outputs.append(scatter)
            return outputs
    if chart_type:
        chart = _chart(result, chart_type)
        if chart:
            outputs.append(chart)
            return outputs
    outputs.append(_table(result))
    return outputs


def rows_to_markdown(result: SemanticQueryResult) -> str:
    if not result.columns:
        return ""
    lines = [
        "| " + " | ".join(column.title for column in result.columns) + " |",
        "| " + " | ".join("---" for _ in result.columns) + " |",
    ]
    for row in result.rows[:50]:
        values = []
        for column in result.columns:
            value = row.get(column.name)
            text = json.dumps(value, ensure_ascii=False) if isinstance(value, (dict, list)) else str(value if value is not None else "")
            values.append(text.replace("|", "\\|").replace("\n", "<br>"))
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines)


async def generate_report(runtime, question: str, result: SemanticQueryResult) -> str:
    messages = [
        SystemMessage(content=(
            "你是谨慎的水文数据分析助手。用中文生成不超过 800 字的简洁 Markdown 报告，"
            "只总结数据可直接验证的结论，不得编造。"
        )),
        HumanMessage(content=f"用户问题：{question}\n\nCube 查询数据：\n{rows_to_markdown(result)}"),
    ]
    model = runtime.get_chat_model(streaming=True).bind(
        extra_body={"enable_thinking": False},
    )
    response = await model.ainvoke(messages, config={"callbacks": []})
    report = stringify_message_content(response.content).strip()
    if not report:
        raise ValueError("报告模型返回了空响应")
    return report
