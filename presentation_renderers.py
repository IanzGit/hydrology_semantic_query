from __future__ import annotations

import json
import math
from collections import Counter
from datetime import UTC, date, datetime, time
from decimal import Decimal, InvalidOperation
from enum import Enum
from typing import Any

from app.agents.streaming import (
    WorkflowOutputType,
    build_structured_output,
    table_output,
)

from .models import SemanticQueryResult
from .presentation_models import (
    ChartSpec,
    ChartType,
    KpiSpec,
    MapSpec,
    PlannedBlock,
    PresentationBlockType,
    ReportBlock,
    ReportSection,
    StatusSpec,
    StructuredReport,
    TableSpec,
)
from .presentation_planner import build_presentation_plan


def _cell_value(value: Any) -> Any:
    if value is None:
        return None
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, bool | int):
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, Decimal):
        converted = float(value)
        return converted if math.isfinite(converted) else None
    if isinstance(value, datetime | date):
        return value.isoformat()
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    if isinstance(value, dict):
        return {str(key): _cell_value(item) for key, item in value.items()}
    if isinstance(value, list | tuple | set):
        return [_cell_value(item) for item in value]
    if isinstance(value, str):
        return value
    return str(value)


def _display_value(value: Any) -> str:
    normalized = _cell_value(value)
    if normalized is None:
        return "--"
    if isinstance(normalized, dict | list):
        return json.dumps(normalized, ensure_ascii=False)
    return str(normalized)


def _finite_number(value: Any) -> float | None:
    if value is None or isinstance(value, bool):
        return None
    try:
        converted = float(Decimal(str(value)))
    except (InvalidOperation, ValueError, TypeError, OverflowError):
        return None
    return converted if math.isfinite(converted) else None


def _time_key(value: Any) -> tuple[int, float | str]:
    if isinstance(value, datetime):
        parsed = value
    elif isinstance(value, date):
        parsed = datetime.combine(value, time.min)
    else:
        text = str(value or "")
        try:
            parsed = datetime.fromisoformat(text.replace("Z", "+00:00"))
        except ValueError:
            return 0, text
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=UTC)
    try:
        return 1, parsed.timestamp()
    except (OverflowError, OSError, ValueError):
        return 0, parsed.isoformat()


def _unique(values: list[Any]) -> list[Any]:
    result: list[Any] = []
    seen: set[str] = set()
    for value in values:
        key = json.dumps(_cell_value(value), ensure_ascii=False, sort_keys=True, default=str)
        if key not in seen:
            seen.add(key)
            result.append(value)
    return result


_FRONTEND_CHART_TYPES = {"BAR", "LINE", "PIE", "BAR_STACK"}


def _frontend_chart_output(
    *,
    chart_type: str,
    chart_name: str,
    series_data: list[dict[str, Any]],
    stack_config: dict[str, Any] | None = None,
) -> dict[str, Any]:
    normalized_type = chart_type.upper()
    if normalized_type not in _FRONTEND_CHART_TYPES:
        raise ValueError(f"前端不支持图表类型: {chart_type}")
    payload: dict[str, Any] = {
        "chartType": normalized_type,
        "chartName": chart_name,
        "hasData": bool(series_data),
        "seriesData": series_data,
    }
    if normalized_type == "BAR_STACK":
        payload["stackConfig"] = stack_config or {"stackGroups": ["总量"]}
    return build_structured_output(
        output_type=WorkflowOutputType.CHART_OUTPUT,
        data=payload,
    )


class BlockRenderer:
    block_type: PresentationBlockType

    def render(
        self,
        plan: PlannedBlock,
        result: SemanticQueryResult,
    ) -> ReportBlock:
        raise NotImplementedError

    def to_stream(self, block: ReportBlock) -> dict[str, Any] | None:
        raise NotImplementedError


class KpiRenderer(BlockRenderer):
    block_type = PresentationBlockType.KPI

    def render(
        self,
        plan: PlannedBlock,
        result: SemanticQueryResult,
    ) -> ReportBlock:
        if not isinstance(plan.config, KpiSpec):
            raise TypeError("KPI block 配置类型无效")
        row = result.rows[0] if result.rows else {}
        if plan.config.mode == "latest" and plan.config.time_field and result.rows:
            row = max(
                result.rows,
                key=lambda item: _time_key(item.get(plan.config.time_field.name)),
            )
        items = [
            {
                "field": field.name,
                "label": field.title,
                "value": _cell_value(row.get(field.name)),
                "dataType": field.data_type,
            }
            for field in plan.config.fields
        ]
        if plan.config.time_field and plan.config.mode == "latest":
            timestamp = _cell_value(row.get(plan.config.time_field.name))
        else:
            timestamp = None
        return ReportBlock(
            id=plan.id,
            type=plan.type,
            title=plan.title,
            description=plan.description,
            data={"items": items, "asOf": timestamp},
            config=plan.config.model_dump(mode="json", exclude_none=True),
            priority=plan.priority,
        )

    def to_stream(self, block: ReportBlock) -> dict[str, Any]:
        items = block.data.get("items", []) if isinstance(block.data, dict) else []
        points = []
        for item in items:
            value = _finite_number(item.get("value"))
            if value is not None:
                points.append({"name": str(item.get("label") or ""), "value": value})
        return _frontend_chart_output(
            chart_type="BAR",
            chart_name=block.title,
            series_data=[{"name": "指标值", "data": points}] if points else [],
        )


def _severity(value: str) -> tuple[str, str]:
    lowered = value.lower()
    if any(marker in lowered for marker in ("严重", "危险", "报警", "告警", "异常", "超限", "critical", "danger", "error", "red")):
        return "critical", "#d94b4b"
    if any(marker in lowered for marker in ("预警", "警告", "注意", "warning", "yellow", "orange")):
        return "warning", "#d99524"
    if any(marker in lowered for marker in ("正常", "安全", "完成", "normal", "success", "green", "completed")):
        return "success", "#2f9e73"
    return "info", "#4b78c2"


class StatusRenderer(BlockRenderer):
    block_type = PresentationBlockType.STATUS

    def render(
        self,
        plan: PlannedBlock,
        result: SemanticQueryResult,
    ) -> ReportBlock:
        if not isinstance(plan.config, StatusSpec):
            raise TypeError("Status block 配置类型无效")
        values = [_display_value(row.get(plan.config.field.name)) for row in result.rows]
        counts = Counter(values)
        items = []
        for label, count in counts.most_common(plan.config.max_items):
            severity, color = _severity(label)
            items.append({
                "label": label,
                "count": count,
                "severity": severity,
                "color": color,
            })
        return ReportBlock(
            id=plan.id,
            type=plan.type,
            title=plan.title,
            description=plan.description,
            data={"items": items, "total": len(result.rows)},
            config=plan.config.model_dump(mode="json", exclude_none=True),
            priority=plan.priority,
        )

    def to_stream(self, block: ReportBlock) -> dict[str, Any]:
        items = block.data.get("items", []) if isinstance(block.data, dict) else []
        points = [
            {
                "name": str(item.get("label") or ""),
                "value": int(item.get("count") or 0),
            }
            for item in items
        ]
        return _frontend_chart_output(
            chart_type="PIE",
            chart_name=block.title,
            series_data=[{"name": "记录数", "data": points}] if points else [],
        )


class ChartRenderer(BlockRenderer):
    block_type = PresentationBlockType.CHART

    def render(
        self,
        plan: PlannedBlock,
        result: SemanticQueryResult,
    ) -> ReportBlock:
        if not isinstance(plan.config, ChartSpec):
            raise TypeError("Chart block 配置类型无效")
        fields = [plan.config.x, *plan.config.y]
        if plan.config.series:
            fields.append(plan.config.series)
        names = list(dict.fromkeys(field.name for field in fields))
        rows = [
            {name: _cell_value(row.get(name)) for name in names}
            for row in result.rows[:plan.config.limit]
        ]
        return ReportBlock(
            id=plan.id,
            type=plan.type,
            title=plan.title,
            description=plan.description,
            data={"rows": rows},
            config=plan.config.model_dump(mode="json", exclude_none=True),
            priority=plan.priority,
        )

    def _ordered_rows(self, rows: list[dict[str, Any]], spec: ChartSpec) -> list[dict[str, Any]]:
        if spec.sort == "none":
            return rows
        reverse = spec.sort == "desc"
        if spec.chart_type in {ChartType.BAR, ChartType.PIE} and spec.y:
            return sorted(
                rows,
                key=lambda row: _finite_number(row.get(spec.y[0].name)) or 0,
                reverse=reverse,
            )
        return sorted(rows, key=lambda row: _time_key(row.get(spec.x.name)), reverse=reverse)

    def _series_data(self, rows: list[dict[str, Any]], spec: ChartSpec) -> list[dict[str, Any]]:
        rows = self._ordered_rows(rows, spec)
        if spec.series:
            group_values = _unique([row.get(spec.series.name) for row in rows])
            categories = _unique([row.get(spec.x.name) for row in rows])
            series = []
            for measure in spec.y:
                for group in group_values:
                    group_rows = [row for row in rows if row.get(spec.series.name) == group]
                    values_by_category = {
                        _display_value(row.get(spec.x.name)): _cell_value(row.get(measure.name))
                        for row in group_rows
                    }
                    name = _display_value(group)
                    if len(spec.y) > 1:
                        name = f"{measure.title} · {name}"
                    series.append({
                        "name": name,
                        "data": [
                            {
                                "name": _display_value(category),
                                "value": values_by_category.get(_display_value(category)),
                            }
                            for category in categories
                        ],
                    })
            return series
        return [
            {
                "name": measure.title,
                "data": [
                    {
                        "name": _display_value(row.get(spec.x.name)),
                        "value": _cell_value(row.get(measure.name)),
                    }
                    for row in rows
                ],
            }
            for measure in spec.y
        ]

    def _inline_chart(
        self,
        block: ReportBlock,
        rows: list[dict[str, Any]],
        spec: ChartSpec,
    ) -> dict[str, Any]:
        frontend_type = {
            ChartType.AREA: "LINE",
            ChartType.BAR: "BAR",
            ChartType.HEATMAP: "LINE",
            ChartType.HISTOGRAM: "BAR",
            ChartType.LINE: "LINE",
            ChartType.PIE: "PIE",
            ChartType.SCATTER: "LINE",
        }[spec.chart_type]
        if spec.chart_type == ChartType.HISTOGRAM:
            bins, values = self._histogram(rows, spec)
            series_data = [{
                "name": "频数",
                "data": [
                    {"name": str(name), "value": value}
                    for name, value in zip(bins, values, strict=False)
                ],
            }]
        elif spec.chart_type == ChartType.PIE:
            measure = spec.y[0]
            ordered = self._ordered_rows(rows, spec)
            series_data = [{
                "name": measure.title,
                "data": [
                    {
                        "name": _display_value(row.get(spec.x.name)),
                        "value": _cell_value(row.get(measure.name)),
                    }
                    for row in ordered
                ],
            }]
        elif spec.chart_type == ChartType.SCATTER:
            measure = spec.y[0]
            series_data = [{
                "name": measure.title,
                "data": [
                    {
                        "name": _display_value(row.get(spec.x.name)),
                        "value": _cell_value(row.get(measure.name)),
                    }
                    for row in rows
                ],
            }]
        else:
            series_data = self._series_data(rows, spec)
        return _frontend_chart_output(
            chart_type=frontend_type,
            chart_name=block.title,
            series_data=series_data,
        )

    def _histogram(
        self,
        rows: list[dict[str, Any]],
        spec: ChartSpec,
    ) -> tuple[list[str], list[int]]:
        values = [
            value
            for row in rows
            if (value := _finite_number(row.get(spec.x.name))) is not None
        ]
        if not values:
            bins: list[str] = []
            counts: list[int] = []
        elif min(values) == max(values):
            bins = [_display_value(values[0])]
            counts = [len(values)]
        else:
            bin_count = min(12, max(5, round(math.sqrt(len(values)))))
            start = min(values)
            width = (max(values) - start) / bin_count
            counts = [0 for _ in range(bin_count)]
            for value in values:
                index = min(bin_count - 1, int((value - start) / width))
                counts[index] += 1
            bins = [
                f"{start + index * width:.2f}–{start + (index + 1) * width:.2f}"
                for index in range(bin_count)
            ]
        return bins, counts

    def to_stream(self, block: ReportBlock) -> dict[str, Any] | None:
        spec = ChartSpec.model_validate(block.config)
        rows = block.data.get("rows", []) if isinstance(block.data, dict) else []
        if not rows:
            return None
        return self._inline_chart(block, rows, spec)


class MapRenderer(BlockRenderer):
    block_type = PresentationBlockType.MAP

    def render(
        self,
        plan: PlannedBlock,
        result: SemanticQueryResult,
    ) -> ReportBlock:
        if not isinstance(plan.config, MapSpec):
            raise TypeError("Map block 配置类型无效")
        fields = [plan.config.longitude, plan.config.latitude]
        if plan.config.value:
            fields.append(plan.config.value)
        if plan.config.label:
            fields.append(plan.config.label)
        names = list(dict.fromkeys(field.name for field in fields))
        points = []
        for row in result.rows[:plan.config.limit]:
            longitude = _finite_number(row.get(plan.config.longitude.name))
            latitude = _finite_number(row.get(plan.config.latitude.name))
            if longitude is None or latitude is None:
                continue
            value = _finite_number(row.get(plan.config.value.name)) if plan.config.value else None
            label = _display_value(row.get(plan.config.label.name)) if plan.config.label else "水文监测点"
            points.append({
                "name": label,
                "value": [longitude, latitude, value],
                "row": {name: _cell_value(row.get(name)) for name in names},
            })
        return ReportBlock(
            id=plan.id,
            type=plan.type,
            title=plan.title,
            description=plan.description,
            data={"points": points},
            config=plan.config.model_dump(mode="json", exclude_none=True),
            priority=plan.priority,
        )

    def to_stream(self, block: ReportBlock) -> dict[str, Any] | None:
        return None


class TableRenderer(BlockRenderer):
    block_type = PresentationBlockType.TABLE

    def render(
        self,
        plan: PlannedBlock,
        result: SemanticQueryResult,
    ) -> ReportBlock:
        if not isinstance(plan.config, TableSpec):
            raise TypeError("Table block 配置类型无效")
        rows = [
            {
                field.name: _cell_value(row.get(field.name))
                for field in plan.config.fields
            }
            for row in result.rows[:plan.config.limit]
        ]
        return ReportBlock(
            id=plan.id,
            type=plan.type,
            title=plan.title,
            description=plan.description,
            data={"rows": rows, "rowCount": len(result.rows)},
            config=plan.config.model_dump(mode="json", exclude_none=True),
            priority=plan.priority,
        )

    def to_stream(self, block: ReportBlock) -> dict[str, Any]:
        spec = TableSpec.model_validate(block.config)
        rows = block.data.get("rows", []) if isinstance(block.data, dict) else []
        headers = [
            {"field": field.name, "title": field.title, "dataType": field.data_type}
            for field in spec.fields
        ]
        return table_output(table_name=block.title, headers=headers, rows=rows)


class RendererRegistry:
    def __init__(self, renderers: list[BlockRenderer]) -> None:
        self._renderers = {renderer.block_type: renderer for renderer in renderers}

    def render_block(
        self,
        plan: PlannedBlock,
        result: SemanticQueryResult,
    ) -> ReportBlock:
        return self._renderers[plan.type].render(plan, result)

    def render_output(self, block: ReportBlock) -> dict[str, Any] | None:
        return self._renderers[block.type].to_stream(block)


DEFAULT_RENDERER_REGISTRY = RendererRegistry([
    KpiRenderer(),
    StatusRenderer(),
    ChartRenderer(),
    MapRenderer(),
    TableRenderer(),
])


def compose_structured_report(
    result: SemanticQueryResult,
    *,
    answer: str,
    question: str,
    registry: RendererRegistry = DEFAULT_RENDERER_REGISTRY,
) -> StructuredReport:
    plan = build_presentation_plan(result, question)
    blocks = [registry.render_block(block, result) for block in plan.blocks]
    section_specs = (
        ("overview", "结果概览", {PresentationBlockType.KPI, PresentationBlockType.STATUS}),
        ("visualizations", "可视化分析", {PresentationBlockType.CHART, PresentationBlockType.MAP}),
        ("details", "详细数据", {PresentationBlockType.TABLE}),
    )
    sections = []
    for section_id, title, block_types in section_specs:
        section_blocks = sorted(
            [block for block in blocks if block.type in block_types],
            key=lambda block: block.priority,
        )
        if section_blocks:
            sections.append(ReportSection(id=section_id, title=title, blocks=section_blocks))
    return StructuredReport(
        title=plan.title,
        summary=answer,
        profile=plan.profile,
        sections=sections,
        metadata={
            "rowCount": plan.profile.row_count,
            "columnCount": plan.profile.column_count,
            "blockCount": len(blocks),
        },
    )


def render_structured_report(
    report: StructuredReport,
    registry: RendererRegistry = DEFAULT_RENDERER_REGISTRY,
) -> list[dict[str, Any]]:
    blocks = sorted(
        [block for section in report.sections for block in section.blocks],
        key=lambda block: block.priority,
    )
    return [output for block in blocks if (output := registry.render_output(block)) is not None]
