from __future__ import annotations

import json
import math
import re
from collections import Counter
from datetime import UTC, date, datetime, time
from decimal import Decimal, InvalidOperation
from enum import Enum
from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage

from app.agents.messages import stringify_message_content
from app.agents.streaming import (
    WorkflowOutputType,
    build_structured_output,
    llm_stream_output,
    table_output,
)

from .models import (
    ChartSpec,
    ChartType,
    ColumnProfile,
    ColumnRole,
    FieldRef,
    KpiSpec,
    MapSpec,
    PlannedBlock,
    PresentationBlockType,
    PresentationPlan,
    QueryOutcome,
    ReportBlock,
    ReportSection,
    ResultProfile,
    ResultShape,
    SemanticColumn,
    SemanticQueryResult,
    StatusSpec,
    StructuredReport,
    TableSpec,
)
from .tool_call_parser import contains_internal_protocol


def is_internal_identifier(name: str) -> bool:
    short_name = name.partition(".")[2] or name
    return (
        short_name == "id"
        or short_name == "relation_key"
        or short_name.endswith("_id")
        or short_name.endswith("_ids")
    )

_LATITUDE_NAMES = {"lat", "latitude", "纬度"}
_LONGITUDE_NAMES = {"lng", "lon", "long", "longitude", "经度"}
_STATUS_MARKERS = ("status", "state", "severity", "level", "状态", "等级", "级别")
_TIME_MARKERS = ("time", "date", "day", "month", "year", "timestamp", "时间", "日期", "月份", "年份")


def _short_names(column: SemanticColumn) -> set[str]:
    short_name = column.name.rsplit(".", 1)[-1].lower()
    tokens = {item for item in re.split(r"[^a-z0-9\u4e00-\u9fff]+", short_name) if item}
    tokens.add(short_name)
    tokens.add(column.title.strip().lower())
    return tokens


def _contains_marker(column: SemanticColumn, markers: tuple[str, ...]) -> bool:
    text = f"{column.name.rsplit('.', 1)[-1]} {column.title}".lower()
    return any(marker in text for marker in markers)


def _role(column: SemanticColumn) -> ColumnRole:
    names = _short_names(column)
    if names & _LATITUDE_NAMES:
        return ColumnRole.LATITUDE
    if names & _LONGITUDE_NAMES:
        return ColumnRole.LONGITUDE
    if column.member_type == "time_dimension" or column.data_type.lower() in {
        "date",
        "datetime",
        "time",
        "timestamp",
    }:
        return ColumnRole.TIME
    if column.member_type == "measure":
        return ColumnRole.MEASURE
    if _contains_marker(column, _TIME_MARKERS):
        return ColumnRole.TIME
    if _contains_marker(column, _STATUS_MARKERS):
        return ColumnRole.STATUS
    if is_internal_identifier(column.name):
        return ColumnRole.IDENTIFIER
    if column.member_type == "dimension":
        return ColumnRole.CATEGORY
    if column.data_type.lower() in {"number", "integer", "float", "decimal"}:
        return ColumnRole.MEASURE
    if column.data_type.lower() in {"boolean", "string"}:
        return ColumnRole.CATEGORY
    return ColumnRole.UNKNOWN


def _distinct_key(value: Any) -> str:
    if isinstance(value, (dict, list, tuple, set)):
        try:
            return json.dumps(value, ensure_ascii=False, sort_keys=True, default=str)
        except TypeError:
            return str(value)
    return str(value)


def _finite_number(value: Any) -> float | None:
    if value is None or isinstance(value, bool):
        return None
    try:
        converted = float(Decimal(str(value)))
    except (InvalidOperation, ValueError, TypeError, OverflowError):
        return None
    return converted if math.isfinite(converted) else None


def _profile_column(
    column: SemanticColumn,
    rows: list[dict[str, Any]],
) -> ColumnProfile:
    values = [row.get(column.name) for row in rows]
    actual = [value for value in values if value is not None]
    numeric = [value for value in (_finite_number(item) for item in actual) if value is not None]
    return ColumnProfile(
        name=column.name,
        title=column.title,
        data_type=column.data_type,
        role=_role(column),
        member_type=column.member_type,
        null_count=len(values) - len(actual),
        distinct_count=len({_distinct_key(value) for value in actual}),
        minimum=min(numeric) if numeric else None,
        maximum=max(numeric) if numeric else None,
    )


def build_result_profile(result: SemanticQueryResult) -> ResultProfile:
    columns = [_profile_column(column, result.rows) for column in result.columns]
    by_role = {
        role: [column.name for column in columns if column.role == role]
        for role in ColumnRole
    }
    measures = by_role[ColumnRole.MEASURE]
    times = by_role[ColumnRole.TIME]
    categories = by_role[ColumnRole.CATEGORY]
    statuses = by_role[ColumnRole.STATUS]
    latitude = next(iter(by_role[ColumnRole.LATITUDE]), None)
    longitude = next(iter(by_role[ColumnRole.LONGITUDE]), None)
    if not result.rows:
        shape = ResultShape.EMPTY
    elif len(result.rows) == 1 and measures:
        shape = ResultShape.SCALAR
    elif latitude and longitude:
        shape = ResultShape.GEOSPATIAL
    elif statuses:
        shape = ResultShape.STATUS
    elif times and measures:
        shape = ResultShape.TEMPORAL
    elif categories and measures:
        shape = ResultShape.CATEGORICAL
    elif measures:
        shape = ResultShape.NUMERIC
    else:
        shape = ResultShape.TABULAR
    return ResultProfile(
        row_count=len(result.rows),
        column_count=len(result.columns),
        shape=shape,
        columns=columns,
        measure_fields=measures,
        time_fields=times,
        category_fields=categories,
        status_fields=statuses,
        identifier_fields=by_role[ColumnRole.IDENTIFIER],
        latitude_field=latitude,
        longitude_field=longitude,
        primary_measure=next(iter(measures), None),
        primary_time=next(iter(times), None),
        primary_category=next(iter(categories), None),
    )

def _field(profile: ResultProfile, name: str | None) -> FieldRef | None:
    if not name:
        return None
    column = next((item for item in profile.columns if item.name == name), None)
    if column is None:
        return None
    return FieldRef(name=column.name, title=column.title, data_type=column.data_type)


def _fields(profile: ResultProfile, names: list[str]) -> list[FieldRef]:
    return [field for name in names if (field := _field(profile, name)) is not None]


def _explicit_chart_type(question: str) -> ChartType | None:
    for markers, chart_type in (
        (("饼图", "环形图", "占比", "比例", "构成", "份额"), ChartType.PIE),
        (("柱状图", "条形图"), ChartType.BAR),
        (("折线图", "趋势"), ChartType.LINE),
    ):
        if any(marker in question for marker in markers):
            return chart_type
    return None


def _category(profile: ResultProfile) -> FieldRef | None:
    return _field(
        profile,
        profile.primary_category or next(iter(profile.status_fields), None),
    )


def _chart_spec(profile: ResultProfile, question: str) -> ChartSpec | None:
    if profile.row_count < 2 or not profile.measure_fields:
        return None
    measures = _fields(profile, profile.measure_fields[:4])
    time = _field(profile, profile.primary_time)
    category = _category(profile)
    explicit = _explicit_chart_type(question)
    if explicit == ChartType.PIE and category:
        return ChartSpec(chart_type=explicit, x=category, y=[measures[0]], sort="desc")
    if explicit in {ChartType.BAR, ChartType.LINE}:
        axis = time or category
        if axis:
            return ChartSpec(
                chart_type=explicit,
                x=axis,
                y=measures,
                series=category if time and category else None,
            )
    if time:
        return ChartSpec(
            chart_type=ChartType.LINE,
            x=time,
            y=measures,
            series=category,
        )
    if category:
        return ChartSpec(chart_type=ChartType.BAR, x=category, y=measures, sort="desc")
    return None


def build_presentation_plan(
    result: SemanticQueryResult,
    question: str,
) -> PresentationPlan:
    profile = build_result_profile(result)
    blocks: list[PlannedBlock] = []
    measure_fields = _fields(profile, profile.measure_fields[:4])
    time_field = _field(profile, profile.primary_time)
    category_profile = next(
        (
            item
            for item in profile.columns
            if item.name == profile.primary_category
        ),
        None,
    )
    latest_kpi = (
        time_field is not None
        and (category_profile is None or category_profile.distinct_count <= 1)
    )
    if measure_fields and (profile.row_count == 1 or latest_kpi):
        blocks.append(PlannedBlock(
            id="overview-kpis",
            type=PresentationBlockType.KPI,
            title="核心指标",
            description="查询结果中的关键数值。",
            priority=10,
            config=KpiSpec(
                fields=measure_fields,
                mode="latest" if profile.row_count > 1 and time_field else "value",
                time_field=time_field if profile.row_count > 1 else None,
            ),
        ))
    label_name = profile.primary_category or next(iter(profile.identifier_fields), None)
    for index, status_name in enumerate(profile.status_fields[:2]):
        status_field = _field(profile, status_name)
        if status_field:
            blocks.append(PlannedBlock(
                id=f"overview-status-{index + 1}",
                type=PresentationBlockType.STATUS,
                title=f"{status_field.title}分布",
                description="按查询记录汇总状态分布。",
                priority=20 + index,
                config=StatusSpec(
                    field=status_field,
                    label_field=_field(profile, label_name),
                ),
            ))
    if profile.latitude_field and profile.longitude_field:
        latitude = _field(profile, profile.latitude_field)
        longitude = _field(profile, profile.longitude_field)
        if latitude and longitude:
            blocks.append(PlannedBlock(
                id="visualization-map",
                type=PresentationBlockType.MAP,
                title="空间分布",
                description="按经纬度展示查询对象的空间位置。",
                priority=30,
                config=MapSpec(
                    latitude=latitude,
                    longitude=longitude,
                    value=_field(profile, profile.primary_measure),
                    label=_category(profile),
                ),
            ))
    chart_spec = _chart_spec(profile, question)
    if chart_spec:
        blocks.append(PlannedBlock(
            id="visualization-primary",
            type=PresentationBlockType.CHART,
            title="数据可视化",
            description="根据结果字段语义和数据形态生成。",
            priority=40,
            config=chart_spec,
        ))
    blocks.append(PlannedBlock(
        id="details-table",
        type=PresentationBlockType.TABLE,
        title="详细数据",
        description=f"查询共返回 {profile.row_count} 行数据。",
        priority=90,
        config=TableSpec(fields=_fields(profile, [item.name for item in profile.columns])),
    ))
    return PresentationPlan(
        title="水文语义查询报告",
        profile=profile,
        blocks=blocks,
    )

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
        if spec.chart_type == ChartType.PIE:
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
        else:
            series_data = self._series_data(rows, spec)
        return _frontend_chart_output(
            chart_type=spec.chart_type.value,
            chart_name=block.title,
            series_data=series_data,
        )

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

REPORT_FAILURE_WARNING = "Markdown 分析报告生成失败，已保留查询摘要。"


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
            "你是谨慎的水文数据分析助手。用中文生成内容丰富的 Markdown 分析报告，"
            "只总结数据可直接验证的结论，不得编造。"
        )),
        HumanMessage(content=f"用户问题：{question}\n\n 数据库查询数据：\n{rows_to_markdown(result)}"),
    ]
    model = runtime.get_chat_model(streaming=True).bind(
        extra_body={"enable_thinking": False},
    )
    response = await model.ainvoke(messages, config={"callbacks": []})
    report = stringify_message_content(response.content).strip()
    if not report:
        raise ValueError("报告模型返回了空响应")
    if contains_internal_protocol(report):
        raise ValueError("报告模型返回了内部工具协议")
    return report


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
    "rows_to_markdown",
]
