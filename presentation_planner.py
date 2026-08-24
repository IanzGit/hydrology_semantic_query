from __future__ import annotations

from .models import SemanticQueryResult
from .presentation_models import (
    ChartSpec,
    ChartType,
    FieldRef,
    KpiSpec,
    MapSpec,
    PlannedBlock,
    PresentationBlockType,
    PresentationPlan,
    ResultProfile,
    StatusSpec,
    TableSpec,
)
from .result_profile import build_result_profile


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
        (("热力图",), ChartType.HEATMAP),
        (("直方图", "分布图"), ChartType.HISTOGRAM),
        (("散点图",), ChartType.SCATTER),
        (("面积图",), ChartType.AREA),
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
    if explicit == ChartType.SCATTER and len(measures) >= 2:
        return ChartSpec(
            chart_type=explicit,
            x=measures[0],
            y=[measures[1]],
            series=category,
            sort="none",
        )
    if explicit == ChartType.HISTOGRAM:
        return ChartSpec(chart_type=explicit, x=measures[0], y=[measures[0]], sort="none")
    if explicit == ChartType.HEATMAP and time and category:
        return ChartSpec(chart_type=explicit, x=time, y=[measures[0]], series=category)
    if explicit == ChartType.PIE and category:
        return ChartSpec(chart_type=explicit, x=category, y=[measures[0]], sort="desc")
    if explicit in {ChartType.BAR, ChartType.LINE, ChartType.AREA}:
        axis = time or category
        if axis:
            return ChartSpec(
                chart_type=explicit,
                x=axis,
                y=measures,
                series=category if time and category else None,
            )
    if time:
        if category:
            category_profile = next(
                item for item in profile.columns if item.name == category.name
            )
            if category_profile.distinct_count > 8:
                return ChartSpec(
                    chart_type=ChartType.HEATMAP,
                    x=time,
                    y=[measures[0]],
                    series=category,
                )
        return ChartSpec(
            chart_type=ChartType.LINE,
            x=time,
            y=measures,
            series=category,
        )
    if category:
        return ChartSpec(chart_type=ChartType.BAR, x=category, y=measures, sort="desc")
    if len(measures) >= 2:
        return ChartSpec(
            chart_type=ChartType.SCATTER,
            x=measures[0],
            y=[measures[1]],
            sort="none",
        )
    if profile.row_count >= 8:
        return ChartSpec(
            chart_type=ChartType.HISTOGRAM,
            x=measures[0],
            y=[measures[0]],
            sort="none",
        )
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
