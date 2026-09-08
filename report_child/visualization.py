from __future__ import annotations

import json
from collections import Counter
from datetime import date, datetime
from itertools import combinations
from typing import Any
from zoneinfo import ZoneInfo

from langchain_core.messages import HumanMessage, SystemMessage

from app.agents.messages import stringify_message_content

from .models import (
    ChartSpec,
    ChartType,
    FieldRef,
    PresentationBlockType,
    ReportAnalysisMethod,
    ReportBlock,
    ReportFactCategory,
    ReportTask,
    SectionAnalysis,
    SectionVisualizationPlan,
    SemanticQueryResult,
    StatusSpec,
    VisualizationCandidate,
    VisualizationPlan,
    VisualizationSelection,
)
from .report_analysis import (
    build_result_profile,
    cell_value,
    display_value,
    finite_number,
)
from .tool_call_parser import contains_internal_protocol

VISUALIZATION_PLANNER_PROMPT = (
    "你是水文报告的可视化规划器。你只决定每个报告章节是否需要图表、需要几张，以及选择哪个候选。\n"
    "必须遵守：\n"
    "1. 只可使用输入中的 candidate_id，不得创造字段、任务、事实或图表类型。\n"
    "2. 综合章节 objective、analysis_methods、事实结论、局限和实际数据画像选择。\n"
    "3. 每章最多 2 张，全报告最多 8 张；没有图表能有效支持结论时返回空 charts 并说明原因。\n"
    "4. 趋势优先折线图，对比优先柱状图，构成才使用饼图。\n"
    "5. 当前协议没有散点图和异常点专用样式。相关性可用对齐序列折线图近似，"
    "异常可用折线图或柱状图近似；必须在 rationale 中说明表达目的。\n"
    "6. 避免选择表达同一结论的重复图表。只返回符合 JSON Schema 的 JSON。"
)

_MAX_CANDIDATES_PER_SECTION = 12
_MAX_CHARTS_PER_REPORT = 8


def visualization_response_format() -> dict[str, Any]:
    """返回可视化规划器使用的严格 JSON Schema。"""
    return {
        "type": "json_schema",
        "json_schema": {
            "name": "hydrology_section_visualization_plan",
            "strict": True,
            "schema": VisualizationPlan.model_json_schema(),
        },
    }


def _field(section: SectionAnalysis, task_id: str, name: str | None) -> FieldRef | None:
    if not name:
        return None
    profile = section.source_profiles.get(task_id)
    if profile is None:
        return None
    column = next((item for item in profile.columns if item.name == name), None)
    if column is None:
        return None
    return FieldRef(name=column.name, title=column.title, data_type=column.data_type)


def _fact_ids(
    section: SectionAnalysis,
    source_task_ids: list[str],
    fields: list[str],
) -> list[str]:
    sources = set(source_task_ids)
    names = set(fields)
    matched: list[str] = []
    for fact in section.facts:
        fact_sources = {
            str(item)
            for item in fact.metadata.get("source_task_ids", [])
        }
        task_id = fact.metadata.get("task_id")
        if task_id:
            fact_sources.add(str(task_id))
        if fact_sources and not fact_sources.issubset(sources):
            continue
        if fact.evidence_fields and names.isdisjoint(fact.evidence_fields):
            continue
        matched.append(fact.fact_id)
    return matched[:8]


def _chart_rows(result: SemanticQueryResult, fields: list[FieldRef]) -> list[dict[str, Any]]:
    names = list(dict.fromkeys(field.name for field in fields))
    return [
        {name: cell_value(row.get(name)) for name in names}
        for row in result.rows
    ]


def _append_candidate(
    candidates: list[VisualizationCandidate],
    *,
    section: SectionAnalysis,
    candidate_id: str,
    title: str,
    purpose: str,
    source_task_ids: list[str],
    fact_ids: list[str],
    block: ReportBlock,
) -> None:
    if not fact_ids or len(candidates) >= _MAX_CANDIDATES_PER_SECTION:
        return
    candidates.append(VisualizationCandidate(
        candidate_id=candidate_id,
        section_id=section.requirement.section_id,
        title=title,
        purpose=purpose,
        source_task_ids=source_task_ids,
        fact_ids=fact_ids,
        block=block,
    ))


def _pie_supported(result: SemanticQueryResult, category: str, measure: str) -> bool:
    values: dict[str, float] = {}
    for row in result.rows:
        number = finite_number(row.get(measure))
        if row.get(category) is None or number is None or number < 0:
            return False
        label = display_value(row.get(category))
        values[label] = values.get(label, 0.0) + number
    return 2 <= len(values) <= 8


def _source_candidates(
    section: SectionAnalysis,
    task_id: str,
    result: SemanticQueryResult,
) -> list[VisualizationCandidate]:
    candidates: list[VisualizationCandidate] = []
    profile = section.source_profiles[task_id]
    methods = set(section.requirement.analysis_methods)
    broad = bool(methods & {
        ReportAnalysisMethod.OVERVIEW,
        ReportAnalysisMethod.CONCLUSION,
    })
    time_field = _field(section, task_id, profile.primary_time)
    measures = [
        field
        for name in profile.measure_fields[:4]
        if (field := _field(section, task_id, name)) is not None
    ]
    category = _field(section, task_id, profile.primary_category)

    if (
        time_field
        and measures
        and profile.row_count >= 2
        and (broad or methods & {
            ReportAnalysisMethod.TREND,
            ReportAnalysisMethod.ANOMALY,
            ReportAnalysisMethod.CORRELATION,
        })
    ):
        series = None
        if len(measures) == 1 and category is not None:
            category_profile = next(
                item for item in profile.columns if item.name == category.name
            )
            if 2 <= category_profile.distinct_count <= 12:
                series = category
        spec = ChartSpec(
            chart_type=ChartType.LINE,
            x=time_field,
            y=measures,
            series=series,
            sort="asc",
            limit=min(max(profile.row_count, 1), 1000),
        )
        fields = [time_field, *measures] + ([series] if series else [])
        title = f"{section.requirement.title}—时间序列"
        _append_candidate(
            candidates,
            section=section,
            candidate_id=f"{section.requirement.section_id}-{task_id}-line",
            title=title,
            purpose="表达指标随时间变化；异常章节中用于近似观察异常值所在序列。",
            source_task_ids=[task_id],
            fact_ids=_fact_ids(
                section,
                [task_id],
                [field.name for field in fields],
            ),
            block=ReportBlock(
                id=f"{section.requirement.section_id}-{task_id}-line",
                type=PresentationBlockType.CHART,
                title=title,
                description="基于本章节来源任务的完整可绘制时间序列。",
                data={"rows": _chart_rows(result, fields)},
                config=spec.model_dump(mode="json", exclude_none=True),
                priority=40,
            ),
        )

    if (
        category
        and measures
        and profile.row_count >= 2
        and (broad or methods & {
            ReportAnalysisMethod.COMPARISON,
            ReportAnalysisMethod.ANOMALY,
            ReportAnalysisMethod.CORRELATION,
        })
    ):
        spec = ChartSpec(
            chart_type=ChartType.BAR,
            x=category,
            y=measures,
            sort="desc",
            limit=min(max(profile.row_count, 1), 1000),
        )
        fields = [category, *measures]
        title = f"{section.requirement.title}—分类对比"
        _append_candidate(
            candidates,
            section=section,
            candidate_id=f"{section.requirement.section_id}-{task_id}-bar",
            title=title,
            purpose="表达分类之间的指标差异；异常章节中用于近似突出极端类别。",
            source_task_ids=[task_id],
            fact_ids=_fact_ids(
                section,
                [task_id],
                [field.name for field in fields],
            ),
            block=ReportBlock(
                id=f"{section.requirement.section_id}-{task_id}-bar",
                type=PresentationBlockType.CHART,
                title=title,
                description="基于本章节来源任务的分类指标生成。",
                data={"rows": _chart_rows(result, fields)},
                config=spec.model_dump(mode="json", exclude_none=True),
                priority=40,
            ),
        )
        primary_measure = measures[0]
        if (
            (broad or ReportAnalysisMethod.COMPARISON in methods)
            and _pie_supported(result, category.name, primary_measure.name)
        ):
            pie_spec = ChartSpec(
                chart_type=ChartType.PIE,
                x=category,
                y=[primary_measure],
                sort="desc",
                limit=min(max(profile.row_count, 1), 1000),
            )
            pie_title = f"{section.requirement.title}—构成占比"
            _append_candidate(
                candidates,
                section=section,
                candidate_id=f"{section.requirement.section_id}-{task_id}-pie",
                title=pie_title,
                purpose="仅用于表达非负指标的有限类别构成关系。",
                source_task_ids=[task_id],
                fact_ids=_fact_ids(
                    section,
                    [task_id],
                    [category.name, primary_measure.name],
                ),
                block=ReportBlock(
                    id=f"{section.requirement.section_id}-{task_id}-pie",
                    type=PresentationBlockType.CHART,
                    title=pie_title,
                    description="基于本章节来源任务的非负分类指标生成。",
                    data={"rows": _chart_rows(result, [category, primary_measure])},
                    config=pie_spec.model_dump(mode="json", exclude_none=True),
                    priority=45,
                ),
            )

    if broad or ReportAnalysisMethod.COMPARISON in methods:
        for index, status_name in enumerate(profile.status_fields[:3], 1):
            status = _field(section, task_id, status_name)
            status_profile = next(
                item for item in profile.columns if item.name == status_name
            )
            if status is None or not 2 <= status_profile.distinct_count <= 8:
                continue
            counts = Counter(
                display_value(row.get(status.name))
                for row in result.rows
                if row.get(status.name) is not None
            )
            items = [
                {"label": label, "count": count}
                for label, count in counts.most_common(8)
            ]
            title = f"{status.title}分布"
            spec = StatusSpec(
                field=status,
                label_field=category,
            )
            _append_candidate(
                candidates,
                section=section,
                candidate_id=(
                    f"{section.requirement.section_id}-{task_id}-status-{index}"
                ),
                title=title,
                purpose="表达状态类别的数量构成。",
                source_task_ids=[task_id],
                fact_ids=_fact_ids(section, [task_id], [status.name]),
                block=ReportBlock(
                    id=(
                        f"{section.requirement.section_id}-{task_id}-status-{index}"
                    ),
                    type=PresentationBlockType.STATUS,
                    title=title,
                    description="按本章节来源任务统计状态组成。",
                    data={"items": items, "total": sum(counts.values())},
                    config=spec.model_dump(mode="json", exclude_none=True),
                    priority=35 + index,
                ),
            )
    return candidates


def _granularity(result: SemanticQueryResult) -> str | None:
    query = result.semantic_query
    if query is None:
        return None
    values = {item.granularity for item in query.time_dimensions if item.granularity}
    return next(iter(values)) if len(values) == 1 else None


def _time_label(value: Any, timezone: str, granularity: str) -> str | None:
    if isinstance(value, datetime):
        parsed = value
    elif isinstance(value, date):
        parsed = datetime(value.year, value.month, value.day)
    elif isinstance(value, str):
        try:
            parsed = datetime.fromisoformat(value.strip().replace("Z", "+00:00"))
        except ValueError:
            return None
    else:
        return None
    zone = ZoneInfo(timezone)
    parsed = parsed.replace(tzinfo=zone) if parsed.tzinfo is None else parsed.astimezone(zone)
    if granularity == "year":
        return f"{parsed.year:04d}"
    if granularity == "quarter":
        return f"{parsed.year:04d}-Q{(parsed.month - 1) // 3 + 1}"
    if granularity == "month":
        return f"{parsed.year:04d}-{parsed.month:02d}"
    if granularity == "week":
        iso_year, iso_week, _ = parsed.isocalendar()
        return f"{iso_year:04d}-W{iso_week:02d}"
    if granularity == "day":
        return parsed.date().isoformat()
    if granularity == "hour":
        parsed = parsed.replace(minute=0, second=0, microsecond=0)
    elif granularity == "minute":
        parsed = parsed.replace(second=0, microsecond=0)
    elif granularity == "second":
        parsed = parsed.replace(microsecond=0)
    return parsed.isoformat()


def _aligned_values(
    result: SemanticQueryResult,
    timezone: str,
    granularity: str,
) -> tuple[dict[str, float], str, str] | None:
    result_profile = build_result_profile(result)
    if not result_profile.primary_time or len(result_profile.measure_fields) != 1:
        return None
    measure = result_profile.measure_fields[0]
    values: dict[str, float] = {}
    for row in result.rows:
        label = _time_label(row.get(result_profile.primary_time), timezone, granularity)
        number = finite_number(row.get(measure))
        if label is None or number is None or label in values:
            return None
        values[label] = number
    if not values:
        return None
    title = next(
        (column.title for column in result.columns if column.name == measure),
        measure,
    )
    return values, measure, title


def _cross_source_candidates(
    section: SectionAnalysis,
    datasets_by_task: dict[str, SemanticQueryResult],
    timezone: str,
) -> list[VisualizationCandidate]:
    if ReportAnalysisMethod.CORRELATION not in section.requirement.analysis_methods:
        return []
    candidates: list[VisualizationCandidate] = []
    for index, (left_id, right_id) in enumerate(
        combinations(section.available_source_task_ids, 2),
        1,
    ):
        left = datasets_by_task[left_id]
        right = datasets_by_task[right_id]
        granularity = _granularity(left)
        if not granularity or granularity != _granularity(right):
            continue
        left_data = _aligned_values(left, timezone, granularity)
        right_data = _aligned_values(right, timezone, granularity)
        if left_data is None or right_data is None:
            continue
        left_values, _, left_title = left_data
        right_values, _, right_title = right_data
        shared = sorted(set(left_values) & set(right_values))
        if len(shared) < 8:
            continue
        time_field = FieldRef(name="aligned_time", title="对齐时间", data_type="time")
        left_field = FieldRef(name=f"series_{index}_left", title=left_title, data_type="number")
        right_field = FieldRef(name=f"series_{index}_right", title=right_title, data_type="number")
        rows = [
            {
                time_field.name: label,
                left_field.name: left_values[label],
                right_field.name: right_values[label],
            }
            for label in shared
        ]
        spec = ChartSpec(
            chart_type=ChartType.LINE,
            x=time_field,
            y=[left_field, right_field],
            sort="asc",
            limit=min(len(rows), 1000),
        )
        fact_ids = [
            fact.fact_id
            for fact in section.facts
            if fact.category == ReportFactCategory.CORRELATION
            and set(fact.metadata.get("source_task_ids", [])) == {left_id, right_id}
        ][:8]
        title = f"{left_title}与{right_title}对齐序列"
        _append_candidate(
            candidates,
            section=section,
            candidate_id=f"{section.requirement.section_id}-correlation-{index}",
            title=title,
            purpose="以同粒度时间对齐的多序列折线图近似表达共同变化。",
            source_task_ids=[left_id, right_id],
            fact_ids=fact_ids,
            block=ReportBlock(
                id=f"{section.requirement.section_id}-correlation-{index}",
                type=PresentationBlockType.CHART,
                title=title,
                description="相关性近似视图；折线共同变化不代表因果关系。",
                data={"rows": rows},
                config=spec.model_dump(mode="json", exclude_none=True),
                priority=30,
            ),
        )
    return candidates


def build_visualization_candidates(
    section_analyses: list[SectionAnalysis],
    datasets_by_task: dict[str, SemanticQueryResult],
    timezone: str,
) -> list[VisualizationCandidate]:
    """为每个章节生成有限、字段已校验的可视化候选。"""
    result: list[VisualizationCandidate] = []
    for section in section_analyses:
        local: list[VisualizationCandidate] = []
        local.extend(_cross_source_candidates(section, datasets_by_task, timezone))
        for task_id in section.available_source_task_ids:
            local.extend(_source_candidates(section, task_id, datasets_by_task[task_id]))
        result.extend(local[:_MAX_CANDIDATES_PER_SECTION])
    return result


def _fallback_section_plan(
    section: SectionAnalysis,
    candidates: list[VisualizationCandidate],
) -> SectionVisualizationPlan:
    local = [
        candidate
        for candidate in candidates
        if candidate.section_id == section.requirement.section_id
    ]
    if not local:
        return SectionVisualizationPlan(
            section_id=section.requirement.section_id,
            charts=[],
            no_chart_reason="本章节没有满足现有 BAR/LINE/PIE 协议的有效候选。",
        )
    selected = local[0]
    return SectionVisualizationPlan(
        section_id=section.requirement.section_id,
        charts=[VisualizationSelection(
            candidate_id=selected.candidate_id,
            rationale="可视化规划失败后选择首个与章节方法匹配的安全候选。",
        )],
    )


def fallback_visualization_plan(
    section_analyses: list[SectionAnalysis],
    candidates: list[VisualizationCandidate],
) -> VisualizationPlan:
    """在模型不可用时按章节生成最多一张图的确定性计划。"""
    sections: list[SectionVisualizationPlan] = []
    remaining = _MAX_CHARTS_PER_REPORT
    for section in section_analyses:
        planned = _fallback_section_plan(section, candidates)
        if planned.charts and remaining <= 0:
            planned = SectionVisualizationPlan(
                section_id=section.requirement.section_id,
                charts=[],
                no_chart_reason="已达到全报告图表数量上限。",
            )
        remaining -= len(planned.charts)
        sections.append(planned)
    return VisualizationPlan(sections=sections)


def _planner_packet(
    report_task: ReportTask,
    section_analyses: list[SectionAnalysis],
    candidates: list[VisualizationCandidate],
) -> dict[str, Any]:
    candidates_by_section: dict[str, list[dict[str, Any]]] = {}
    for candidate in candidates:
        config = dict(candidate.block.config)
        candidates_by_section.setdefault(candidate.section_id, []).append({
            "candidate_id": candidate.candidate_id,
            "title": candidate.title,
            "purpose": candidate.purpose,
            "chart_type": config.get("chart_type", "PIE"),
            "source_task_ids": candidate.source_task_ids,
            "fact_ids": candidate.fact_ids,
            "config": config,
        })
    return {
        "question": report_task.original_question,
        "sections": [
            {
                "requirement": section.requirement.model_dump(mode="json"),
                "source_profiles": {
                    task_id: profile.model_dump(mode="json")
                    for task_id, profile in section.source_profiles.items()
                },
                "facts": [fact.model_dump(mode="json") for fact in section.facts],
                "limitations": [
                    limitation.model_dump(mode="json")
                    for limitation in section.limitations
                ],
                "candidates": candidates_by_section.get(
                    section.requirement.section_id,
                    [],
                ),
            }
            for section in section_analyses
        ],
    }


def validate_visualization_plan(
    raw: str,
    section_analyses: list[SectionAnalysis],
    candidates: list[VisualizationCandidate],
) -> tuple[VisualizationPlan, set[str]]:
    """校验模型选择并仅对无效章节应用确定性降级。"""
    if not raw.strip():
        raise ValueError("可视化规划模型返回了空响应")
    if contains_internal_protocol(raw):
        raise ValueError("可视化规划模型返回了内部工具协议")
    draft = VisualizationPlan.model_validate_json(raw)
    expected = {
        section.requirement.section_id: section
        for section in section_analyses
    }
    supplied: dict[str, SectionVisualizationPlan] = {}
    for section in draft.sections:
        if section.section_id not in expected:
            raise ValueError(f"可视化规划包含未知章节: {section.section_id}")
        if section.section_id in supplied:
            raise ValueError(f"可视化规划重复章节: {section.section_id}")
        supplied[section.section_id] = section
    candidates_by_id = {candidate.candidate_id: candidate for candidate in candidates}
    normalized: list[SectionVisualizationPlan] = []
    fallback_ids: set[str] = set()
    remaining = _MAX_CHARTS_PER_REPORT
    for section_id, analysis in expected.items():
        planned = supplied.get(section_id)
        valid = planned is not None
        if planned is not None:
            seen: set[str] = set()
            normalized_selections: list[VisualizationSelection] = []
            for selection in planned.charts:
                candidate = candidates_by_id.get(selection.candidate_id)
                if (
                    candidate is None
                    or candidate.section_id != section_id
                    or selection.candidate_id in seen
                    or not set(candidate.source_task_ids).issubset(
                        analysis.requirement.source_task_ids
                    )
                ):
                    valid = False
                    break
                seen.add(selection.candidate_id)
                normalized_selections.append(VisualizationSelection(
                    candidate_id=candidate.candidate_id,
                    rationale=candidate.purpose,
                ))
            if valid:
                planned = planned.model_copy(update={
                    "charts": normalized_selections,
                    "no_chart_reason": (
                        None
                        if normalized_selections
                        else (
                            "现有候选不能有效支持本章节结论，按规划不配置图表。"
                            if any(
                                candidate.section_id == section_id
                                for candidate in candidates
                            )
                            else "本章节没有满足现有 BAR/LINE/PIE 协议的有效候选。"
                        )
                    ),
                })
        if not valid:
            fallback_ids.add(section_id)
            planned = _fallback_section_plan(analysis, candidates)
        assert planned is not None
        if len(planned.charts) > remaining:
            planned = SectionVisualizationPlan(
                section_id=section_id,
                charts=planned.charts[:remaining],
                no_chart_reason=(
                    None if remaining else "已达到全报告图表数量上限。"
                ),
            )
        remaining -= len(planned.charts)
        normalized.append(planned)
    return VisualizationPlan(sections=normalized), fallback_ids


async def generate_visualization_plan(
    runtime: Any,
    report_task: ReportTask,
    section_analyses: list[SectionAnalysis],
    candidates: list[VisualizationCandidate],
) -> tuple[VisualizationPlan, set[str]]:
    """调用报告模型完成章节级可视化选择并验证输出。"""
    messages = [
        SystemMessage(content=VISUALIZATION_PLANNER_PROMPT),
        HumanMessage(content=json.dumps(
            _planner_packet(report_task, section_analyses, candidates),
            ensure_ascii=False,
            separators=(",", ":"),
            default=str,
        )),
    ]
    model = runtime.get_chat_model(streaming=True).bind(
        extra_body={"enable_thinking": False},
        response_format=visualization_response_format(),
    )
    response = await model.ainvoke(messages, config={"callbacks": []})
    return validate_visualization_plan(
        stringify_message_content(response.content).strip(),
        section_analyses,
        candidates,
    )


__all__ = [
    "VISUALIZATION_PLANNER_PROMPT",
    "build_visualization_candidates",
    "fallback_visualization_plan",
    "generate_visualization_plan",
    "validate_visualization_plan",
    "visualization_response_format",
]
