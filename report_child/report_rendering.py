from __future__ import annotations

from collections import Counter
from typing import Any

from app.agents.streaming import (
    WorkflowOutputType,
    build_structured_output,
    llm_stream_output,
    table_output,
)

from .models import (
    ChartSpec,
    ChartType,
    FieldRef,
    PlannedBlock,
    PresentationBlockType,
    PresentationPlan,
    QueryOutcome,
    ReportAnalysis,
    ReportAnalysisMethod,
    ReportBlock,
    ReportFact,
    ReportFactCategory,
    ReportNarrativeDraft,
    ReportSection,
    ReportSectionContent,
    ReportSectionNarrative,
    ReportSectionRequirement,
    ResultProfile,
    SectionAnalysis,
    SemanticQueryResult,
    StatusSpec,
    StructuredReport,
    TableSpec,
    VisualizationCandidate,
    VisualizationPlan,
)
from .report_analysis import (
    analyze_result,
    cell_value,
    display_value,
    finite_number,
    time_key,
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


def _pie_supported(result: SemanticQueryResult, category: str, measure: str) -> bool:
    values: dict[str, float] = {}
    for row in result.rows:
        number = finite_number(row.get(measure))
        if row.get(category) is None or number is None or number < 0:
            return False
        label = display_value(row.get(category))
        values[label] = values.get(label, 0.0) + number
    return 2 <= len(values) <= 8 and bool(values)


def build_presentation_plan(result: SemanticQueryResult, question: str, profile: ResultProfile) -> PresentationPlan:
    blocks: list[PlannedBlock] = []
    label_name = profile.primary_category or next(iter(profile.identifier_fields), None)
    for index, status_name in enumerate(profile.status_fields):
        status_profile = next((column for column in profile.columns if column.name == status_name), None)
        status_field = _field(profile, status_name)
        if status_field and status_profile and 2 <= status_profile.distinct_count <= 8:
            blocks.append(PlannedBlock(
                id=f"status-{index + 1}",
                type=PresentationBlockType.STATUS,
                title=f"{status_field.title}分布",
                description="按全部查询结果汇总状态组成。",
                priority=20 + index,
                config=StatusSpec(field=status_field, label_field=_field(profile, label_name)),
            ))
    measure = _field(profile, profile.primary_measure)
    time = _field(profile, profile.primary_time)
    category = _field(profile, profile.primary_category)
    explicit = _explicit_chart_type(question)
    chart: ChartSpec | None = None
    if measure and time and profile.row_count >= 2:
        chart = ChartSpec(chart_type=ChartType.LINE, x=time, y=[measure], series=category, sort="asc", limit=min(max(profile.row_count, 1), 1000))
    elif measure and category and profile.row_count >= 2:
        chart_type = ChartType.PIE if explicit == ChartType.PIE and _pie_supported(result, category.name, measure.name) else ChartType.BAR
        chart = ChartSpec(chart_type=chart_type, x=category, y=[measure], sort="desc", limit=min(max(profile.row_count, 1), 1000))
    if chart:
        blocks.append(PlannedBlock(
            id="primary-chart",
            type=PresentationBlockType.CHART,
            title="时间趋势" if chart.chart_type == ChartType.LINE else "分类对比",
            description="基于完整查询结果生成。",
            priority=40,
            config=chart,
        ))
    blocks.append(PlannedBlock(
        id="details-table",
        type=PresentationBlockType.TABLE,
        title="详细数据",
        description=f"查询共返回 {profile.row_count} 行数据。",
        priority=90,
        config=TableSpec(fields=_fields(profile, [item.name for item in profile.columns]), limit=min(max(profile.row_count, 1), 2000)),
    ))
    return PresentationPlan(title="水文语义查询综合分析报告", profile=profile, blocks=blocks)


def _frontend_chart_output(chart_type: ChartType, chart_name: str, series_data: list[dict[str, Any]]) -> dict[str, Any]:
    return build_structured_output(
        output_type=WorkflowOutputType.CHART_OUTPUT,
        data={
            "chartType": chart_type.value,
            "chartName": chart_name,
            "hasData": True,
            "seriesData": series_data,
        },
    )


def _render_status(plan: PlannedBlock, result: SemanticQueryResult) -> ReportBlock:
    spec = StatusSpec.model_validate(plan.config)
    counts = Counter(display_value(row.get(spec.field.name)) for row in result.rows if row.get(spec.field.name) is not None)
    items = [{"label": label, "count": count} for label, count in counts.most_common(spec.max_items)]
    return ReportBlock(id=plan.id, type=plan.type, title=plan.title, description=plan.description, data={"items": items, "total": sum(counts.values())}, config=spec.model_dump(mode="json", exclude_none=True), priority=plan.priority)


def _render_chart(plan: PlannedBlock, result: SemanticQueryResult) -> ReportBlock:
    spec = ChartSpec.model_validate(plan.config)
    fields = [spec.x, *spec.y] + ([spec.series] if spec.series else [])
    names = list(dict.fromkeys(field.name for field in fields))
    rows = [{name: cell_value(row.get(name)) for name in names} for row in result.rows]
    return ReportBlock(id=plan.id, type=plan.type, title=plan.title, description=plan.description, data={"rows": rows}, config=spec.model_dump(mode="json", exclude_none=True), priority=plan.priority)


def _render_table(plan: PlannedBlock, result: SemanticQueryResult) -> ReportBlock:
    spec = TableSpec.model_validate(plan.config)
    rows = [{field.name: cell_value(row.get(field.name)) for field in spec.fields} for row in result.rows]
    return ReportBlock(id=plan.id, type=plan.type, title=plan.title, description=plan.description, data={"rows": rows, "rowCount": len(rows)}, config=spec.model_dump(mode="json", exclude_none=True), priority=plan.priority)


def _fact_lines(facts: list[ReportFact], categories: set[ReportFactCategory], fallback: str) -> list[str]:
    selected = [fact for fact in facts if fact.category in categories]
    return [f"- {fact.display_text} `[{fact.fact_id}]`" for fact in selected] or [fallback]


def _fact_source_ids(fact: ReportFact) -> set[str]:
    source_ids = set(fact.metadata.get("source_task_ids", []))
    task_id = fact.metadata.get("task_id")
    if task_id:
        source_ids.add(str(task_id))
    return source_ids


def _method_categories(
    methods: list[ReportAnalysisMethod],
) -> set[ReportFactCategory]:
    categories: set[ReportFactCategory] = set()
    mapping = {
        ReportAnalysisMethod.OVERVIEW: {
            ReportFactCategory.SCOPE,
            ReportFactCategory.QUALITY,
            ReportFactCategory.METRIC,
            ReportFactCategory.STATUS,
        },
        ReportAnalysisMethod.TREND: {
            ReportFactCategory.METRIC,
            ReportFactCategory.TREND,
        },
        ReportAnalysisMethod.ANOMALY: {
            ReportFactCategory.THRESHOLD,
            ReportFactCategory.ANOMALY,
        },
        ReportAnalysisMethod.COMPARISON: {
            ReportFactCategory.DISTRIBUTION,
            ReportFactCategory.STATUS,
        },
        ReportAnalysisMethod.CORRELATION: {
            ReportFactCategory.CORRELATION,
        },
        ReportAnalysisMethod.CONCLUSION: set(ReportFactCategory),
    }
    for method in methods:
        categories.update(mapping[method])
    return categories


def _render_custom_markdown(
    question: str,
    analysis: ReportAnalysis,
    narrative: ReportNarrativeDraft | None,
    report_sections: list[ReportSectionRequirement],
) -> str:
    title = narrative.title.strip() if narrative else "水文语义查询综合分析报告"
    lines = [f"# {title}", ""]
    if narrative:
        lines.extend([narrative.executive_summary, ""])
    else:
        lines.extend([
            f"本报告仅呈现由查询结果直接验证的事实。原始问题：{question}",
            "",
        ])
    insights_by_section: dict[str, list[Any]] = {}
    if narrative:
        for insight in narrative.insights:
            if insight.section_id:
                insights_by_section.setdefault(insight.section_id, []).append(insight)
    facts_by_id = {fact.fact_id: fact for fact in analysis.facts}
    certainty_labels = {"high": "高", "medium": "中", "low": "低"}
    for index, section in enumerate(report_sections, 1):
        lines.extend([f"## {index}. {section.title}", ""])
        source_ids = set(section.source_task_ids)
        categories = _method_categories(section.analysis_methods)
        selected_facts = [
            fact
            for fact in analysis.facts
            if fact.category in categories
            and (not source_ids or not _fact_source_ids(fact) or bool(source_ids & _fact_source_ids(fact)))
        ]
        if selected_facts:
            lines.extend(
                f"- {fact.display_text} `[{fact.fact_id}]`"
                for fact in selected_facts
            )
        relevant_limitations = [
            limitation
            for limitation in analysis.limitations
            if not source_ids
            or any(
                limitation.code.startswith(f"{task_id}-")
                for task_id in source_ids
            )
            or limitation.code.startswith(f"{section.section_id}-")
        ]
        lines.extend(
            f"- 分析局限：{limitation.message}"
            for limitation in relevant_limitations
        )
        section_insights = insights_by_section.get(section.section_id, [])
        for insight in section_insights:
            verified = "；".join(
                facts_by_id[fact_id].display_text for fact_id in insight.fact_ids
            )
            lines.extend([
                "",
                f"### {insight.title}",
                "",
                f"- 数据事实：{verified}",
                f"- 分析解释：{insight.interpretation}",
                f"- 影响判断：{insight.impact}",
                f"- 可能原因（推测）：{insight.possible_cause}",
                f"- 条件性建议：{insight.recommendation}",
                f"- 确定性：{certainty_labels[insight.certainty]}",
            ])
        if not selected_facts and not section_insights and not relevant_limitations:
            lines.append(
                "- 当前查询结果不足以支持本章节分析。"
            )
        lines.append("")
    return "\n".join(lines).strip()


def render_markdown(
    question: str,
    result: SemanticQueryResult,
    analysis: ReportAnalysis,
    narrative: ReportNarrativeDraft | None,
    report_sections: list[ReportSectionRequirement] | None = None,
) -> str:
    if report_sections is not None:
        return _render_custom_markdown(
            question,
            analysis,
            narrative,
            report_sections,
        )
    facts = analysis.facts
    sections: list[str] = []
    title = narrative.title.strip() if narrative else "水文语义查询综合分析报告"
    sections.extend([f"# {title}", "", "## 1. 报告概览与查询范围", "", f"- 原始问题：{question}"])
    sections.extend(_fact_lines(facts, {ReportFactCategory.SCOPE}, "- 未获得可用的查询范围信息。"))
    sections.extend(["", "## 2. 执行摘要", ""])
    if narrative:
        sections.append(narrative.executive_summary)
    else:
        sections.append(f"本报告对语义查询返回的全部 {len(result.rows)} 行数据进行了确定性分析。由于结构化叙事未通过校验，本版仅呈现可由数据直接验证的事实。")
    sections.extend(["", "## 3. 数据完整性与质量", ""])
    sections.extend(_fact_lines(facts, {ReportFactCategory.QUALITY}, "- 没有可用数据，无法评估数据质量。"))
    sections.extend(["", "## 4. 核心指标", ""])
    sections.extend(_fact_lines(facts, {ReportFactCategory.METRIC}, "- 结果中没有可用数值字段，无法计算核心指标。"))
    sections.extend(["", "## 5. 趋势分析", ""])
    trend_fallback = next((item.message for item in analysis.limitations if item.code in {"missing_time", "insufficient_trend"}), "当前数据不支持趋势验证。")
    sections.extend(_fact_lines(facts, {ReportFactCategory.TREND}, f"- {trend_fallback}"))
    sections.extend(["", "## 6. 分布与横向对比", ""])
    sections.extend(_fact_lines(facts, {ReportFactCategory.DISTRIBUTION, ReportFactCategory.CORRELATION}, "- 结果中没有可用分类字段或可比数值，无法验证分布和组间差异。"))
    sections.extend(["", "## 7. 状态、阈值与异常分析", ""])
    sections.extend(_fact_lines(facts, {ReportFactCategory.STATUS, ReportFactCategory.THRESHOLD, ReportFactCategory.ANOMALY}, "- 结果不具备可验证的状态、明确阈值或足量异常分析样本。"))
    sections.extend(["", "## 8. 可能原因、影响与条件性建议", ""])
    if narrative and narrative.insights:
        fact_map = {fact.fact_id: fact for fact in facts}
        certainty_labels = {"high": "高", "medium": "中", "low": "低"}
        for insight in narrative.insights:
            verified = "；".join(fact_map[fact_id].display_text for fact_id in insight.fact_ids)
            sections.extend([
                f"### {insight.title}",
                "",
                f"- 数据事实：{verified}",
                f"- 分析解释：{insight.interpretation}",
                f"- 影响判断：{insight.impact}",
                f"- 可能原因（推测）：{insight.possible_cause}",
                f"- 条件性建议：{insight.recommendation}",
                f"- 确定性：{certainty_labels[insight.certainty]}",
                "",
            ])
    else:
        sections.extend([
            "- 可能原因（推测）：当前仅有查询结果，缺少现场工况、设备状态和外部影响证据，不归因。",
            "- 影响判断：仅凭当前结果不将统计变化或异常等同于业务风险。",
            "- 条件性建议：如需采取行动，应先结合明确阈值、现场复核和连续监测结果进行判断。",
        ])
    sections.extend(["", "## 9. 分析局限性", ""])
    sections.extend([f"- {item.message}" for item in analysis.limitations] or ["- 未识别到额外分析局限。"])
    sections.extend(["", "## 10. 详细数据", "", f"详细数据表保留语义查询返回的全部 {len(result.rows)} 行、{len(result.columns)} 个字段，见随报告输出的结构化明细表。"])
    return "\n".join(sections).strip()


def compose_structured_report(result: SemanticQueryResult, question: str, analysis: ReportAnalysis, narrative: ReportNarrativeDraft | None) -> StructuredReport:
    plan = build_presentation_plan(result, question, analysis.profile)
    blocks = []
    for planned in plan.blocks:
        if planned.type == PresentationBlockType.STATUS:
            blocks.append(_render_status(planned, result))
        elif planned.type == PresentationBlockType.CHART:
            blocks.append(_render_chart(planned, result))
        elif planned.type == PresentationBlockType.TABLE:
            blocks.append(_render_table(planned, result))
    sections = []
    visual_blocks = [block for block in blocks if block.type in {PresentationBlockType.STATUS, PresentationBlockType.CHART}]
    detail_blocks = [block for block in blocks if block.type == PresentationBlockType.TABLE]
    if visual_blocks:
        sections.append(ReportSection(id="visualizations", title="可视化分析", blocks=visual_blocks))
    sections.append(ReportSection(id="details", title="详细数据", blocks=detail_blocks))
    markdown = render_markdown(question, result, analysis, narrative)
    return StructuredReport(
        title=narrative.title if narrative else plan.title,
        summary=markdown,
        profile=analysis.profile,
        sections=sections,
        metadata={"rowCount": analysis.profile.row_count, "columnCount": analysis.profile.column_count, "blockCount": len(blocks), "narrative": narrative is not None},
        facts=analysis.facts,
        insights=narrative.insights if narrative else [],
        limitations=analysis.limitations,
    )


def compose_multi_task_structured_report(
    result: SemanticQueryResult,
    question: str,
    analysis: ReportAnalysis,
    narrative: ReportNarrativeDraft | None,
    report_sections: list[ReportSectionRequirement],
    datasets_by_task: dict[str, SemanticQueryResult],
    section_analyses: list[SectionAnalysis],
    visualization_candidates: list[VisualizationCandidate],
    visualization_plan: VisualizationPlan,
) -> StructuredReport:
    """按主 Agent 章节顺序组合文字、图表和必要数据表。"""

    del result, question
    analyses_by_id = {
        item.requirement.section_id: item
        for item in section_analyses
    }
    candidates_by_id = {
        item.candidate_id: item
        for item in visualization_candidates
    }
    plans_by_id = {
        item.section_id: item
        for item in visualization_plan.sections
    }
    sections: list[ReportSection] = []
    for requirement in report_sections:
        section_analysis = analyses_by_id[requirement.section_id]
        section_plan = plans_by_id[requirement.section_id]
        blocks: list[ReportBlock] = []
        for priority, selection in enumerate(section_plan.charts, 30):
            candidate = candidates_by_id[selection.candidate_id]
            blocks.append(candidate.block.model_copy(update={
                "title": candidate.title,
                "description": selection.rationale,
                "priority": priority,
            }, deep=True))
        blocks.extend(_build_section_tables(
            section_analysis,
            datasets_by_task,
            visualization_candidates,
            has_chart=bool(section_plan.charts),
        ))
        sections.append(ReportSection(
            id=requirement.section_id,
            title=requirement.title,
            objective=requirement.objective,
            source_task_ids=requirement.source_task_ids,
            analysis_methods=requirement.analysis_methods,
            fact_ids=[fact.fact_id for fact in section_analysis.facts],
            limitation_codes=[
                limitation.code for limitation in section_analysis.limitations
            ],
            content=_build_section_content(section_analysis, narrative),
            no_chart_reason=section_plan.no_chart_reason,
            blocks=sorted(blocks, key=lambda block: block.priority),
        ))
    title = narrative.title if narrative else "水文语义查询综合分析报告"
    executive_summary = (
        narrative.executive_summary
        if narrative
        else "本报告仅呈现由查询结果直接验证的章节化事实与结论。"
    )
    markdown = "\n\n".join(
        _report_text_fragments(title, executive_summary, sections)
    )
    return StructuredReport(
        protocol_version="1.1",
        title=title,
        summary=markdown,
        profile=analysis.profile,
        sections=sections,
        metadata={
            "rowCount": analysis.profile.row_count,
            "columnCount": analysis.profile.column_count,
            "blockCount": sum(len(section.blocks) for section in sections),
            "taskCount": len(datasets_by_task),
            "sectionCount": len(sections),
            "compositionMode": "section_centric",
            "executiveSummary": executive_summary,
            "narrative": narrative is not None,
        },
        facts=analysis.facts,
        insights=narrative.insights if narrative else [],
        limitations=analysis.limitations,
    )


def _section_narrative(
    section_id: str,
    narrative: ReportNarrativeDraft | None,
) -> ReportSectionNarrative | None:
    if narrative is None:
        return None
    return next(
        (
            item
            for item in narrative.section_narratives
            if item.section_id == section_id
        ),
        None,
    )


def _build_section_content(
    section: SectionAnalysis,
    narrative: ReportNarrativeDraft | None,
) -> ReportSectionContent:
    facts = section.facts
    limitations = section.limitations
    evidence_lines = ["### 数据依据"]
    evidence_lines.extend(
        f"- {fact.display_text} `[{fact.fact_id}]`"
        for fact in facts
    )
    if not facts:
        evidence_lines.append("- 当前来源任务没有提供支持本章节目标的可验证事实。")
    evidence_lines.extend(
        f"- 数据局限：{limitation.message}"
        for limitation in limitations
    )

    generated = _section_narrative(section.requirement.section_id, narrative)
    if generated is not None:
        certainty_label = {
            "high": "高",
            "medium": "中",
            "low": "低",
        }[generated.certainty]
        analysis_lines = [
            "### 分析过程",
            generated.analysis,
            f"影响判断：{generated.impact}",
            f"可能原因（推测）：{generated.possible_cause}",
        ]
        conclusion_lines = [
            "### 结论说明",
            generated.conclusion,
            f"条件性建议：{generated.recommendation}",
            f"确定性：{certainty_label}",
        ]
    else:
        method_text = "、".join(
            method.value for method in section.requirement.analysis_methods
        )
        analysis_lines = [
            "### 分析过程",
            f"本节按照 {method_text} 方法整理上述数据事实；未增加数据中无法验证的归因。",
        ]
        conclusion_lines = [
            "### 结论说明",
            (
                "本章节结论以上述可验证事实为限，需结合所列数据局限解释。"
                if facts
                else "当前数据不足以形成可验证的章节结论。"
            ),
        ]
    return ReportSectionContent(
        evidence="\n\n".join([evidence_lines[0], "\n".join(evidence_lines[1:])]),
        analysis="\n\n".join(analysis_lines),
        conclusion="\n\n".join(conclusion_lines),
    )


def _explicit_table_request(section: ReportSectionRequirement) -> bool:
    text = f"{section.title} {section.objective}"
    return any(
        marker in text
        for marker in ("明细", "记录", "列表", "表格", "数据表", "逐项")
    )


def _exception_row_indexes(section: SectionAnalysis, task_id: str) -> set[int]:
    indexes: set[int] = set()
    for fact in section.facts:
        if fact.category not in {
            ReportFactCategory.THRESHOLD,
            ReportFactCategory.ANOMALY,
        }:
            continue
        if str(fact.metadata.get("task_id") or "") != task_id:
            continue
        value = fact.value if isinstance(fact.value, dict) else {}
        indexes.update(
            int(index)
            for index in value.get("row_indexes", [])
            if isinstance(index, int) and index >= 0
        )
    return indexes


def _table_fields(
    section: SectionAnalysis,
    task_id: str,
    dataset: SemanticQueryResult,
) -> list[FieldRef]:
    names = {
        field
        for fact in section.facts
        if str(fact.metadata.get("task_id") or "") == task_id
        for field in fact.evidence_fields
    }
    profile = section.source_profiles[task_id]
    names.update(
        name
        for name in [
            profile.primary_time,
            profile.primary_category,
            *profile.status_fields,
            *profile.identifier_fields,
        ]
        if name
    )
    if not names:
        names = {column.name for column in dataset.columns}
    return [
        FieldRef(
            name=column.name,
            title=column.title,
            data_type=column.data_type,
        )
        for column in dataset.columns
        if column.name in names
    ]


def _correlation_table(
    section: SectionAnalysis,
    candidates: list[VisualizationCandidate],
) -> ReportBlock | None:
    candidate = next(
        (
            item
            for item in candidates
            if item.section_id == section.requirement.section_id
            and len(item.source_task_ids) > 1
        ),
        None,
    )
    if candidate is None:
        return None
    spec = ChartSpec.model_validate(candidate.block.config)
    fields = [spec.x, *spec.y]
    rows = candidate.block.data.get("rows", [])
    table_spec = TableSpec(fields=fields, limit=min(max(len(rows), 1), 2000))
    return ReportBlock(
        id=f"{section.requirement.section_id}-correlation-table",
        type=PresentationBlockType.TABLE,
        title=f"{section.requirement.title}配对数据",
        description="展示用于相关性计算的同粒度配对样本。",
        data={"rows": rows[: table_spec.limit], "rowCount": len(rows)},
        config=table_spec.model_dump(mode="json"),
        priority=75,
    )


def _build_section_tables(
    section: SectionAnalysis,
    datasets_by_task: dict[str, SemanticQueryResult],
    candidates: list[VisualizationCandidate],
    *,
    has_chart: bool,
) -> list[ReportBlock]:
    methods = set(section.requirement.analysis_methods)
    explicit = _explicit_table_request(section.requirement)
    result: list[ReportBlock] = []
    if ReportAnalysisMethod.CORRELATION in methods and not has_chart:
        correlation = _correlation_table(section, candidates)
        if correlation is not None:
            result.append(correlation)
    fallback_table = (
        not has_chart
        and ReportAnalysisMethod.CONCLUSION not in methods
        and not result
    )
    for task_id in section.available_source_task_ids:
        dataset = datasets_by_task[task_id]
        indexes = _exception_row_indexes(section, task_id)
        if not (explicit or fallback_table or indexes):
            continue
        fields = _table_fields(section, task_id, dataset)
        if not fields:
            continue
        source_rows = (
            [row for index, row in enumerate(dataset.rows) if index in indexes]
            if indexes
            else dataset.rows
        )
        limit = min(max(len(source_rows), 1), 2000)
        rows = [
            {field.name: cell_value(row.get(field.name)) for field in fields}
            for row in source_rows[:limit]
        ]
        spec = TableSpec(fields=fields, limit=limit)
        result.append(ReportBlock(
            id=f"{section.requirement.section_id}-{task_id}-table",
            type=PresentationBlockType.TABLE,
            title=(
                f"{section.requirement.title}异常/阈值记录"
                if indexes
                else f"{section.requirement.title}数据依据"
            ),
            description=f"来源任务 {task_id}，展示 {len(rows)} 行。",
            data={
                "rows": rows,
                "rowCount": len(source_rows),
                "displayedRowCount": len(rows),
            },
            config=spec.model_dump(mode="json"),
            priority=80 + len(result),
        ))
    return result


def _section_lead(section: ReportSection, index: int) -> str:
    content = section.content
    if content is None:
        return f"## {index}. {section.title}"
    chart_titles = [
        block.title
        for block in section.blocks
        if block.type in {PresentationBlockType.CHART, PresentationBlockType.STATUS}
    ]
    visual_note = (
        f"\n\n配套图表：{'、'.join(chart_titles)}。"
        if chart_titles
        else f"\n\n未配置图表：{section.no_chart_reason or '本章节无需图表。'}"
    )
    return (
        f"## {index}. {section.title}\n\n"
        f"章节目标：{section.objective}\n\n"
        f"{content.evidence}\n\n{content.analysis}{visual_note}"
    )


def _report_text_fragments(
    title: str,
    executive_summary: str,
    sections: list[ReportSection],
) -> list[str]:
    fragments = [f"# {title}\n\n{executive_summary}"]
    for index, section in enumerate(sections, 1):
        fragments.append(_section_lead(section, index))
        if section.content is not None:
            fragments.append(section.content.conclusion)
    return fragments


def _status_output(block: ReportBlock) -> dict[str, Any] | None:
    items = block.data.get("items", []) if isinstance(block.data, dict) else []
    points = [{"name": str(item["label"]), "value": int(item["count"])} for item in items if int(item.get("count") or 0) >= 0]
    if not 2 <= len(points) <= 8:
        return None
    return _frontend_chart_output(ChartType.PIE, block.title, [{"name": "记录数", "data": points}])


def _chart_output(block: ReportBlock) -> dict[str, Any] | None:
    spec = ChartSpec.model_validate(block.config)
    rows = block.data.get("rows", []) if isinstance(block.data, dict) else []
    if not rows or not spec.y:
        return None
    measure = spec.y[0]
    if spec.chart_type == ChartType.PIE:
        grouped: dict[str, float] = {}
        for row in rows:
            number = finite_number(row.get(measure.name))
            if number is not None and number >= 0:
                label = display_value(row.get(spec.x.name))
                grouped[label] = grouped.get(label, 0.0) + number
        if not 2 <= len(grouped) <= 8:
            return None
        points = [{"name": label, "value": value} for label, value in sorted(grouped.items(), key=lambda item: item[1], reverse=True)]
        return _frontend_chart_output(spec.chart_type, block.title, [{"name": measure.title, "data": points}])
    valid_rows = [
        row
        for row in rows
        if row.get(spec.x.name) is not None
        and any(finite_number(row.get(field.name)) is not None for field in spec.y)
    ]
    if not valid_rows:
        return None
    valid_rows.sort(key=lambda row: time_key(row.get(spec.x.name)) if spec.chart_type == ChartType.LINE else finite_number(row.get(measure.name)) or 0, reverse=spec.sort == "desc")
    valid_rows = valid_rows[: spec.limit]
    if spec.series:
        labels = list(dict.fromkeys(display_value(row.get(spec.x.name)) for row in valid_rows))
        groups = list(dict.fromkeys(display_value(row.get(spec.series.name)) for row in valid_rows))
        series_data = []
        for field in spec.y:
            for group in groups:
                values = {
                    display_value(row.get(spec.x.name)): finite_number(
                        row.get(field.name)
                    )
                    for row in valid_rows
                    if display_value(row.get(spec.series.name)) == group
                }
                name = group if len(spec.y) == 1 else f"{field.title}—{group}"
                series_data.append({
                    "name": name,
                    "data": [
                        {"name": label, "value": values.get(label)}
                        for label in labels
                    ],
                })
    else:
        series_data = [
            {
                "name": field.title,
                "data": [
                    {
                        "name": display_value(row.get(spec.x.name)),
                        "value": finite_number(row.get(field.name)),
                    }
                    for row in valid_rows
                ],
            }
            for field in spec.y
        ]
    return _frontend_chart_output(spec.chart_type, block.title, series_data)


def _core_metrics_output(report: StructuredReport) -> dict[str, Any] | None:
    facts = [fact for fact in report.facts if fact.category in {ReportFactCategory.METRIC, ReportFactCategory.TREND}]
    if not facts:
        return None
    return table_output(
        table_name="核心指标与趋势事实",
        headers=[
            {"field": "fact_id", "title": "事实ID", "dataType": "string"},
            {"field": "title", "title": "指标", "dataType": "string"},
            {"field": "conclusion", "title": "可验证结论", "dataType": "string"},
            {"field": "unit", "title": "单位", "dataType": "string"},
        ],
        rows=[{"fact_id": fact.fact_id, "title": fact.title, "conclusion": fact.display_text, "unit": fact.unit} for fact in facts],
    )


def _detail_output(block: ReportBlock) -> dict[str, Any]:
    spec = TableSpec.model_validate(block.config)
    rows = block.data.get("rows", []) if isinstance(block.data, dict) else []
    headers = [{"field": field.name, "title": field.title, "dataType": field.data_type} for field in spec.fields]
    return table_output(table_name=block.title, headers=headers, rows=rows)


def render_structured_report(report: StructuredReport) -> list[dict[str, Any]]:
    """把结构化报告转换为按章节交错排列的前端输出事件。"""

    if report.protocol_version == "1.1":
        outputs: list[dict[str, Any]] = [
            llm_stream_output(
                text=(
                    f"# {report.title}\n\n"
                    f"{str(report.metadata.get('executiveSummary') or '').strip()}"
                ).strip()
            )
        ]
        # summary 由相同的章节文字片段拼成；结构化块在分析与结论之间交错发送。
        for index, section in enumerate(report.sections, 1):
            outputs.append(llm_stream_output(text=_section_lead(section, index)))
            for block in sorted(section.blocks, key=lambda item: item.priority):
                if block.type == PresentationBlockType.STATUS:
                    output = _status_output(block)
                elif block.type == PresentationBlockType.CHART:
                    output = _chart_output(block)
                elif block.type == PresentationBlockType.TABLE:
                    output = _detail_output(block)
                else:
                    output = None
                if output is not None:
                    outputs.append(output)
            if section.content is not None:
                outputs.append(llm_stream_output(text=section.content.conclusion))
        return outputs
    outputs: list[dict[str, Any]] = [llm_stream_output(text=report.summary)]
    blocks = sorted([block for section in report.sections for block in section.blocks], key=lambda block: block.priority)
    for block in blocks:
        output = _status_output(block) if block.type == PresentationBlockType.STATUS else _chart_output(block) if block.type == PresentationBlockType.CHART else None
        if output is not None:
            outputs.append(output)
    core = _core_metrics_output(report)
    if core is not None:
        outputs.append(core)
    outputs.extend(_detail_output(block) for block in blocks if block.type == PresentationBlockType.TABLE)
    return outputs


def build_result_outputs(result: SemanticQueryResult, answer: str, question: str) -> list[dict[str, Any]]:
    if result.outcome != QueryOutcome.SUCCESS:
        return [llm_stream_output(text=answer)] if answer else []
    if result.presentation is None:
        result.presentation = compose_structured_report(result, question, analyze_result(result), None)
    return render_structured_report(result.presentation)


__all__ = [
    "build_presentation_plan",
    "build_result_outputs",
    "compose_structured_report",
    "compose_multi_task_structured_report",
    "render_markdown",
    "render_structured_report",
]
