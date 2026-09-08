from __future__ import annotations

import json

import pytest

from ..contracts import (
    QueryExecutionRecord,
    QueryOutcome,
    QueryTask,
    ReportAnalysisMethod,
    ReportSectionRequirement,
    ReportTask,
    SectionVisualizationPlan,
    SemanticColumn,
    SemanticQuery,
    SemanticQueryResult,
    TaskExecutionResult,
    TaskExecutionStatus,
    VisualizationPlan,
    VisualizationSelection,
)
from ..report_child.node import build_section_analyses
from ..report_child.report import validate_narrative
from ..report_child.report_rendering import (
    compose_multi_task_structured_report,
    render_structured_report,
)
from ..report_child.visualization import (
    build_visualization_candidates,
    validate_visualization_plan,
)


def _record(task_id: str, title: str, factor: float = 1.0) -> QueryExecutionRecord:
    time_name = f"{task_id}.time"
    value_name = f"{task_id}.value"
    query = SemanticQuery.model_validate({
        "query_mode": "cube",
        "models": [task_id],
        "measures": [value_name],
        "time_dimensions": [{
            "dimension": time_name,
            "granularity": "day",
            "date_range": ["2026-08-01", "2026-08-08"],
        }],
    })
    rows = [
        {
            time_name: f"2026-08-{index:02d}",
            value_name: index * factor,
        }
        for index in range(1, 9)
    ]
    return QueryExecutionRecord(
        query_number=1,
        task_id=task_id,
        query_goal=f"查询{title}",
        semantic_query=query,
        outcome=QueryOutcome.SUCCESS,
        columns=[
            SemanticColumn(
                name=time_name,
                title="时间",
                data_type="time",
                member_type="time_dimension",
            ),
            SemanticColumn(
                name=value_name,
                title=title,
                data_type="number",
                member_type="measure",
            ),
        ],
        rows=rows,
        row_count=len(rows),
        attempt=1,
        selected_models=[task_id],
    )


def _task_result(record: QueryExecutionRecord) -> TaskExecutionResult:
    return TaskExecutionResult(
        task=QueryTask(task_id=record.task_id or "missing", objective=record.query_goal),
        status=TaskExecutionStatus.SUCCESS,
        outcome=QueryOutcome.SUCCESS,
        query_record=record,
        attempts=1,
        summary="成功",
    )


def _dataset(record: QueryExecutionRecord) -> SemanticQueryResult:
    return SemanticQueryResult(
        outcome=QueryOutcome.SUCCESS,
        semantic_query=record.semantic_query,
        columns=record.columns,
        rows=record.rows,
        row_count=record.row_count,
        selected_models=record.selected_models,
    )


def _fixture() -> tuple[ReportTask, SemanticQueryResult]:
    flow = _record("flow", "涌水量")
    rain = _record("rain", "降雨量", 2.0)
    report_task = ReportTask(
        original_question="分析涌水趋势及其与降雨的关系",
        sections=[
            ReportSectionRequirement(
                section_id="trend",
                title="涌水趋势",
                objective="分析涌水量时间变化。",
                source_task_ids=["flow"],
                analysis_methods=[ReportAnalysisMethod.TREND],
            ),
            ReportSectionRequirement(
                section_id="correlation",
                title="降雨关联",
                objective="分析涌水量与降雨量的相关变化。",
                source_task_ids=["flow", "rain"],
                analysis_methods=[ReportAnalysisMethod.CORRELATION],
            ),
        ],
        task_results=[_task_result(flow), _task_result(rain)],
    )
    aggregate = SemanticQueryResult(
        outcome=QueryOutcome.SUCCESS,
        columns=[*flow.columns, *rain.columns],
        rows=[*flow.rows, *rain.rows],
        row_count=16,
    )
    return report_task, aggregate


def test_section_analysis_is_scoped_and_query_can_be_reused() -> None:
    report_task, aggregate = _fixture()

    sections, analysis = build_section_analyses(
        report_task,
        aggregate,
        "Asia/Shanghai",
    )

    assert [item.requirement.section_id for item in sections] == [
        "trend",
        "correlation",
    ]
    assert all(fact.fact_id.startswith("trend-flow-") for fact in sections[0].facts)
    correlation = next(
        fact for fact in sections[1].facts if fact.category.value == "correlation"
    )
    assert set(correlation.metadata["source_task_ids"]) == {"flow", "rain"}
    assert all(
        fact.metadata.get("section_id") == "correlation"
        for fact in sections[1].facts
    )
    assert len(analysis.facts) == sum(len(item.facts) for item in sections)


def test_visualization_plan_rejects_cross_section_candidate_locally() -> None:
    report_task, aggregate = _fixture()
    sections, _ = build_section_analyses(
        report_task,
        aggregate,
        "Asia/Shanghai",
    )
    datasets = {
        result.task.task_id: _dataset(result.query_record)
        for result in report_task.task_results
        if result.query_record is not None
    }
    candidates = build_visualization_candidates(
        sections,
        datasets,
        "Asia/Shanghai",
    )
    correlation_candidate = next(
        item for item in candidates if len(item.source_task_ids) == 2
    )
    raw = json.dumps({
        "sections": [
            {
                "section_id": "trend",
                "charts": [{
                    "candidate_id": correlation_candidate.candidate_id,
                    "rationale": "不应通过校验。",
                }],
                "no_chart_reason": None,
            },
            {
                "section_id": "correlation",
                "charts": [],
                "no_chart_reason": "使用文字说明。",
            },
        ],
    }, ensure_ascii=False)

    plan, fallback_ids = validate_visualization_plan(raw, sections, candidates)

    assert fallback_ids == {"trend"}
    assert plan.sections[0].charts
    assert plan.sections[0].charts[0].candidate_id.startswith("trend-flow-")
    assert plan.sections[1].charts == []


def test_structured_report_and_outputs_follow_section_order() -> None:
    report_task, aggregate = _fixture()
    sections, analysis = build_section_analyses(
        report_task,
        aggregate,
        "Asia/Shanghai",
    )
    datasets = {
        result.task.task_id: _dataset(result.query_record)
        for result in report_task.task_results
        if result.query_record is not None
    }
    candidates = build_visualization_candidates(
        sections,
        datasets,
        "Asia/Shanghai",
    )
    trend_candidate = next(item for item in candidates if item.section_id == "trend")
    correlation_candidate = next(
        item for item in candidates if len(item.source_task_ids) == 2
    )
    plan = VisualizationPlan(sections=[
        SectionVisualizationPlan(
            section_id="trend",
            charts=[VisualizationSelection(
                candidate_id=trend_candidate.candidate_id,
                rationale="表达涌水量随时间变化。",
            )],
        ),
        SectionVisualizationPlan(
            section_id="correlation",
            charts=[VisualizationSelection(
                candidate_id=correlation_candidate.candidate_id,
                rationale="用对齐折线近似表达共同变化。",
            )],
        ),
    ])

    report = compose_multi_task_structured_report(
        aggregate,
        report_task.original_question,
        analysis,
        None,
        report_task.sections,
        datasets,
        sections,
        candidates,
        plan,
    )
    outputs = render_structured_report(report)

    assert report.protocol_version == "1.1"
    assert [section.id for section in report.sections] == ["trend", "correlation"]
    assert [section.source_task_ids for section in report.sections] == [
        ["flow"],
        ["flow", "rain"],
    ]
    assert [output["output_type"] for output in outputs] == [
        "LLM_STREAM",
        "LLM_STREAM",
        "CHART_OUTPUT",
        "LLM_STREAM",
        "LLM_STREAM",
        "CHART_OUTPUT",
        "LLM_STREAM",
    ]
    assert "## 1. 涌水趋势" in outputs[1]["data"]["text"]
    assert "### 结论说明" in outputs[3]["data"]["text"]
    assert "## 2. 降雨关联" in outputs[4]["data"]["text"]


def test_report_section_requires_at_least_one_source_task() -> None:
    with pytest.raises(ValueError, match="at least 1 item"):
        ReportSectionRequirement(
            section_id="empty",
            title="空章节",
            objective="没有来源。",
            source_task_ids=[],
            analysis_methods=[ReportAnalysisMethod.OVERVIEW],
        )


def test_section_report_requires_complete_section_narratives() -> None:
    report_task, aggregate = _fixture()
    _, analysis = build_section_analyses(
        report_task,
        aggregate,
        "Asia/Shanghai",
    )
    raw = json.dumps({
        "title": "水文报告",
        "executive_summary": "仅依据已验证事实。",
        "insights": [],
        "section_narratives": [],
    }, ensure_ascii=False)

    with pytest.raises(ValueError, match="必须覆盖全部规划章节"):
        validate_narrative(
            raw,
            analysis,
            report_task.sections,
            require_section_narratives=True,
        )


def test_section_narrative_cannot_reuse_fact_from_another_section() -> None:
    report_task, aggregate = _fixture()
    sections, analysis = build_section_analyses(
        report_task,
        aggregate,
        "Asia/Shanghai",
    )
    trend_fact_id = sections[0].facts[0].fact_id

    def section_narrative(section_id: str) -> dict[str, object]:
        return {
            "section_id": section_id,
            "fact_ids": [trend_fact_id],
            "analysis": "依据所引事实进行分析。",
            "impact": "影响范围限于查询结果。",
            "possible_cause": "原因仍需核验。",
            "conclusion": "结论以所引事实为限。",
            "recommendation": "建议结合现场信息复核。",
            "certainty": "medium",
        }

    raw = json.dumps({
        "title": "水文报告",
        "executive_summary": "仅依据已验证事实。",
        "insights": [],
        "section_narratives": [
            section_narrative("trend"),
            section_narrative("correlation"),
        ],
    }, ensure_ascii=False)

    with pytest.raises(ValueError, match="引用了其他章节的事实"):
        validate_narrative(
            raw,
            analysis,
            report_task.sections,
            require_section_narratives=True,
        )
