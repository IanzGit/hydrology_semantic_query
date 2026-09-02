from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import ValidationError

from ..contracts import (
    MainAgentDecision,
    QueryExecutionRecord,
    QueryOutcome,
    QueryTask,
    ReportAnalysisMethod,
    ReportSectionRequirement,
    ReportTask,
    SemanticColumn,
    SemanticQuery,
    SemanticQueryResult,
    TaskExecutionResult,
    TaskExecutionStatus,
)
from ..knowledge import load_business_playbooks, render_business_playbooks
from ..report_child import build_multi_task_analysis


def test_business_playbooks_are_sorted_and_invalid_files_are_skipped(
    tmp_path: Path,
) -> None:
    (tmp_path / "b.md").write_text("场景 B", encoding="utf-8")
    (tmp_path / "a.md").write_text("场景 A", encoding="utf-8")
    (tmp_path / "empty.md").write_text("  ", encoding="utf-8")
    (tmp_path / "invalid.md").write_bytes(b"\xff\xfe")
    (tmp_path / "ignored.txt").write_text("忽略", encoding="utf-8")

    playbooks, warnings = load_business_playbooks(tmp_path)

    assert [playbook.name for playbook in playbooks] == ["a.md", "b.md"]
    assert [playbook.content for playbook in playbooks] == ["场景 A", "场景 B"]
    assert len(warnings) == 2
    rendered = render_business_playbooks(playbooks)
    assert '<business_playbook name="a.md">' in rendered
    assert '<business_playbook name="b.md">' in rendered


def test_missing_business_playbook_directory_is_optional(tmp_path: Path) -> None:
    playbooks, warnings = load_business_playbooks(tmp_path / "missing")

    assert playbooks == ()
    assert warnings == []
    assert render_business_playbooks(playbooks) == "无"


def test_main_decision_enforces_action_specific_shape() -> None:
    with pytest.raises(ValidationError, match="query 动作"):
        MainAgentDecision.model_validate({
            "action": "query",
            "matched_playbook": None,
            "query_tasks": [],
            "report_sections": [],
            "direct_answer": None,
            "summary": "查询",
        })
    with pytest.raises(ValidationError, match="respond 动作"):
        MainAgentDecision.model_validate({
            "action": "respond",
            "matched_playbook": None,
            "query_tasks": [],
            "report_sections": [],
            "direct_answer": None,
            "summary": "回答",
        })
    with pytest.raises(ValidationError, match="query 动作必须包含报告章节"):
        MainAgentDecision.model_validate({
            "action": "query",
            "matched_playbook": None,
            "query_tasks": [{"task_id": "q1", "objective": "查询涌水量"}],
            "report_sections": [],
            "direct_answer": None,
            "summary": "查询",
        })
    with pytest.raises(ValidationError, match="report 动作不能包含查询任务"):
        MainAgentDecision.model_validate({
            "action": "report",
            "matched_playbook": None,
            "query_tasks": [{"task_id": "q1", "objective": "查询涌水量"}],
            "report_sections": [{
                "section_id": "overview",
                "title": "总体情况",
                "objective": "概括结果",
                "source_task_ids": ["q1"],
                "analysis_methods": ["overview"],
            }],
            "direct_answer": None,
            "summary": "报告",
        })


def _record(
    task_id: str,
    measure: str,
    title: str,
    values: list[float],
    *,
    granularity: str | None = "day",
    duplicate_time: bool = False,
) -> QueryExecutionRecord:
    time_field = f"{task_id}.observed_at"
    measure_field = f"{task_id}.{measure}"
    rows = [
        {
            time_field: (
                "2026-08-01T00:00:00+08:00"
                if duplicate_time
                else f"2026-08-{index + 1:02d}T00:00:00+08:00"
            ),
            measure_field: value,
        }
        for index, value in enumerate(values)
    ]
    query = SemanticQuery.model_validate({
        "query_mode": "cube",
        "models": [task_id],
        "measures": [measure_field],
        "time_dimensions": [{
            "dimension": time_field,
            "granularity": granularity,
            "date_range": ["2026-08-01", "2026-08-31"],
        }],
    })
    return QueryExecutionRecord(
        query_number=1,
        task_id=task_id,
        query_goal=f"查询{title}",
        semantic_query=query,
        outcome=QueryOutcome.SUCCESS,
        columns=[
            SemanticColumn(
                name=time_field,
                title="时间",
                data_type="time",
                member_type="time_dimension",
            ),
            SemanticColumn(
                name=measure_field,
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
        task=QueryTask(
            task_id=record.task_id or "unknown",
            objective=record.query_goal,
        ),
        status=TaskExecutionStatus.SUCCESS,
        outcome=QueryOutcome.SUCCESS,
        query_record=record,
        attempts=1,
        summary="成功",
    )


def _report_task(
    records: list[QueryExecutionRecord],
) -> ReportTask:
    return ReportTask(
        original_question="分析涌水与降雨关联",
        sections=[ReportSectionRequirement(
            section_id="correlation",
            title="降雨关联分析",
            objective="分析两个同粒度时间序列的相关性。",
            source_task_ids=[record.task_id or "" for record in records],
            analysis_methods=[ReportAnalysisMethod.CORRELATION],
        )],
        task_results=[_task_result(record) for record in records],
    )


def _aggregate(records: list[QueryExecutionRecord]) -> SemanticQueryResult:
    rows = [row for record in records for row in record.rows]
    columns = [column for record in records for column in record.columns]
    return SemanticQueryResult(
        outcome=QueryOutcome.SUCCESS,
        columns=columns,
        rows=rows,
        row_count=len(rows),
        query_history=records,
    )


def test_cross_task_analysis_aligns_time_and_computes_pearson() -> None:
    flow = _record("flow", "value", "涌水量", [1, 2, 3, 4, 5, 6, 7, 8])
    rain = _record("rain", "value", "降雨量", [2, 4, 6, 8, 10, 12, 14, 16])

    analysis = build_multi_task_analysis(
        _report_task([flow, rain]),
        _aggregate([flow, rain]),
        "Asia/Shanghai",
    )

    correlation = next(
        fact
        for fact in analysis.facts
        if fact.category.value == "correlation"
    )
    assert correlation.value["sample_count"] == 8
    assert correlation.value["coefficient"] == pytest.approx(1.0)
    assert correlation.metadata["source_task_ids"] == ["flow", "rain"]
    assert "相关性不代表因果关系" in correlation.display_text
    assert any(fact.fact_id.startswith("flow-") for fact in analysis.facts)
    assert any(fact.fact_id.startswith("rain-") for fact in analysis.facts)


@pytest.mark.parametrize(
    ("rain", "expected_code"),
    [
        (
            _record(
                "rain",
                "value",
                "降雨量",
                [2, 4, 6, 8, 10, 12, 14, 16],
                granularity="hour",
            ),
            "granularity",
        ),
        (
            _record(
                "rain",
                "value",
                "降雨量",
                [2, 4, 6, 8, 10, 12, 14, 16],
                duplicate_time=True,
            ),
            "series",
        ),
        (
            _record("rain", "value", "降雨量", [2, 4, 6, 8]),
            "sample",
        ),
        (
            _record(
                "rain",
                "value",
                "降雨量",
                [2, 4, 6, 8, 10, 12, 14, 16],
                granularity=None,
            ),
            "granularity",
        ),
    ],
)
def test_cross_task_analysis_refuses_ambiguous_alignment(
    rain: QueryExecutionRecord,
    expected_code: str,
) -> None:
    flow = _record("flow", "value", "涌水量", [1, 2, 3, 4, 5, 6, 7, 8])

    analysis = build_multi_task_analysis(
        _report_task([flow, rain]),
        _aggregate([flow, rain]),
        "Asia/Shanghai",
    )

    assert not any(fact.category.value == "correlation" for fact in analysis.facts)
    assert any(expected_code in limitation.code for limitation in analysis.limitations)


def test_cross_task_analysis_normalizes_timestamps_to_declared_day() -> None:
    flow = _record("flow", "value", "涌水量", [1, 2, 3, 4, 5, 6, 7, 8])
    rain = _record("rain", "value", "降雨量", [2, 4, 6, 8, 10, 12, 14, 16])
    rain.rows = [
        {
            "rain.observed_at": f"2026-08-{index + 1:02d}",
            "rain.value": value,
        }
        for index, value in enumerate([2, 4, 6, 8, 10, 12, 14, 16])
    ]

    analysis = build_multi_task_analysis(
        _report_task([flow, rain]),
        _aggregate([flow, rain]),
        "Asia/Shanghai",
    )

    correlation = next(
        fact
        for fact in analysis.facts
        if fact.category.value == "correlation"
    )
    assert correlation.value["sample_count"] == 8
    assert correlation.value["coefficient"] == pytest.approx(1.0)
