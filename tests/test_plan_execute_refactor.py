from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import ValidationError

from ..contracts import (
    MainAgentDecision,
    QueryExecutionRecord,
    QueryOutcome,
    QueryTask,
    QueryTaskExecutionContext,
    SemanticColumn,
    SemanticQuery,
    TaskExecutionResult,
    TaskExecutionStatus,
)
from ..knowledge import load_business_playbooks, render_business_playbooks
from ..query_child.prompts import _execution_context_payload


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
    with pytest.raises(ValidationError):
        MainAgentDecision.model_validate({
            "action": "report",
            "matched_playbook": None,
            "query_tasks": [{"task_id": "q1", "objective": "查询涌水量"}],
            "report_sections": [{
                "section_id": "overview",
                "title": "总体情况",
                "objective": "概括结果",
                "source_task_ids": ["q1"],
            }],
            "direct_answer": None,
            "summary": "报告",
        })


def test_main_decision_schema_requires_all_top_level_fields() -> None:
    schema = MainAgentDecision.model_json_schema()

    assert set(schema["required"]) == {
        "action",
        "matched_playbook",
        "query_tasks",
        "report_sections",
        "direct_answer",
        "summary",
    }


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


def test_execution_context_includes_all_rows_only_for_dependencies() -> None:
    dependency = _task_result(_record(
        "source",
        "value",
        "来源指标",
        [float(value) for value in range(60)],
    ))
    unrelated = _task_result(_record(
        "unrelated",
        "value",
        "无关指标",
        [1.0, 2.0],
    ))
    current = QueryTask(
        task_id="current",
        objective="使用来源任务结果继续查询",
        depends_on=["source"],
    )
    context = QueryTaskExecutionContext(
        original_question="完成多步水文查询",
        standalone_question="完成多步水文查询",
        plan=[dependency.task, unrelated.task, current],
        current_task=current,
        completed_results=[dependency, unrelated],
    )

    payload = _execution_context_payload({"execution_context": context})

    source_history, unrelated_history = payload["history"]
    assert len(source_history["rows"]) == 60
    assert "rows_truncated" not in source_history
    assert "rows" not in unrelated_history
    assert payload["current_task"]["task_id"] == "current"
