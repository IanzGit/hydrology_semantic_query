from __future__ import annotations

from ..contracts import (
    QueryExecutionRecord,
    QueryOutcome,
    QueryTask,
    ReportSectionRequirement,
    ReportTask,
    SemanticColumn,
    SemanticQuery,
    TaskExecutionResult,
    TaskExecutionStatus,
)
from ..report_child.graph import build_report_agent_graph
from ..report_child.node import successful_datasets
from ..report_child.report import report_task_to_markdown


def _task_result(task_id: str, title: str, values: list[float]) -> TaskExecutionResult:
    time_field = f"{task_id}.time"
    value_field = f"{task_id}.value"
    query = SemanticQuery(
        query_mode="view",
        models=[task_id],
        measures=[value_field],
        time_dimensions=[{
            "dimension": time_field,
            "granularity": "day",
            "date_range": ["2026-09-01", "2026-09-02"],
        }],
    )
    record = QueryExecutionRecord(
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
                name=value_field,
                title=title,
                data_type="number",
                member_type="measure",
            ),
        ],
        rows=[
            {time_field: f"2026-09-0{index + 1}", value_field: value}
            for index, value in enumerate(values)
        ],
        row_count=len(values),
        attempt=1,
        selected_models=[task_id],
    )
    return TaskExecutionResult(
        task=QueryTask(task_id=task_id, objective=f"查询{title}"),
        status=TaskExecutionStatus.SUCCESS,
        outcome=QueryOutcome.SUCCESS,
        query_record=record,
        attempts=1,
    )


def _report_task() -> ReportTask:
    results = [
        _task_result("flow", "涌水量", [1.0, 2.0]),
        _task_result("rain", "降雨量", [3.0, 4.0]),
        TaskExecutionResult(
            task=QueryTask(task_id="quality", objective="查询水质"),
            status=TaskExecutionStatus.FAILED,
            summary="查询失败",
        ),
    ]
    return ReportTask(
        original_question="分析涌水、降雨和水质",
        sections=[
            ReportSectionRequirement(
                section_id="overview",
                title="总体情况",
                objective="概括全部可用数据。",
                source_task_ids=["flow", "rain"],
            ),
            ReportSectionRequirement(
                section_id="quality",
                title="水质情况",
                objective="说明水质查询结果。",
                source_task_ids=["quality"],
            ),
        ],
        task_results=results,
    )


def test_report_context_keeps_task_tables_separate_and_marks_unavailable_data() -> None:
    task = _report_task()
    datasets = successful_datasets(task.task_results)

    markdown = report_task_to_markdown(task, datasets)

    assert markdown.index("## 总体情况") < markdown.index("## 水质情况")
    assert "quality（查询水质，数据不可用）" in markdown
    assert "### flow：查询涌水量" in markdown
    assert "### rain：查询降雨量" in markdown
    assert markdown.count("| 时间 |") == 2


def test_report_context_includes_all_query_rows() -> None:
    task = _report_task()
    record = task.task_results[0].query_record
    assert record is not None
    record.rows = [
        {"flow.time": f"t-{index}", "flow.value": index}
        for index in range(60)
    ]
    datasets = successful_datasets(task.task_results)

    markdown = report_task_to_markdown(task, datasets)
    flow_table = markdown.split("### flow：查询涌水量", 1)[1].split(
        "### rain：查询降雨量",
        1,
    )[0]

    assert "| t-49 | 49 |" in flow_table
    assert "| t-50 | 50 |" in flow_table
    assert "| t-59 | 59 |" in flow_table


def test_report_subgraph_plans_validates_and_generates_report() -> None:
    graph = build_report_agent_graph(object())

    assert set(graph.nodes) == {"planner", "validate_render", "report"}
