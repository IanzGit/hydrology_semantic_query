from __future__ import annotations

import json
from typing import Any

import pytest
from langchain_core.messages import AIMessage, HumanMessage

from ..contracts import (
    ChartPlanningResponse,
    DynamicChartPlan,
    QueryExecutionRecord,
    QueryOutcome,
    QueryTask,
    ReportSectionRequirement,
    ReportTask,
    SemanticColumn,
    SemanticQuery,
    SemanticQueryResult,
    TaskChartPlans,
    TaskExecutionResult,
    TaskExecutionStatus,
)
from ..report_child.graph import build_report_agent_graph
from ..report_child.node import chart_planner_response_format, successful_datasets
from ..report_child.report import (
    ChartPlanValidationError,
    assign_chart_ids,
    build_task_data_profile,
    build_task_data_profiles,
    ensure_chart_references,
    validate_and_render_chart_plan,
    validate_chart_planning_response,
)


def _column(name: str, title: str, data_type: str = "string") -> SemanticColumn:
    return SemanticColumn(
        name=name,
        title=title,
        data_type=data_type,
        member_type="dimension",
    )


def _result(
    rows: list[dict[str, Any]],
    columns: list[SemanticColumn] | None = None,
) -> SemanticQueryResult:
    return SemanticQueryResult(
        outcome=QueryOutcome.SUCCESS,
        columns=columns or [
            _column("time", "时间", "time"),
            _column("category", "类别"),
            _column("value", "数值", "number"),
            _column("unit", "单位"),
        ],
        rows=rows,
        row_count=len(rows),
    )


def _plan(**updates: Any) -> DynamicChartPlan:
    payload = {
        "title": "数值对比",
        "chart_type": "BAR",
        "priority": 10,
        "x_field": "category",
        "value_field": "value",
        "series_field": None,
        "filters": [],
        "group_by": ["category"],
        "aggregation": "sum",
        "sort": {"by": "value", "direction": "desc"},
        "unit_field": "unit",
        "limit": 30,
    }
    payload.update(updates)
    return DynamicChartPlan.model_validate(payload)


def _task_result(task_id: str, result: SemanticQueryResult) -> TaskExecutionResult:
    query = SemanticQuery(
        query_mode="view",
        models=[task_id],
        dimensions=[column.name for column in result.columns],
        ungrouped=True,
    )
    record = QueryExecutionRecord(
        query_number=1,
        task_id=task_id,
        query_goal=f"查询{task_id}",
        semantic_query=query,
        outcome=QueryOutcome.SUCCESS,
        columns=result.columns,
        rows=result.rows,
        row_count=len(result.rows),
        attempt=1,
        selected_models=[task_id],
    )
    return TaskExecutionResult(
        task=QueryTask(task_id=task_id, objective=f"查询{task_id}"),
        status=TaskExecutionStatus.SUCCESS,
        outcome=QueryOutcome.SUCCESS,
        query_record=record,
        attempts=1,
    )


def test_data_profile_uses_full_rows_and_deterministic_tail_sample() -> None:
    rows = [
        {
            "index": index,
            "value": None if index == 0 else index,
            "category": "常见" if index < 50 else f"类别{index}",
        }
        for index in range(70)
    ]
    result = _result(rows, [
        _column("index", "索引", "number"),
        _column("value", "数值", "number"),
        _column("category", "类别"),
    ])

    first = build_task_data_profile("task_1", "画像", result)
    second = build_task_data_profile("task_1", "画像", result)
    value = next(column for column in first.columns if column.name == "value")
    category = next(column for column in first.columns if column.name == "category")

    assert first == second
    assert value.null_count == 1
    assert value.null_rate == pytest.approx(1 / 70, abs=1e-6)
    assert value.numeric_min == 1
    assert value.numeric_max == 69
    assert category.top_values[0].value == "常见"
    assert category.top_values[0].count == 50
    assert [row["index"] for row in first.first_rows] == list(range(10))
    assert len(first.sampled_rows) == 40
    assert first.sampled_rows[-1]["index"] == 69


def test_chart_schema_requires_visible_grouping_and_count_shape() -> None:
    with pytest.raises(ValueError, match="group_by"):
        _plan(group_by=["category", "unit"])
    with pytest.raises(ValueError, match="count"):
        _plan(aggregation="count", value_field="value")
    with pytest.raises(ValueError, match="value_field"):
        _plan(aggregation="avg", value_field=None)

    schema = chart_planner_response_format()["json_schema"]["schema"]

    assert chart_planner_response_format()["json_schema"]["strict"] is True
    assert schema["additionalProperties"] is False
    assert schema["required"] == ["tasks"]


def test_mixed_units_require_filter_and_missing_units_are_explicit() -> None:
    result = _result([
        {"category": "A", "value": 1, "unit": "m"},
        {"category": "B", "value": 2, "unit": "mm"},
        {"category": "C", "value": 3, "unit": "m"},
    ])

    with pytest.raises(ChartPlanValidationError, match="多个单位"):
        validate_and_render_chart_plan(_plan(), "task_1", result)

    filtered = _plan(filters=[{"field": "unit", "operator": "eq", "value": "m"}])
    chart = validate_and_render_chart_plan(filtered, "task_1", result)
    unknown = validate_and_render_chart_plan(
        _plan(unit_field=None),
        "task_1",
        result,
    )

    assert chart.evidence.unit == "m"
    assert chart.evidence.filtered_row_count == 2
    assert unknown.evidence.unit == "单位未提供"


def test_numeric_quality_duplicate_keys_and_empty_filters_are_rejected() -> None:
    poor = _result([
        {"category": "A", "value": 1, "unit": "m"},
        {"category": "B", "value": "bad", "unit": "m"},
        {"category": "C", "value": 3, "unit": "m"},
    ])
    duplicate = _result([
        {"category": "A", "value": 1, "unit": "m"},
        {"category": "A", "value": 2, "unit": "m"},
    ])

    with pytest.raises(ChartPlanValidationError, match="80%"):
        validate_and_render_chart_plan(_plan(), "task_1", poor)
    with pytest.raises(ChartPlanValidationError, match="组合必须唯一"):
        validate_and_render_chart_plan(
            _plan(aggregation="none"),
            "task_1",
            duplicate,
        )
    with pytest.raises(ChartPlanValidationError, match="过滤后没有数据"):
        validate_and_render_chart_plan(
            _plan(filters=[{"field": "category", "operator": "eq", "value": "Z"}]),
            "task_1",
            duplicate,
        )


@pytest.mark.parametrize(
    ("operator", "value", "field", "expected_rows"),
    [
        ("eq", 2, "value", 1),
        ("ne", 1, "value", 2),
        ("in", [1, 2], "value", 2),
        ("not_in", [1], "value", 2),
        ("gt", 1, "value", 2),
        ("gte", 2, "value", 2),
        ("lt", 3, "value", 2),
        ("lte", 2, "value", 2),
        ("between", [1, 2], "value", 2),
        ("is_null", None, "optional", 2),
        ("not_null", None, "optional", 1),
    ],
)
def test_all_filter_operators_execute_as_and_conditions(
    operator: str,
    value: Any,
    field: str,
    expected_rows: int,
) -> None:
    result = _result([
        {"category": "A", "value": 1, "unit": "m", "optional": None},
        {"category": "B", "value": 2, "unit": "m", "optional": "x"},
        {"category": "C", "value": 3, "unit": "m", "optional": None},
    ], [
        _column("category", "类别"),
        _column("value", "数值", "number"),
        _column("unit", "单位"),
        _column("optional", "可选值"),
    ])
    plan = _plan(
        aggregation="count",
        value_field=None,
        unit_field=None,
        filters=[{"field": field, "operator": operator, "value": value}],
    )

    chart = validate_and_render_chart_plan(plan, "task_1", result)

    assert chart.evidence.filtered_row_count == expected_rows


def test_line_requires_two_times_per_series_and_samples_first_and_last() -> None:
    snapshot = _result([
        {"time": "2026-01-01", "category": "A", "value": 1, "unit": "m"},
        {"time": "2026-01-01", "category": "B", "value": 2, "unit": "m"},
    ])
    line_plan = _plan(
        chart_type="LINE",
        x_field="time",
        series_field="category",
        group_by=["time", "category"],
        aggregation="none",
        sort={"by": "value", "direction": "desc"},
        limit=4,
    )

    with pytest.raises(ChartPlanValidationError, match="至少需要 2 个不同时间点"):
        validate_and_render_chart_plan(line_plan, "task_1", snapshot)

    trend = _result([
        {"time": f"2026-01-{day:02d}", "category": "A", "value": day, "unit": "m"}
        for day in range(1, 11)
    ])
    chart = validate_and_render_chart_plan(
        line_plan.model_copy(update={"series_field": None, "group_by": ["time"]}),
        "task_1",
        trend,
    )
    points = chart.series_data[0]["data"]

    assert chart.evidence.truncated is True
    assert len(points) == 4
    assert points[0]["name"] == "2026-01-01"
    assert points[-1]["name"] == "2026-01-10"
    assert "变化量9" in chart.evidence.summary


def test_pie_constraints_and_bar_topn_are_deterministic() -> None:
    negative = _result([
        {"category": "A", "value": -1, "unit": "m"},
        {"category": "B", "value": 2, "unit": "m"},
    ])
    pie = _plan(chart_type="PIE", aggregation="sum")

    with pytest.raises(ChartPlanValidationError, match="负值"):
        validate_and_render_chart_plan(pie, "task_1", negative)

    result = _result([
        {"category": f"C{index:02d}", "value": index, "unit": "m"}
        for index in range(35)
    ])
    chart = validate_and_render_chart_plan(_plan(limit=10), "task_1", result)
    values = [point["value"] for point in chart.series_data[0]["data"]]

    assert chart.evidence.truncated is True
    assert values == list(range(34, 24, -1))


def test_all_four_frontend_chart_types_render_valid_series() -> None:
    result = _result([
        {"time": "2026-01-01", "category": "A", "value": 1, "unit": "m"},
        {"time": "2026-01-02", "category": "A", "value": 2, "unit": "m"},
        {"time": "2026-01-01", "category": "B", "value": 3, "unit": "m"},
        {"time": "2026-01-02", "category": "B", "value": 4, "unit": "m"},
    ])
    plans = [
        _plan(),
        _plan(
            chart_type="LINE",
            x_field="time",
            series_field="category",
            group_by=["time", "category"],
            aggregation="none",
        ),
        _plan(chart_type="PIE"),
        _plan(
            chart_type="BAR_STACK",
            x_field="time",
            series_field="category",
            group_by=["time", "category"],
        ),
    ]

    charts = [
        validate_and_render_chart_plan(plan, "task_1", result)
        for plan in plans
    ]

    assert [chart.evidence.chart_type.value for chart in charts] == [
        "BAR",
        "LINE",
        "PIE",
        "BAR_STACK",
    ]
    assert all(chart.series_data for chart in charts)


def test_duplicate_plans_and_global_budget_are_rejected_in_priority_order() -> None:
    datasets = []
    profiles = []
    task_plans = []
    for index in range(9):
        task_id = f"task_{index}"
        result = _result([
            {"category": "A", "value": index + 1, "unit": "m"},
            {"category": "B", "value": index + 2, "unit": "m"},
        ])
        datasets.append((task_id, task_id, result))
        profiles.append(build_task_data_profile(task_id, task_id, result))
        task_plans.append(TaskChartPlans(
            source_task_id=task_id,
            charts=[_plan(priority=9 - index)],
        ))
    response = ChartPlanningResponse(tasks=task_plans)

    charts, errors = validate_chart_planning_response(response, datasets, profiles)
    assigned = assign_chart_ids(charts, datasets)

    assert len(assigned) == 8
    assert assigned[0].evidence.source_task_id == "task_8"
    assert any("全局图表预算" in error["error"] for error in errors)

    duplicate_response = ChartPlanningResponse(tasks=[TaskChartPlans(
        source_task_id="task_0",
        charts=[_plan(), _plan(title="同义图")],
    )])
    duplicate_charts, duplicate_errors = validate_chart_planning_response(
        duplicate_response,
        datasets[:1],
        profiles[:1],
    )

    assert len(duplicate_charts) == 1
    assert any("重复" in error["error"] for error in duplicate_errors)


class QueueModel:
    def __init__(self, responses: list[str | Exception]) -> None:
        self.responses = responses
        self.messages: list[list[Any]] = []
        self.bindings: list[dict[str, Any]] = []

    def bind(self, **kwargs: Any) -> QueueModel:
        self.bindings.append(kwargs)
        return self

    async def ainvoke(self, messages: list[Any], config: Any = None) -> AIMessage:
        del config
        self.messages.append(messages)
        response = self.responses.pop(0)
        if isinstance(response, Exception):
            raise response
        return AIMessage(content=response)


class PipelineRuntime:
    def __init__(
        self,
        planner_responses: list[str | Exception],
        report_response: str | Exception,
    ) -> None:
        self.planner = QueueModel(planner_responses)
        self.report = QueueModel([report_response])
        self.streaming: list[bool] = []

    def get_chat_model(self, streaming: bool) -> QueueModel:
        self.streaming.append(streaming)
        return self.report if streaming else self.planner


def _planning_task(task_id: str, charts: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "source_task_id": task_id,
        "charts": charts,
        "no_chart_reason": None if charts else "没有适合的图表",
    }


def _plan_payload(task_id: str, **updates: Any) -> dict[str, Any]:
    plan = _plan(**updates).model_dump(mode="json")
    return _planning_task(task_id, [plan])


@pytest.mark.asyncio
async def test_report_pipeline_repairs_only_invalid_tasks_and_fills_missing_references() -> None:
    first_result = _result([
        {"time": "2026-01-01", "category": "A", "value": 1, "unit": "m"},
        {"time": "2026-01-02", "category": "B", "value": 2, "unit": "m"},
    ])
    second_result = _result([
        {"time": "2026-01-01", "category": "A", "value": 3, "unit": "m"},
        {"time": "2026-01-02", "category": "B", "value": 4, "unit": "m"},
    ])
    task_results = [
        _task_result("first", first_result),
        _task_result("second", second_result),
    ]
    first_response = json.dumps({"tasks": [
        _plan_payload("first"),
        _plan_payload("second", x_field="missing", group_by=["missing"]),
    ]}, ensure_ascii=False)
    repair_response = json.dumps({"tasks": [
        _plan_payload("second", title="修正图表"),
    ]}, ensure_ascii=False)
    runtime = PipelineRuntime(
        [first_response, repair_response],
        "# 分析报告\n\n模型没有引用图表。",
    )
    report_task = ReportTask(
        original_question="比较两个任务",
        sections=[ReportSectionRequirement(
            section_id="overview",
            title="总体情况",
            objective="比较数据",
            source_task_ids=["first", "second"],
        )],
        task_results=task_results,
    )

    output = await build_report_agent_graph(runtime).compile().ainvoke({
        "report_task": report_task,
        "fallback_answer": "查询完成",
        "steps": [],
    })
    correction = json.loads(str(next(
        message
        for message in reversed(runtime.planner.messages[1])
        if isinstance(message, HumanMessage)
    ).content))
    chart_outputs = [
        item for item in output["outputs"] if item["output_type"] == "CHART_OUTPUT"
    ]

    assert runtime.streaming == [False, False, True]
    assert [task["source_task_id"] for task in correction["tasks"]] == ["second"]
    assert len(chart_outputs) == 2
    assert [item["data"]["chartId"] for item in chart_outputs] == [
        "chart-first-01",
        "chart-second-01",
    ]
    assert all(item["data"]["hasData"] for item in chart_outputs)
    assert "数值对比" in output["answer"]
    assert "修正图表" in output["answer"]
    assert "chart-first-01" not in output["answer"]
    assert "chart-second-01" not in output["answer"]
    assert "图表解读" in output["answer"]
    assert not output["warnings"]
    report_prompt = str(runtime.report.messages[0][1].content)
    assert "数据库查询数据" in report_prompt
    assert "图表证据" in report_prompt
    assert "chart-first-01" in report_prompt


@pytest.mark.asyncio
async def test_second_invalid_plan_degrades_to_text_and_table_after_one_retry() -> None:
    result = _result([
        {"category": "A", "value": 1, "unit": "m"},
        {"category": "B", "value": 2, "unit": "m"},
    ])
    no_chart = json.dumps({
        "tasks": [_planning_task("task_1", [])],
    }, ensure_ascii=False)
    runtime = PipelineRuntime(
        [no_chart, no_chart],
        "# 文字报告\n\n保留分析。",
    )
    report_task = ReportTask(
        original_question="分析数据",
        sections=[ReportSectionRequirement(
            section_id="overview",
            title="总体情况",
            objective="分析数据",
            source_task_ids=["task_1"],
        )],
        task_results=[_task_result("task_1", result)],
    )

    output = await build_report_agent_graph(runtime).compile().ainvoke({
        "report_task": report_task,
        "fallback_answer": "查询完成",
        "steps": [],
    })

    assert runtime.streaming == [False, False, True]
    assert not [
        item for item in output["outputs"] if item["output_type"] == "CHART_OUTPUT"
    ]
    assert [
        item for item in output["outputs"] if item["output_type"] == "TABLE_OUTPUT"
    ]
    assert output["answer"].startswith("# 文字报告")
    assert any("没有有效图表计划" in warning for warning in output["warnings"])


@pytest.mark.asyncio
async def test_report_failure_still_returns_planned_chart_and_detail_table() -> None:
    result = _result([
        {"category": "A", "value": 1, "unit": "m"},
        {"category": "B", "value": 2, "unit": "m"},
    ])
    planner_response = json.dumps({
        "tasks": [_plan_payload("task_1")],
    }, ensure_ascii=False)
    runtime = PipelineRuntime([planner_response], TimeoutError("report timeout"))
    report_task = ReportTask(
        original_question="分析数据",
        sections=[ReportSectionRequirement(
            section_id="overview",
            title="总体情况",
            objective="分析数据",
            source_task_ids=["task_1"],
        )],
        task_results=[_task_result("task_1", result)],
    )

    output = await build_report_agent_graph(runtime).compile().ainvoke({
        "report_task": report_task,
        "fallback_answer": "查询完成",
        "steps": [],
    })
    output_types = [item["output_type"] for item in output["outputs"]]

    assert output_types == ["CHART_OUTPUT", "TABLE_OUTPUT"]
    assert output["outputs"][0]["data"]["chartId"] == "chart-task_1-01"
    assert "数值对比" in output["answer"]
    assert "chart-task_1-01" not in output["answer"]
    assert any("Markdown 分析报告生成失败" in warning for warning in output["warnings"])


def test_ensure_chart_references_keeps_complete_report_unchanged() -> None:
    result = _result([
        {"category": "A", "value": 1, "unit": "m"},
        {"category": "B", "value": 2, "unit": "m"},
    ])
    chart = validate_and_render_chart_plan(
        _plan(),
        "task_1",
        result,
        chart_id="chart-task_1-01",
    )
    answer = "根据《数值对比》可知存在差异。"

    assert ensure_chart_references(answer, [chart]) == answer


def test_ensure_chart_references_appends_title_without_chart_id() -> None:
    result = _result([
        {"category": "A", "value": 1, "unit": "m"},
        {"category": "B", "value": 2, "unit": "m"},
    ])
    chart = validate_and_render_chart_plan(
        _plan(),
        "task_1",
        result,
        chart_id="chart-task_1-01",
    )

    answer = ensure_chart_references("# 分析报告", [chart])

    assert "## 图表解读" in answer
    assert "### 数值对比" in answer
    assert chart.evidence.summary in answer
    assert "chart-task_1-01" not in answer


def test_profiles_are_built_per_successful_task_without_combining_rows() -> None:
    first = _task_result("first", _result([
        {"category": "A", "value": 1, "unit": "m"},
        {"category": "B", "value": 2, "unit": "m"},
    ]))
    second = _task_result("second", _result([
        {"category": "C", "value": 9, "unit": "mm"},
        {"category": "D", "value": 10, "unit": "mm"},
    ]))
    datasets = successful_datasets([first, second])

    profiles = build_task_data_profiles(datasets)

    assert [profile.source_task_id for profile in profiles] == ["first", "second"]
    assert profiles[0].row_count == 2
    assert profiles[1].first_rows[0]["value"] == 9
