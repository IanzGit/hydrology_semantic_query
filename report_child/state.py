from __future__ import annotations

from typing import Any

from typing_extensions import TypedDict

from ..contracts import (
    ChartPlanningResponse,
    RenderedChart,
    ReportTask,
    StepRecord,
    TaskDataProfile,
)


class ReportAgentState(TypedDict, total=False):
    report_task: ReportTask
    fallback_answer: str
    answer: str
    data_profiles: list[TaskDataProfile]
    chart_planning_response: ChartPlanningResponse
    planner_messages: list[Any]
    planner_raw_response: str
    planner_errors: list[dict[str, str]]
    planner_attempts: int
    rendered_charts: list[RenderedChart]
    warnings: list[str]
    steps: list[StepRecord]
    outputs: list[dict[str, Any]]


__all__ = ["ReportAgentState"]
