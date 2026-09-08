from __future__ import annotations

from typing import Any

from typing_extensions import TypedDict

from ..contracts import (
    ReportAnalysis,
    ReportNarrativeDraft,
    ReportTask,
    SectionAnalysis,
    SemanticQueryResult,
    StepRecord,
    StructuredReport,
    VisualizationCandidate,
    VisualizationPlan,
)


class ReportAgentState(TypedDict, total=False):
    report_task: ReportTask
    result: SemanticQueryResult
    aggregate_result: SemanticQueryResult
    datasets: list[tuple[str, str, SemanticQueryResult]]
    datasets_by_task: dict[str, SemanticQueryResult]
    section_analyses: list[SectionAnalysis]
    analysis: ReportAnalysis
    visualization_candidates: list[VisualizationCandidate]
    visualization_plan: VisualizationPlan
    narrative: ReportNarrativeDraft | None
    report: StructuredReport
    answer: str
    warnings: list[str]
    steps: list[StepRecord]
    stream_outputs: list[dict[str, Any]]


__all__ = ["ReportAgentState"]
