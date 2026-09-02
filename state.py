from __future__ import annotations

from typing import Any

from app.agents.state import AgentState

from .contracts import (
    ExecutionPlanRevision,
    MainAgentAction,
    QueryExecutionRecord,
    QueryOutcome,
    QueryTask,
    ReportSectionRequirement,
    RetrievalTrace,
    SemanticCatalogMode,
    SemanticColumn,
    SemanticQuery,
    SemanticQueryError,
    SemanticQueryResult,
    StepRecord,
    TaskExecutionResult,
)


class HydrologySemanticQueryState(AgentState, total=False):
    standalone_question: str | None
    catalog_mode: SemanticCatalogMode | None
    retrieval_trace: RetrievalTrace | None
    selected_models: list[str]
    semantic_query: SemanticQuery | None
    compiled_sql: str | None
    compiled_params: list[Any]
    columns: list[SemanticColumn]
    rows: list[dict[str, Any]]
    steps: list[StepRecord]
    warnings: list[str]
    attempts: int
    max_rows: int
    stage: str
    error: SemanticQueryError | None
    outcome: QueryOutcome | None
    result: SemanticQueryResult | None
    query_history: list[QueryExecutionRecord]
    matched_playbook: str | None
    plan_revisions: list[ExecutionPlanRevision]
    pending_tasks: list[QueryTask]
    report_sections: list[ReportSectionRequirement]
    task_results: list[TaskExecutionResult]
    current_task: QueryTask | None
    main_action: MainAgentAction | None
    direct_answer: str
    dispatch_count: int
    querying_blocked: bool


__all__ = ["HydrologySemanticQueryState"]
