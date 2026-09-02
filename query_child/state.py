from __future__ import annotations

from typing import Any

from app.agents.state import AgentState

from ..contracts import (
    QueryExecutionRecord,
    QueryOutcome,
    QueryTask,
    RetrievalTrace,
    SemanticCatalogMode,
    SemanticColumn,
    SemanticQuery,
    SemanticQueryError,
    SemanticQueryResult,
    StepRecord,
)
from .models import SemanticCatalog, SemanticContext


class QueryAgentState(AgentState, total=False):
    standalone_question: str | None
    catalog: SemanticCatalog | None
    full_catalog: SemanticCatalog | None
    catalog_mode: SemanticCatalogMode | None
    semantic_context: SemanticContext | None
    retrieval_trace: RetrievalTrace | None
    selected_models: list[str]
    semantic_query: SemanticQuery | None
    previous_query: SemanticQuery | None
    cube_query: dict[str, Any] | None
    compiled_sql: str | None
    compiled_params: list[Any]
    cube_response: dict[str, Any] | None
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
    agent_answer: str
    last_tool_terminal: bool
    search_count: int
    query_count: int
    query_history: list[QueryExecutionRecord]
    current_task: QueryTask | None


__all__ = ["QueryAgentState"]
