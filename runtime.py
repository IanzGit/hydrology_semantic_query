from __future__ import annotations

import logging
import re
import time
from typing import Any

from langchain_core.messages import AIMessage, HumanMessage
from typing_extensions import TypedDict

from app.agents.messages import stringify_message_content
from app.agents.state import AgentState
from app.agents.streaming import chain_of_thought_output

from .client import CubeClient
from .config import HYDROLOGY_SEMANTIC_QUERY_ID, HydrologySemanticQuerySettings
from .models import (
    FailureKind,
    QueryOutcome,
    RetrievalTrace,
    SemanticCatalog,
    SemanticCatalogMode,
    SemanticColumn,
    SemanticContext,
    SemanticQuery,
    SemanticQueryError,
    SemanticQueryResult,
    StepRecord,
    StepStatus,
)


class HydrologySemanticQueryServices:
    def __init__(
        self,
        settings: HydrologySemanticQuerySettings,
        client: CubeClient | None = None,
        embedding_client: Any | None = None,
    ) -> None:
        self.settings = settings
        self.client = client or CubeClient(
            base_url=settings.cube_url,
            token=settings.cube_token,
            timeout_seconds=settings.timeout_seconds,
            continue_wait_retries=settings.continue_wait_retries,
            meta_cache_ttl_seconds=settings.meta_cache_ttl_seconds,
        )
        self.embedding = embedding_client
        self.startup_warnings: list[str] = []
        self.catalog_search: Any | None = None

class HydrologySemanticQueryState(AgentState, total=False):
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

class RequestData(TypedDict):
    question: str
    business_knowledge: str | None
    conversation_context: Any
    catalog_mode: SemanticCatalogMode | None
    catalog_metadata_filters: dict[str, Any]
    max_rows: int


def request_data(
    state: HydrologySemanticQueryState,
    settings: HydrologySemanticQuerySettings,
) -> RequestData:
    metadata = state.get("metadata") or {}
    question = str(metadata.get("original_query") or state.get("query") or "").strip()
    if not question:
        raise ValueError("查询问题不能为空")
    raw_max_rows = metadata.get("maxRows", metadata.get("max_rows", settings.max_rows))
    max_rows = int(raw_max_rows)
    if max_rows < 1:
        raise ValueError("maxRows 必须大于 0")
    mode_value = metadata.get("catalog_mode")
    catalog_mode = SemanticCatalogMode(mode_value) if mode_value else None
    raw_filters = metadata.get("catalog_metadata_filters")
    conversation_context = metadata.get("conversation_context")
    if conversation_context is None:
        turns: list[dict[str, str]] = []
        for message in state.get("messages", []):
            role = (
                "user"
                if isinstance(message, HumanMessage)
                else "assistant"
                if isinstance(message, AIMessage)
                else None
            )
            content = stringify_message_content(message.content).strip()
            if role and content and not (role == "user" and content == question):
                turns.append({"role": role, "content": content})
        conversation_context = {"messages": turns} if turns else None
    return {
        "question": question,
        "business_knowledge": metadata.get("business_knowledge"),
        "conversation_context": conversation_context,
        "catalog_mode": catalog_mode,
        "catalog_metadata_filters": raw_filters if isinstance(raw_filters, dict) else {},
        "max_rows": max_rows,
    }

_SENSITIVE_RESPONSE_VALUE = re.compile(
    r"(?i)([\"']?(?:token|authorization|api[_-]?key|cookie|cube_token)[\"']?)(\s*[:=]\s*)(?:(?:bearer\s+)?(?:\"[^\"]*\"|'[^']*'|[^,\s}\]]+))"
)
_BEARER_TOKEN = re.compile(r"(?i)\bbearer\s+[^\s,]+")
_RESPONSE_EXCERPT_LIMIT = 4096


def safe_response_excerpt(value: str) -> str:
    redacted = _SENSITIVE_RESPONSE_VALUE.sub(r"\1\2***", value)
    redacted = _BEARER_TOKEN.sub("Bearer ***", redacted)
    return redacted[:_RESPONSE_EXCERPT_LIMIT]


def build_error(
    *,
    stage: str,
    code: str,
    kind: FailureKind,
    exc: Exception | str,
    retryable: bool = False,
    status_code: int | None = None,
    details: dict[str, Any] | None = None,
) -> SemanticQueryError:
    return SemanticQueryError(
        stage=stage,
        code=code,
        kind=kind,
        internal_message=str(exc)[:1000],
        internal_details=details or {},
        retryable=retryable,
        status_code=status_code,
    )


def outcome_for_error(error: SemanticQueryError | None) -> QueryOutcome:
    if error is None:
        return QueryOutcome.SYSTEM_ERROR
    if error.kind in {FailureKind.PLANNER, FailureKind.VALIDATION}:
        return QueryOutcome.PLANNER_ERROR
    if error.kind == FailureKind.EXECUTION:
        return QueryOutcome.EXECUTION_ERROR
    return QueryOutcome.SYSTEM_ERROR

logger = logging.getLogger("uvicorn.error")


def build_step(
    stage: str,
    started: float,
    *,
    attempt: int,
    status: StepStatus,
    summary: str | None = None,
    metadata: dict[str, Any] | None = None,
) -> StepRecord:
    step = StepRecord(
        stage=stage,
        status=status,
        duration_ms=round((time.perf_counter() - started) * 1000, 3),
        attempt=attempt,
        summary=summary,
        metadata=metadata or {},
    )
    logger.info(
        "hydrology_semantic_query node timing: stage=%s status=%s attempt=%s duration_ms=%.3f",
        step.stage,
        step.status.value,
        step.attempt,
        step.duration_ms,
    )
    return step


def thought_output(text: str, detail: str) -> list[dict[str, Any]]:
    return [
        chain_of_thought_output(
            step_type="analysis",
            text=text,
            detail=detail,
            intent_id=HYDROLOGY_SEMANTIC_QUERY_ID,
        )
    ]
