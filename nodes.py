from __future__ import annotations

import asyncio
import json
import logging
import re
import time
from contextlib import suppress
from typing import Any

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from openai import APIError
from typing_extensions import TypedDict

from app.agents.messages import stringify_message_content
from app.agents.state import AgentState
from app.agents.streaming import chain_of_thought_output

from .config import HydrologySemanticQuerySettings
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
from .output import (
    REPORT_FAILURE_WARNING,
    build_result_outputs,
    generate_report,
    normalize_cube_response,
)
from .semantic_catalog import catalog_from_meta
from .semantic_catalog_retriever import (
    EmbeddingClient,
    RetrievedSemanticContext,
    SemanticCatalogRetriever,
    SemanticContextRetrievalError,
    SentenceTransformerEmbedding,
    merge_retrieved_context,
)
from .semantic_cube_client import CubeClient, CubeClientError
from .semantic_query_planner import (
    StructuredOutputParseError,
    build_messages,
    parse_semantic_query,
    semantic_query_response_format,
)
from .semantic_query_validator import SemanticQueryValidationError, validate_semantic_query

logger = logging.getLogger("uvicorn.error")

_SENSITIVE_RESPONSE_VALUE = re.compile(
    r"(?i)([\"']?(?:token|authorization|api[_-]?key|cookie|cube_token)[\"']?)(\s*[:=]\s*)(?:(?:bearer\s+)?(?:\"[^\"]*\"|'[^']*'|[^,\s}\]]+))"
)
_BEARER_TOKEN = re.compile(r"(?i)\bbearer\s+[^\s,]+")
_RESPONSE_EXCERPT_LIMIT = 4096
_CONTEXT_REFRESH_CODES = frozenset({
    "unknown_model",
    "unknown_member",
    "member_scope_mismatch",
    "member_type_mismatch",
    "join_unreachable",
})


def _safe_response_excerpt(value: str) -> str:
    redacted = _SENSITIVE_RESPONSE_VALUE.sub(r"\1\2***", value)
    redacted = _BEARER_TOKEN.sub("Bearer ***", redacted)
    return redacted[:_RESPONSE_EXCERPT_LIMIT]


class HydrologySemanticQueryState(AgentState, total=False):
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
    max_attempts: int
    max_rows: int
    stage: str
    error: SemanticQueryError | None
    outcome: QueryOutcome | None
    result: SemanticQueryResult | None


class RequestData(TypedDict):
    question: str
    business_knowledge: str | None
    conversation_context: Any
    catalog_mode: SemanticCatalogMode | None
    catalog_metadata_filters: dict[str, Any]
    max_rows: int
    report: Any


class HydrologySemanticQueryServices:
    def __init__(
        self,
        settings: HydrologySemanticQuerySettings,
        client: CubeClient | None = None,
        embedding_client: EmbeddingClient | None = None,
    ) -> None:
        self.settings = settings
        self.client = client or CubeClient(
            base_url=settings.cube_url,
            token=settings.cube_token,
            timeout_seconds=settings.timeout_seconds,
            continue_wait_retries=settings.continue_wait_retries,
            meta_cache_ttl_seconds=settings.meta_cache_ttl_seconds,
        )
        self.startup_warnings: list[str] = []
        self._retriever_lock = asyncio.Lock()
        self.embedding = embedding_client
        if self.embedding is None and settings.embedding_model:
            try:
                self.embedding = SentenceTransformerEmbedding(settings.embedding_model)
            except Exception as exc:
                warning = (
                    "嵌入模型不可用，已改用统一词法目录检索。"
                    f"原因：{str(exc)[:200]}"
                )
                self.startup_warnings.append(warning)
                logger.warning("hydrology_semantic_query startup warning: %s", warning)
        self.retriever: SemanticCatalogRetriever | None = None

    async def retrieve_context(
        self,
        question: str,
        catalog: SemanticCatalog,
        *,
        mode: SemanticCatalogMode | None = None,
        metadata_filters: dict[str, Any] | None = None,
        limit: int | None = None,
        retrieval_round: int = 1,
    ) -> RetrievedSemanticContext:
        async with self._retriever_lock:
            if self.retriever is None or self.retriever.catalog != catalog:
                self.retriever = SemanticCatalogRetriever(
                    catalog,
                    context_top_k=self.settings.context_top_k,
                    vector_index_path=self.settings.vector_index_path,
                    embedding_client=self.embedding,
                    mode=self.settings.catalog_mode,
                    embedding_batch_size=self.settings.embedding_batch_size,
                    embedding_concurrency=self.settings.embedding_concurrency,
                    auto_full_context_max_chars=self.settings.auto_full_context_max_chars,
                )
            retriever = self.retriever
        return await retriever.retrieve(
            question,
            mode=mode,
            metadata_filters=metadata_filters,
            limit=limit,
            retrieval_round=retrieval_round,
        )


def _request(
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
        "report": metadata.get("report"),
    }


def _boolean(value: Any, default: bool) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"1", "true", "yes", "on"}:
            return True
        if normalized in {"0", "false", "no", "off"}:
            return False
    return default


def _extend_unique(target: list[str], values: list[str]) -> None:
    target.extend(value for value in values if value not in target)


def _step(
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


def _thought(text: str, detail: str) -> list[dict[str, Any]]:
    from .agent import HYDROLOGY_SEMANTIC_QUERY_ID

    return [chain_of_thought_output(
        step_type="analysis",
        text=text,
        detail=detail,
        intent_id=HYDROLOGY_SEMANTIC_QUERY_ID,
    )]


def _reset() -> dict[str, Any]:
    return {
        "answer": "",
        "catalog": None,
        "full_catalog": None,
        "catalog_mode": None,
        "semantic_context": None,
        "retrieval_trace": None,
        "selected_models": [],
        "semantic_query": None,
        "previous_query": None,
        "cube_query": None,
        "compiled_sql": None,
        "compiled_params": [],
        "cube_response": None,
        "columns": [],
        "rows": [],
        "steps": [],
        "warnings": [],
        "attempts": 0,
        "max_attempts": 1,
        "max_rows": 0,
        "stage": "catalog_prepare",
        "error": None,
        "outcome": None,
        "result": None,
    }


def _error(
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


def _outcome_for_error(error: SemanticQueryError | None) -> QueryOutcome:
    if error is None:
        return QueryOutcome.SYSTEM_ERROR
    if error.kind in {FailureKind.PLANNER, FailureKind.VALIDATION}:
        return QueryOutcome.PLANNER_ERROR
    if error.kind == FailureKind.EXECUTION:
        return QueryOutcome.EXECUTION_ERROR
    return QueryOutcome.SYSTEM_ERROR


def _is_llm_infrastructure_error(exc: Exception) -> bool:
    return isinstance(exc, (APIError, TimeoutError, ConnectionError))


def _error_feedback(error: SemanticQueryError | None) -> str | None:
    if error is None:
        return None
    return json.dumps(
        {
            "stage": error.stage,
            "code": error.code,
            "kind": error.kind.value,
            "status_code": error.status_code,
            "message": error.internal_message,
            "details": error.internal_details,
        },
        ensure_ascii=False,
        default=str,
    )


def _retrieval_metadata(retrieved: RetrievedSemanticContext) -> dict[str, Any]:
    item_counts: dict[str, int] = {}
    for item in retrieved.context.items:
        item_counts[item.item_type] = item_counts.get(item.item_type, 0) + 1
    return {
        "catalog_mode": retrieved.mode.value,
        "strategy": retrieved.context.strategy.value,
        "retrieval_round": retrieved.context.retrieval_round,
        "context_item_count": len(retrieved.context.items),
        "context_item_counts": item_counts,
        "retrieval_queries": retrieved.trace.queries,
        "retrieval_hits": [
            hit.model_dump(mode="json", exclude_none=True)
            for hit in retrieved.trace.hits
        ],
        "accessible_model_count": len(retrieved.catalog.models),
        "index_source": retrieved.trace.index_source,
    }


def make_catalog_prepare_node(services: HydrologySemanticQueryServices):
    async def prepare_catalog(state: HydrologySemanticQueryState) -> dict[str, Any]:
        reset = _reset()
        started = time.perf_counter()
        try:
            request = _request(state, services.settings)
            meta = await services.client.get_meta()
            full_catalog = catalog_from_meta(meta)
            step = _step(
                "catalog_prepare",
                started,
                attempt=1,
                status=StepStatus.SUCCESS,
                metadata={
                    "total_model_count": len(full_catalog.models),
                    "view_count": sum(
                        model.model_type == "view"
                        for model in full_catalog.models.values()
                    ),
                    "cube_count": sum(
                        model.model_type == "cube"
                        for model in full_catalog.models.values()
                    ),
                },
            )
            return {
                **reset,
                "full_catalog": full_catalog,
                "steps": [step],
                "warnings": list(services.startup_warnings),
                "max_attempts": services.settings.max_retries + 1,
                "max_rows": request["max_rows"],
                "stream_outputs": _thought(
                    "准备语义目录",
                    f"已加载 {len(full_catalog.models)} 个受治理的公开 View/Cube",
                ),
            }
        except Exception as exc:
            code = exc.code if isinstance(exc, CubeClientError) else exc.__class__.__name__
            status = exc.status_code if isinstance(exc, CubeClientError) else None
            step = _step(
                "catalog_prepare",
                started,
                attempt=1,
                status=StepStatus.FAILED,
                summary=str(exc)[:1000],
            )
            return {
                **reset,
                "steps": [step],
                "max_attempts": services.settings.max_retries + 1,
                "stage": "catalog_prepare",
                "error": _error(
                    stage="catalog_prepare",
                    code=code,
                    kind=FailureKind.SYSTEM,
                    exc=exc,
                    status_code=status,
                ),
                "outcome": QueryOutcome.SYSTEM_ERROR,
                "stream_outputs": _thought("加载语义模型", "水文语义模型加载失败"),
            }

    return prepare_catalog


def make_retrieval_node(services: HydrologySemanticQueryServices):
    async def retrieve_context(state: HydrologySemanticQueryState) -> dict[str, Any]:
        started = time.perf_counter()
        steps = list(state["steps"])
        warnings = list(state["warnings"])
        try:
            request = _request(state, services.settings)
            assert state["full_catalog"] is not None
            retrieved = await services.retrieve_context(
                request["question"],
                state["full_catalog"],
                mode=request["catalog_mode"],
                metadata_filters=request["catalog_metadata_filters"],
            )
            _extend_unique(warnings, retrieved.warnings)
            steps.append(_step(
                "semantic_retrieval",
                started,
                attempt=1,
                status=StepStatus.SUCCESS,
                metadata=_retrieval_metadata(retrieved),
            ))
            return {
                "catalog": retrieved.catalog,
                "catalog_mode": retrieved.mode,
                "semantic_context": retrieved.context,
                "retrieval_trace": retrieved.trace,
                "selected_models": [],
                "steps": steps,
                "warnings": warnings,
                "stage": "semantic_retrieval",
                "error": None,
                "stream_outputs": _thought(
                    "构建语义上下文",
                    f"已提供 {len(retrieved.context.items)} 个相关目录上下文项，由 Planner 全局决策",
                ),
            }
        except Exception as exc:
            validation_error = isinstance(exc, SemanticContextRetrievalError)
            error = _error(
                stage="semantic_retrieval",
                code=exc.__class__.__name__,
                kind=FailureKind.VALIDATION if validation_error else FailureKind.SYSTEM,
                exc=exc,
            )
            steps.append(_step(
                "semantic_retrieval",
                started,
                attempt=1,
                status=StepStatus.FAILED,
                summary=str(exc)[:1000],
            ))
            return {
                "steps": steps,
                "stage": "semantic_retrieval",
                "error": error,
                "outcome": _outcome_for_error(error),
                "stream_outputs": _thought("构建语义上下文", "语义上下文检索失败"),
            }

    return retrieve_context


def make_generation_node(runtime, services: HydrologySemanticQueryServices):
    async def generate_semantic_query(state: HydrologySemanticQueryState) -> dict[str, Any]:
        started = time.perf_counter()
        steps = list(state["steps"])
        attempt = state["attempts"] + 1
        response_text = ""
        error_stage = "llm_invocation_error"
        validation_errors: list[dict[str, Any]] = []
        error_summary = "SemanticQuery 模型调用失败"
        try:
            request = _request(state, services.settings)
            assert state["semantic_context"] is not None
            messages: list[BaseMessage] = build_messages(
                question=request["question"],
                context=state["semantic_context"],
                business_knowledge=request["business_knowledge"],
                conversation_context=request["conversation_context"],
                max_rows=min(request["max_rows"], services.settings.hard_max_rows),
                previous_query=state.get("previous_query"),
                previous_error=_error_feedback(state.get("error")),
            )
            model = runtime.get_chat_model(streaming=False).bind(
                response_format=semantic_query_response_format(),
                extra_body={"enable_thinking": False},
            )
            response = await model.ainvoke(messages, config={"callbacks": []})
            response_text = stringify_message_content(response.content)
            if not response_text.strip():
                error_stage = "empty_response"
                error_summary = "SemanticQuery 模型响应为空"
                raise ValueError(error_summary)
            try:
                query = parse_semantic_query(response_text)
            except StructuredOutputParseError as exc:
                error_stage = exc.stage
                validation_errors = exc.validation_errors
                error_summary = (
                    "SemanticQuery JSON 格式无效"
                    if exc.stage == "json_syntax_error"
                    else "SemanticQuery 未通过结构校验"
                )
                raise
            query_text = json.dumps(
                query.model_dump(mode="json", by_alias=True, exclude_none=True),
                ensure_ascii=False,
                indent=2,
            )
            logger.info(
                "hydrology_semantic_query generated query: attempt=%s query=%s",
                attempt,
                query_text,
            )
            steps.append(_step(
                "semantic_generation",
                started,
                attempt=attempt,
                status=StepStatus.SUCCESS,
                metadata={
                    "catalog_mode": state["catalog_mode"].value,
                    "context_strategy": state["semantic_context"].strategy.value,
                    "context_item_count": len(state["semantic_context"].items),
                    "retrieval_round": state["semantic_context"].retrieval_round,
                },
            ))
            return {
                "semantic_query": query,
                "steps": steps,
                "attempts": attempt,
                "stage": "semantic_generation",
                "error": None,
                "stream_outputs": _thought(
                    "生成语义查询",
                    f"已完成第 {attempt} 次 SemanticQuery 生成\n\n```json\n{query_text}\n```",
                ),
            }
        except Exception as exc:
            infrastructure_error = (
                error_stage == "llm_invocation_error"
                or _is_llm_infrastructure_error(exc)
            )
            failure_kind = FailureKind.SYSTEM if infrastructure_error else FailureKind.PLANNER
            retryable = not infrastructure_error and error_stage in {
                "empty_response",
                "json_syntax_error",
                "schema_validation_error",
            }
            exception_message = _safe_response_excerpt(str(exc) or error_summary)
            logger.warning(
                "hydrology_semantic_query generation failure: attempt=%s stage=%s raw_response_excerpt=%r validation_errors=%s exception_type=%s exception_message=%r",
                attempt,
                error_stage,
                _safe_response_excerpt(response_text),
                validation_errors,
                exc.__class__.__name__,
                exception_message,
            )
            steps.append(_step(
                "semantic_generation",
                started,
                attempt=attempt,
                status=StepStatus.FAILED,
                summary=error_summary,
                metadata={
                    "catalog_mode": state["catalog_mode"].value if state.get("catalog_mode") else None,
                    "error_stage": error_stage,
                    "failure_kind": failure_kind.value,
                    "retryable": retryable,
                    "exception_type": exc.__class__.__name__,
                },
            ))
            return {
                "steps": steps,
                "attempts": attempt,
                "stage": "semantic_generation",
                "error": _error(
                    stage="semantic_generation",
                    code=exc.__class__.__name__ if infrastructure_error else error_stage,
                    kind=failure_kind,
                    exc=exception_message,
                    retryable=retryable,
                    details={
                        "error_stage": error_stage,
                        "exception_type": exc.__class__.__name__,
                        "validation_errors": validation_errors,
                        "raw_response_excerpt": _safe_response_excerpt(response_text),
                    },
                ),
                "stream_outputs": _thought("生成语义查询", "SemanticQuery 生成或解析失败"),
            }

    return generate_semantic_query


def make_validation_node(services: HydrologySemanticQueryServices):
    async def validate_query(state: HydrologySemanticQueryState) -> dict[str, Any]:
        started = time.perf_counter()
        steps = list(state["steps"])
        warnings = list(state["warnings"])
        try:
            assert state["semantic_query"] is not None and state["catalog"] is not None
            validated = validate_semantic_query(
                state["semantic_query"],
                state["catalog"],
                requested_max_rows=state["max_rows"],
                hard_max_rows=services.settings.hard_max_rows,
            )
            warnings.extend(validated.warnings)
            steps.append(_step(
                "semantic_validation",
                started,
                attempt=state["attempts"],
                status=StepStatus.SUCCESS,
                metadata={"models": validated.query.models},
            ))
            return {
                "semantic_query": validated.query,
                "selected_models": validated.query.models,
                "steps": steps,
                "warnings": warnings,
                "stage": "semantic_validation",
                "error": None,
                "stream_outputs": _thought(
                    "校验语义查询",
                    "完整可访问目录、成员类型、查询形态和行数限制校验通过",
                ),
            }
        except Exception as exc:
            validation_error = isinstance(exc, SemanticQueryValidationError)
            code = (
                exc.code
                if validation_error
                else exc.__class__.__name__
            )
            steps.append(_step(
                "semantic_validation",
                started,
                attempt=state["attempts"],
                status=StepStatus.FAILED,
                summary=str(exc)[:1000],
                metadata={"code": code},
            ))
            return {
                "steps": steps,
                "stage": "semantic_validation",
                "error": _error(
                    stage="semantic_validation",
                    code=code,
                    kind=(
                        FailureKind.VALIDATION
                        if validation_error
                        else FailureKind.SYSTEM
                    ),
                    exc=exc,
                    retryable=validation_error,
                ),
                "stream_outputs": _thought("校验语义查询", "SemanticQuery 未通过本地校验"),
            }

    return validate_query


def make_compilation_node(services: HydrologySemanticQueryServices):
    async def compile_query(state: HydrologySemanticQueryState) -> dict[str, Any]:
        started = time.perf_counter()
        steps = list(state["steps"])
        try:
            assert state["semantic_query"] is not None
            cube_query = state["semantic_query"].to_cube_query()
            cube_query["timezone"] = services.settings.timezone
            sql, params = await services.client.get_sql(cube_query)
            logger.info(
                "hydrology_semantic_query compiled sql: attempt=%s sql=%s params=%s",
                state["attempts"],
                sql,
                json.dumps(params, ensure_ascii=False, default=str),
            )
            steps.append(_step(
                "semantic_compilation",
                started,
                attempt=state["attempts"],
                status=StepStatus.SUCCESS,
                metadata={"sql_generated": True},
            ))
            return {
                "cube_query": cube_query,
                "compiled_sql": sql,
                "compiled_params": params,
                "steps": steps,
                "stage": "semantic_compilation",
                "error": None,
                "stream_outputs": _thought(
                    "编译语义查询",
                    f"SemanticQuery 已通过 Cube /sql 编译预检\n\n```sql\n{sql}\n```",
                ),
            }
        except Exception as exc:
            retryable = isinstance(exc, CubeClientError) and exc.retryable_by_model
            code = exc.code if isinstance(exc, CubeClientError) else exc.__class__.__name__
            status = exc.status_code if isinstance(exc, CubeClientError) else None
            kind = (
                FailureKind.VALIDATION
                if retryable
                else FailureKind.EXECUTION
                if isinstance(exc, CubeClientError)
                else FailureKind.SYSTEM
            )
            steps.append(_step(
                "semantic_compilation",
                started,
                attempt=state["attempts"],
                status=StepStatus.FAILED,
                summary=str(exc)[:1000],
            ))
            return {
                "cube_query": None,
                "compiled_sql": None,
                "compiled_params": [],
                "steps": steps,
                "stage": "semantic_compilation",
                "error": _error(
                    stage="semantic_compilation",
                    code=code,
                    kind=kind,
                    exc=exc,
                    retryable=retryable,
                    status_code=status,
                ),
                "stream_outputs": _thought("编译语义查询", "SemanticQuery 无法通过 Cube /sql 预检"),
            }

    return compile_query


def make_execution_node(services: HydrologySemanticQueryServices):
    async def execute_cube(state: HydrologySemanticQueryState) -> dict[str, Any]:
        started = time.perf_counter()
        steps = list(state["steps"])
        try:
            assert state["semantic_query"] is not None
            assert state["cube_query"] is not None
            response = await services.client.load(state["cube_query"])
            columns, rows = normalize_cube_response(response)
            rows = rows[: state["semantic_query"].limit]
            empty_result = not rows
            steps.append(_step(
                "cube_execution",
                started,
                attempt=state["attempts"],
                status=StepStatus.SUCCESS,
                metadata={
                    "row_count": len(rows),
                    "catalog_mode": state["catalog_mode"].value,
                    "query_mode": state["semantic_query"].query_mode.value,
                },
            ))
            return {
                "cube_response": response,
                "columns": columns,
                "rows": rows,
                "steps": steps,
                "stage": "cube_execution",
                "error": None,
                "outcome": QueryOutcome.NO_DATA if empty_result else QueryOutcome.SUCCESS,
                "stream_outputs": _thought(
                    "执行语义查询",
                    "查询结果为空，按当前语义返回无数据"
                    if empty_result
                    else f"Cube 查询成功，返回 {len(rows)} 行",
                ),
            }
        except Exception as exc:
            retryable = isinstance(exc, CubeClientError) and exc.retryable_by_model
            code = exc.code if isinstance(exc, CubeClientError) else exc.__class__.__name__
            status = exc.status_code if isinstance(exc, CubeClientError) else None
            steps.append(_step(
                "cube_execution",
                started,
                attempt=state["attempts"],
                status=StepStatus.FAILED,
                summary=str(exc)[:1000],
            ))
            return {
                "steps": steps,
                "stage": "cube_execution",
                "error": _error(
                    stage="cube_execution",
                    code=code,
                    kind=FailureKind.EXECUTION,
                    exc=exc,
                    retryable=retryable,
                    status_code=status,
                ),
                "stream_outputs": _thought("执行语义查询", "语义查询执行失败"),
            }

    return execute_cube


def _should_refresh_context(error: SemanticQueryError) -> bool:
    if error.stage == "semantic_validation":
        return error.code in _CONTEXT_REFRESH_CODES
    return error.retryable and error.stage in {
        "semantic_compilation",
        "cube_execution",
    }


def make_recovery_node(services: HydrologySemanticQueryServices):
    async def recover(state: HydrologySemanticQueryState) -> dict[str, Any]:
        started = time.perf_counter()
        steps = list(state["steps"])
        warnings = list(state["warnings"])
        error = state.get("error")
        can_retry = bool(
            error and error.retryable and state["attempts"] < state["max_attempts"]
        )
        refresh_context = bool(error and _should_refresh_context(error))
        steps.append(_step(
            "failure_recovery",
            started,
            attempt=max(1, state["attempts"]),
            status=StepStatus.SUCCESS if can_retry else StepStatus.SKIPPED,
            summary=(
                "已安排语义上下文扩展后重试"
                if can_retry and refresh_context
                else "已将结构化错误反馈给 Planner"
                if can_retry
                else "不可重试或已达最大重试次数"
            ),
            metadata={
                "error_stage": error.stage if error else None,
                "error_code": error.code if error else None,
                "context_refresh": refresh_context,
            },
        ))
        if not can_retry:
            return {
                "steps": steps,
                "warnings": warnings,
                "outcome": _outcome_for_error(error),
                "stream_outputs": _thought("恢复查询流程", "无法继续修正，正在整理结构化失败"),
            }
        previous_query = state.get("semantic_query") or state.get("previous_query")
        warnings.append(
            f"第 {state['attempts']} 次尝试在 {error.stage if error else 'unknown'} 阶段失败，已反馈给 Planner。"
        )
        updates: dict[str, Any] = {
            "steps": steps,
            "warnings": warnings,
            "previous_query": previous_query,
            "selected_models": [],
            "semantic_query": None,
            "cube_query": None,
            "compiled_sql": None,
            "compiled_params": [],
            "outcome": None,
        }
        if not refresh_context:
            updates["stream_outputs"] = _thought(
                "恢复查询流程",
                "已记录结构化失败反馈，准备重新生成 SemanticQuery",
            )
            return updates
        refresh_started = time.perf_counter()
        try:
            request = _request(state, services.settings)
            full_catalog = state.get("full_catalog")
            catalog = state.get("catalog")
            context = state.get("semantic_context")
            trace = state.get("retrieval_trace")
            catalog_mode = state.get("catalog_mode")
            assert full_catalog is not None
            assert catalog is not None
            assert context is not None
            assert trace is not None
            assert catalog_mode is not None
            feedback = _error_feedback(error) or "未知错误"
            refreshed = await services.retrieve_context(
                f"{request['question']}\n结构化错误反馈：{feedback}",
                full_catalog,
                mode=request["catalog_mode"],
                metadata_filters=request["catalog_metadata_filters"],
                limit=40,
                retrieval_round=context.retrieval_round + 1,
            )
            current = RetrievedSemanticContext(
                mode=catalog_mode,
                catalog=catalog,
                context=context,
                trace=trace,
            )
            merged = merge_retrieved_context(current, refreshed)
            _extend_unique(warnings, merged.warnings)
            steps.append(_step(
                "context_refresh",
                refresh_started,
                attempt=state["attempts"],
                status=StepStatus.SUCCESS,
                metadata=_retrieval_metadata(merged),
            ))
            updates.update({
                "catalog": merged.catalog,
                "catalog_mode": merged.mode,
                "semantic_context": merged.context,
                "retrieval_trace": merged.trace,
                "steps": steps,
                "warnings": warnings,
                "stage": "context_refresh",
                "stream_outputs": _thought(
                    "恢复查询流程",
                    f"已扩展到 {len(merged.context.items)} 个目录上下文项，准备重新规划",
                ),
            })
            return updates
        except Exception as exc:
            steps.append(_step(
                "context_refresh",
                refresh_started,
                attempt=state["attempts"],
                status=StepStatus.FAILED,
                summary=str(exc)[:1000],
            ))
            updates.update({
                "steps": steps,
                "stage": "context_refresh",
                "error": _error(
                    stage="context_refresh",
                    code=exc.__class__.__name__,
                    kind=FailureKind.SYSTEM,
                    exc=exc,
                ),
                "outcome": QueryOutcome.SYSTEM_ERROR,
                "stream_outputs": _thought("恢复查询流程", "语义上下文扩展失败"),
            })
            return updates

    return recover


def make_finalize_node(runtime, services: HydrologySemanticQueryServices):
    async def finalize_result(state: HydrologySemanticQueryState) -> dict[str, Any]:
        started = time.perf_counter()
        steps = list(state["steps"])
        warnings = list(state["warnings"])
        outcome = state.get("outcome") or _outcome_for_error(state.get("error"))
        success = outcome == QueryOutcome.SUCCESS
        semantic_query = state.get("semantic_query") or state.get("previous_query")
        result = SemanticQueryResult(
            outcome=outcome,
            semantic_query=semantic_query,
            columns=state.get("columns", []) if success else [],
            rows=state.get("rows", []) if success else [],
            row_count=len(state.get("rows", [])) if success else 0,
            attempts=state.get("attempts", 0),
            compiled_sql=state.get("compiled_sql"),
            compiled_params=state.get("compiled_params", []),
            catalog_mode=state.get("catalog_mode"),
            query_mode=semantic_query.query_mode if semantic_query else None,
            selected_models=state.get("selected_models", []),
            retrieval_trace=state.get("retrieval_trace"),
            warnings=warnings,
            steps=steps,
            error=state.get("error"),
        )
        request: RequestData | None = None
        with suppress(Exception):
            request = _request(state, services.settings)
        if outcome == QueryOutcome.SUCCESS:
            answer = f"查询完成，共返回 {result.row_count} 行数据。"
            if request and _boolean(request["report"], services.settings.enable_report):
                try:
                    answer = await generate_report(runtime, request["question"], result)
                except Exception:
                    warnings.append(REPORT_FAILURE_WARNING)
                    result.warnings = warnings
        elif outcome == QueryOutcome.NO_DATA:
            answer = "未查询到符合当前条件的数据。"
        elif outcome == QueryOutcome.PLANNER_ERROR:
            answer = "当前问题暂时无法转换为有效的数据查询，请调整查询条件后重试。"
        elif outcome == QueryOutcome.EXECUTION_ERROR:
            answer = "当前数据查询暂时未能完成，请稍后重试。"
        else:
            answer = "当前查询暂时无法完成。"
        steps.append(_step(
            "result_finalize",
            started,
            attempt=max(1, state.get("attempts", 0)),
            status=StepStatus.SUCCESS,
        ))
        result.steps = steps
        outputs = build_result_outputs(
            result,
            answer=answer,
            question=request["question"] if request else str(state.get("query") or ""),
        )
        metadata = dict(state.get("metadata") or {})
        metadata["hydrology_semantic_query_result"] = result.model_dump(
            mode="json", by_alias=True, exclude_none=True
        )
        return {
            "answer": answer,
            "result": result,
            "steps": steps,
            "warnings": warnings,
            "metadata": metadata,
            "stream_outputs": outputs,
        }

    return finalize_result


def after_catalog(state: HydrologySemanticQueryState) -> str:
    return "finish" if state.get("outcome") is not None else "retrieve"


def after_retrieval(state: HydrologySemanticQueryState) -> str:
    return "finish" if state.get("outcome") is not None else "generate"


def after_generation(state: HydrologySemanticQueryState) -> str:
    return "validate" if state.get("semantic_query") is not None else "recover"


def after_validation(state: HydrologySemanticQueryState) -> str:
    return "compile" if state.get("error") is None else "recover"


def after_compilation(state: HydrologySemanticQueryState) -> str:
    if state.get("error") is not None or not state.get("compiled_sql"):
        return "recover"
    return "execute"


def after_execution(state: HydrologySemanticQueryState) -> str:
    return "finish" if state.get("outcome") is not None else "recover"


def after_recovery(state: HydrologySemanticQueryState) -> str:
    return "finish" if state.get("outcome") is not None else "retry"
