from __future__ import annotations

import json
import logging
import time
from typing import Any, Literal

from langchain_core.messages import BaseMessage
from typing_extensions import TypedDict

from app.agents.messages import stringify_message_content
from app.agents.state import AgentState
from app.agents.streaming import chain_of_thought_output

from .config import HydrologySemanticQuerySettings
from .models import (
    OrderItem,
    RetrievalTrace,
    SemanticCatalog,
    SemanticCatalogMode,
    SemanticColumn,
    SemanticContext,
    SemanticIntent,
    SemanticModelGap,
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
from .semantic_catalog_selector import (
    EmbeddingClient,
    SelectedSemanticCatalog,
    SemanticCatalogSelector,
    SentenceTransformerEmbedding,
)
from .semantic_cube_client import CubeClient, CubeClientError
from .semantic_query_planner import (
    build_intent_messages,
    build_messages,
    parse_semantic_intent,
    parse_semantic_query,
)
from .semantic_query_validator import validate_semantic_query

logger = logging.getLogger("uvicorn.error")
SQL_DEBUG_FAILURE_WARNING = "Cube 调试 SQL 获取失败，已保留查询结果。"


class HydrologySemanticQueryState(AgentState, total=False):
    catalog: SemanticCatalog | None
    full_catalog: SemanticCatalog | None
    catalog_mode: SemanticCatalogMode | None
    semantic_intent: SemanticIntent | None
    semantic_context: SemanticContext | None
    retrieval_trace: RetrievalTrace | None
    semantic_model_gap: SemanticModelGap | None
    selected_models: list[str]
    semantic_query: SemanticQuery | None
    previous_query: SemanticQuery | None
    cube_response: dict[str, Any] | None
    columns: list[SemanticColumn]
    rows: list[dict[str, Any]]
    steps: list[StepRecord]
    warnings: list[str]
    attempts: int
    max_attempts: int
    max_rows: int
    stage: str
    error_message: str | None
    error_code: str | None
    error_status_code: int | None
    retryable_by_model: bool
    retry_origin: Literal["empty_result", "stage_failure"] | None
    outcome: Literal["executed", "error"] | None
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
        self.embedding = embedding_client
        if self.embedding is None and settings.embedding_model:
            try:
                self.embedding = SentenceTransformerEmbedding(settings.embedding_model)
            except Exception as exc:
                self.startup_warnings.append(
                    "嵌入模型不可用，已改用分批词法目录分析。"
                    f"原因：{str(exc)[:200]}"
                )
        self.selector: SemanticCatalogSelector | None = None

    async def select_catalog(
        self,
        question: str,
        catalog: SemanticCatalog,
        *,
        intent: SemanticIntent,
        mode: SemanticCatalogMode | None = None,
        metadata_filters: dict[str, Any] | None = None,
        minimum_fallback_level: int = 0,
    ) -> SelectedSemanticCatalog:
        if self.selector is None or self.selector.catalog != catalog:
            self.selector = SemanticCatalogSelector(
                catalog,
                view_top_k=self.settings.view_top_k,
                cube_top_k=self.settings.cube_top_k,
                member_top_k=self.settings.member_top_k,
                vector_index_path=self.settings.vector_index_path,
                embedding_client=self.embedding,
                mode=self.settings.catalog_mode,
                embedding_batch_size=self.settings.embedding_batch_size,
                embedding_concurrency=self.settings.embedding_concurrency,
                retrieval_concurrency=self.settings.retrieval_concurrency,
                context_member_limit=self.settings.context_member_limit,
                catalog_batch_size=self.settings.catalog_batch_size,
                max_cube_models=self.settings.max_cube_models,
                member_match_threshold=self.settings.member_match_threshold,
            )
        selected = await self.selector.select(
            question,
            intent=intent,
            mode=mode,
            metadata_filters=metadata_filters,
            minimum_fallback_level=minimum_fallback_level,
        )
        if self.startup_warnings:
            selected.warnings[:0] = self.startup_warnings
        return selected


def _request(state: HydrologySemanticQueryState, settings: HydrologySemanticQuerySettings) -> RequestData:
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
    return {
        "question": question,
        "business_knowledge": metadata.get("business_knowledge"),
        "conversation_context": metadata.get("conversation_context"),
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


def _apply_query_defaults(
    query: SemanticQuery,
    context: SemanticContext,
) -> SemanticQuery:
    updates: dict[str, Any] = {}
    has_fields = bool(query.measures or query.dimensions or query.time_dimensions)
    if not has_fields and context.projection_policy == "model_default":
        if context.intent.result_shape == "aggregate":
            measures = [
                name
                for name in context.suggested_members
                if context.member_details.get(name, {}).get("kind") == "measure"
            ]
            if measures:
                updates.update({"measures": measures[:1], "ungrouped": False})
        else:
            dimensions = [
                name
                for name in context.suggested_members
                if context.member_details.get(name, {}).get("kind") == "dimension"
            ]
            if dimensions:
                updates.update({"dimensions": dimensions, "ungrouped": True})
    operator_resolution = context.operator_resolution
    if operator_resolution.get("operator") == "latest":
        time_member = operator_resolution.get("time_member")
        if time_member and not query.order:
            updates["order"] = [OrderItem(member=time_member, direction="desc")]
        if query.limit is None or query.limit > 1:
            updates["limit"] = 1
    return query.model_copy(update=updates) if updates else query


def _reset() -> dict[str, Any]:
    return {
        "answer": "",
        "catalog": None,
        "full_catalog": None,
        "catalog_mode": None,
        "semantic_intent": None,
        "semantic_context": None,
        "retrieval_trace": None,
        "semantic_model_gap": None,
        "selected_models": [],
        "semantic_query": None,
        "previous_query": None,
        "cube_response": None,
        "columns": [],
        "rows": [],
        "steps": [],
        "warnings": [],
        "attempts": 0,
        "max_attempts": 1,
        "max_rows": 0,
        "stage": "catalog_prepare",
        "error_message": None,
        "error_code": None,
        "error_status_code": None,
        "retryable_by_model": False,
        "retry_origin": None,
        "outcome": None,
        "result": None,
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
                "error_message": str(exc)[:1000],
                "error_code": code,
                "error_status_code": status,
                "outcome": "error",
                "stream_outputs": _thought("准备语义目录", "Cube 语义目录加载失败"),
            }

    return prepare_catalog


def make_intent_node(runtime, services: HydrologySemanticQueryServices):
    async def understand_query(state: HydrologySemanticQueryState) -> dict[str, Any]:
        started = time.perf_counter()
        steps = list(state["steps"])
        try:
            request = _request(state, services.settings)
            messages = build_intent_messages(
                question=request["question"],
                business_knowledge=request["business_knowledge"],
                conversation_context=request["conversation_context"],
            )
            model = runtime.get_chat_model(streaming=False).bind(
                response_format={"type": "json_object"},
                extra_body={"enable_thinking": False},
            )
            response = await model.ainvoke(messages, config={"callbacks": []})
            intent = parse_semantic_intent(
                stringify_message_content(response.content)
            )
            steps.append(_step(
                "query_understanding",
                started,
                attempt=1,
                status=StepStatus.SUCCESS,
                metadata={
                    "subject_count": len(intent.subjects),
                    "metric_count": len(intent.metrics),
                    "dimension_count": len(intent.dimensions),
                    "filter_count": len(intent.filters),
                    "qualifier_count": len(intent.qualifiers),
                    "requirement_count": len(intent.member_requirements),
                    "has_time": intent.temporal is not None,
                },
            ))
            return {
                "semantic_intent": intent,
                "steps": steps,
                "stage": "query_understanding",
                "error_message": None,
                "error_code": None,
                "stream_outputs": _thought(
                    "理解查询意图",
                    "已区分业务范围、字段需求、业务限定和查询操作",
                ),
            }
        except Exception as exc:
            steps.append(_step(
                "query_understanding",
                started,
                attempt=1,
                status=StepStatus.FAILED,
                summary=str(exc)[:1000],
            ))
            return {
                "steps": steps,
                "stage": "query_understanding",
                "error_message": str(exc)[:1000],
                "error_code": exc.__class__.__name__,
                "outcome": "error",
                "stream_outputs": _thought("理解查询意图", "SemanticIntent 生成或解析失败"),
            }

    return understand_query


def make_retrieval_node(services: HydrologySemanticQueryServices):
    async def retrieve_context(state: HydrologySemanticQueryState) -> dict[str, Any]:
        started = time.perf_counter()
        steps = list(state["steps"])
        warnings = list(state["warnings"])
        try:
            request = _request(state, services.settings)
            assert state["full_catalog"] is not None
            assert state["semantic_intent"] is not None
            selected = await services.select_catalog(
                request["question"],
                state["full_catalog"],
                intent=state["semantic_intent"],
                mode=request["catalog_mode"],
                metadata_filters=request["catalog_metadata_filters"],
            )
            warnings.extend(selected.warnings)
            trace = selected.trace
            metadata = {
                "catalog_mode": selected.mode.value,
                "query_mode": (
                    selected.context.query_mode.value if selected.context else None
                ),
                "view_candidates": trace.view_candidates,
                "cube_candidates": trace.cube_candidates,
                "member_parent_models": trace.member_parent_models,
                "scope_scores": trace.scope_scores,
                "member_coverage": trace.member_coverage,
                "view_coverage": trace.view_coverage,
                "resolved_requirements": trace.resolved_requirements,
                "missing_requirements": trace.missing_requirements,
                "resolved_qualifiers": trace.resolved_qualifiers,
                "operator_resolution": trace.operator_resolution,
                "fallback_anchor": trace.fallback_anchor,
                "suggested_members": trace.suggested_members,
                "rerank_scores": trace.rerank_scores,
                "selected_models": selected.selected_models,
                "allowed_member_count": (
                    len(selected.context.allowed_members) if selected.context else 0
                ),
                "join_paths": trace.join_paths,
                "fallback_level": trace.fallback_level,
                "catalog_batches_analyzed": trace.catalog_batches_analyzed,
                "index_source": selected.index_source,
            }
            steps.append(_step(
                "semantic_retrieval",
                started,
                attempt=1,
                status=StepStatus.FAILED if selected.gap else StepStatus.SUCCESS,
                summary=selected.gap.message if selected.gap else None,
                metadata=metadata,
            ))
            updates: dict[str, Any] = {
                "catalog": selected.catalog,
                "catalog_mode": selected.mode,
                "selected_models": selected.selected_models,
                "semantic_context": selected.context,
                "retrieval_trace": trace,
                "semantic_model_gap": selected.gap,
                "steps": steps,
                "warnings": warnings,
                "stage": "semantic_retrieval",
            }
            if selected.gap:
                updates.update({
                    "error_message": selected.gap.message,
                    "error_code": selected.gap.code,
                    "outcome": "error",
                    "stream_outputs": _thought(
                        "构建语义上下文", "公开语义模型无法覆盖明确的字段或能力需求"
                    ),
                })
            else:
                assert selected.context is not None
                updates.update({
                    "error_message": None,
                    "error_code": None,
                    "stream_outputs": _thought(
                        "构建语义上下文",
                        f"已选择 {selected.context.query_mode.value} 模式，"
                        f"保留 {len(selected.context.models)} 个模型和 "
                        f"{len(selected.context.allowed_members)} 个成员",
                    ),
                })
            return updates
        except Exception as exc:
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
                "error_message": str(exc)[:1000],
                "error_code": exc.__class__.__name__,
                "outcome": "error",
                "stream_outputs": _thought("构建语义上下文", "语义检索或路由失败"),
            }

    return retrieve_context


def make_generation_node(runtime, services: HydrologySemanticQueryServices):
    async def generate_semantic_query(state: HydrologySemanticQueryState) -> dict[str, Any]:
        started = time.perf_counter()
        steps = list(state["steps"])
        attempt = state["attempts"] + 1
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
                previous_error=state.get("error_message"),
            )
            model = runtime.get_chat_model(streaming=False).bind(
                response_format={"type": "json_object"},
                extra_body={"enable_thinking": False},
            )
            response = await model.ainvoke(messages, config={"callbacks": []})
            query = parse_semantic_query(stringify_message_content(response.content))
            query = _apply_query_defaults(query, state["semantic_context"])
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
                    "selected_models": state["selected_models"],
                    "query_mode": state["semantic_context"].query_mode.value,
                    "retrieval_level": state["semantic_context"].retrieval_level,
                },
            ))
            return {
                "semantic_query": query,
                "steps": steps,
                "attempts": attempt,
                "stage": "semantic_generation",
                "error_message": None,
                "error_code": None,
                "error_status_code": None,
                "retryable_by_model": False,
                "retry_origin": None,
                "stream_outputs": _thought(
                    "生成语义查询",
                    f"已完成第 {attempt} 次 SemanticQuery 生成\n\n```json\n{query_text}\n```",
                ),
            }
        except Exception as exc:
            steps.append(_step(
                "semantic_generation",
                started,
                attempt=attempt,
                status=StepStatus.FAILED,
                summary=str(exc)[:1000],
                metadata={"catalog_mode": state["catalog_mode"].value},
            ))
            return {
                "steps": steps,
                "attempts": attempt,
                "stage": "semantic_generation",
                "error_message": str(exc)[:1000],
                "error_code": exc.__class__.__name__,
                "retryable_by_model": True,
                "retry_origin": "stage_failure",
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
            ))
            return {
                "semantic_query": validated.query,
                "steps": steps,
                "warnings": warnings,
                "stage": "semantic_validation",
                "error_message": None,
                "error_code": None,
                "retryable_by_model": False,
                "retry_origin": None,
                "stream_outputs": _thought("校验语义查询", "成员白名单、类型和行数限制校验通过"),
            }
        except Exception as exc:
            steps.append(_step(
                "semantic_validation",
                started,
                attempt=state["attempts"],
                status=StepStatus.FAILED,
                summary=str(exc)[:1000],
            ))
            return {
                "steps": steps,
                "stage": "semantic_validation",
                "error_message": str(exc)[:1000],
                "error_code": exc.__class__.__name__,
                "retryable_by_model": True,
                "retry_origin": "stage_failure",
                "stream_outputs": _thought("校验语义查询", "SemanticQuery 未通过本地校验"),
            }

    return validate_query


def make_execution_node(services: HydrologySemanticQueryServices):
    async def execute_cube(state: HydrologySemanticQueryState) -> dict[str, Any]:
        started = time.perf_counter()
        steps = list(state["steps"])
        warnings = list(state["warnings"])
        try:
            assert state["semantic_query"] is not None
            wire_query = state["semantic_query"].to_cube_query()
            wire_query["timezone"] = services.settings.timezone
            response = await services.client.load(wire_query)
            columns, rows = normalize_cube_response(response)
            rows = rows[: state["semantic_query"].limit]
            empty_result = not rows
            retry_empty = (
                empty_result
                and services.settings.retry_on_empty_result
                and state["attempts"] < state["max_attempts"]
            )
            sql = None
            sql_params: list[Any] = []
            if not retry_empty:
                try:
                    sql, sql_params = await services.client.get_sql(wire_query)
                    logger.info(
                        "hydrology_semantic_query generated sql: attempt=%s sql=%s params=%s",
                        state["attempts"],
                        sql,
                        json.dumps(sql_params, ensure_ascii=False, default=str),
                    )
                except Exception as exc:
                    warnings.append(SQL_DEBUG_FAILURE_WARNING)
                    code = exc.code if isinstance(exc, CubeClientError) else exc.__class__.__name__
                    status = exc.status_code if isinstance(exc, CubeClientError) else None
                    logger.warning(
                        "hydrology_semantic_query sql generation failed: attempt=%s code=%s status=%s error=%s",
                        state["attempts"],
                        code,
                        status,
                        str(exc)[:1000],
                    )
            steps.append(_step(
                "cube_execution",
                started,
                attempt=state["attempts"],
                status=StepStatus.SUCCESS,
                metadata={
                    "row_count": len(rows),
                    "catalog_mode": state["catalog_mode"].value,
                    "query_mode": state["semantic_query"].query_mode.value,
                    "empty_result_retry": retry_empty,
                    "sql_generated": sql is not None,
                },
            ))
            if retry_empty:
                detail = "查询结果为空，正在扩大语义目录范围重试"
            elif sql:
                detail = (
                    f"Cube 查询成功，返回 {len(rows)} 行\n\n"
                    f"Cube 生成 SQL：\n\n```sql\n{sql}\n```\n\n"
                    "绑定参数：\n\n"
                    f"```json\n{json.dumps(sql_params, ensure_ascii=False, indent=2, default=str)}\n```"
                )
            else:
                detail = (
                    f"Cube 查询成功，返回 {len(rows)} 行\n\n"
                    f"> {SQL_DEBUG_FAILURE_WARNING}"
                )
            return {
                "cube_response": response,
                "columns": columns,
                "rows": rows,
                "steps": steps,
                "warnings": warnings,
                "stage": "cube_execution",
                "error_message": (
                    "Cube 查询成功但结果为空，请重新检查 View、成员和过滤条件。"
                    if retry_empty
                    else None
                ),
                "error_code": "empty_result" if retry_empty else None,
                "error_status_code": None,
                "retryable_by_model": retry_empty,
                "retry_origin": "empty_result" if retry_empty else None,
                "outcome": None if retry_empty else "executed",
                "stream_outputs": _thought("执行语义查询", detail),
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
                "error_message": str(exc)[:1000],
                "error_code": code,
                "error_status_code": status,
                "retryable_by_model": retryable,
                "retry_origin": "stage_failure",
                "stream_outputs": _thought("执行语义查询", "语义查询执行失败"),
            }

    return execute_cube


def make_recovery_node(services: HydrologySemanticQueryServices):
    async def recover(state: HydrologySemanticQueryState) -> dict[str, Any]:
        started = time.perf_counter()
        steps = list(state["steps"])
        warnings = list(state["warnings"])
        retryable = state.get("retryable_by_model", False)
        can_retry = retryable and state["attempts"] < state["max_attempts"]
        relink = state.get("stage") in {"semantic_validation", "cube_execution"}
        summary = (
            "已安排 connected component 分批扩展后重试"
            if can_retry and relink
            else "已将查询错误反馈给模型"
            if can_retry
            else "不可重试或已达最大重试次数"
        )
        steps.append(_step(
            "failure_recovery",
            started,
            attempt=max(1, state["attempts"]),
            status=StepStatus.SUCCESS if can_retry else StepStatus.SKIPPED,
            summary=summary,
            metadata={
                "catalog_mode": (
                    state["catalog_mode"].value if state.get("catalog_mode") else None
                ),
                "retry_origin": state.get("retry_origin"),
                "relink": relink,
            },
        ))
        if not can_retry:
            return {
                "steps": steps,
                "warnings": warnings,
                "outcome": "error",
                "stream_outputs": _thought(
                    "恢复查询流程", "无法继续修正，正在整理结构化失败"
                ),
            }
        previous_query = state.get("semantic_query") or state.get("previous_query")
        if state.get("retry_origin") == "empty_result":
            warnings.append(
                f"第 {state['attempts']} 次 Cube 查询结果为空，已将空结果反馈给模型继续修正。"
            )
        else:
            warnings.append(
                f"第 {state['attempts']} 次尝试在 {state.get('stage', 'unknown')} 阶段失败，已重新生成 SemanticQuery。"
            )
        updates: dict[str, Any] = {
            "steps": steps,
            "warnings": warnings,
            "previous_query": previous_query,
            "semantic_query": None,
            "retryable_by_model": False,
            "outcome": None,
        }
        if not relink:
            updates["stream_outputs"] = _thought(
                "恢复查询流程", "已记录失败反馈，准备重新生成 SemanticQuery"
            )
            return updates
        retry_started = time.perf_counter()
        try:
            request = _request(state, services.settings)
            full_catalog = state.get("full_catalog")
            assert full_catalog is not None
            intent = state.get("semantic_intent")
            assert intent is not None
            selected = await services.select_catalog(
                f"{request['question']}\n上一次尝试反馈：{state.get('error_message') or '未知错误'}",
                full_catalog,
                intent=intent,
                mode=request["catalog_mode"],
                metadata_filters=request["catalog_metadata_filters"],
                minimum_fallback_level=2,
            )
            warnings.extend(selected.warnings)
            steps.append(_step(
                "catalog_linking_retry",
                retry_started,
                attempt=state["attempts"],
                status=StepStatus.SUCCESS,
                metadata={
                    "catalog_mode": selected.mode.value,
                    "query_mode": (
                        selected.context.query_mode.value if selected.context else None
                    ),
                    "selected_model_count": len(selected.selected_models),
                    "selected_models": selected.selected_models,
                    "selected_member_count": sum(
                        len(model.members)
                        for model in selected.catalog.models.values()
                    ),
                    "index_source": selected.index_source,
                    "fallback_level": selected.trace.fallback_level,
                    "catalog_batches_analyzed": selected.trace.catalog_batches_analyzed,
                    "join_paths": selected.trace.join_paths,
                },
            ))
            if selected.gap:
                updates.update({
                    "catalog": selected.catalog,
                    "selected_models": [],
                    "semantic_context": None,
                    "retrieval_trace": selected.trace,
                    "semantic_model_gap": selected.gap,
                    "steps": steps,
                    "warnings": warnings,
                    "stage": "catalog_linking_retry",
                    "error_message": selected.gap.message,
                    "error_code": selected.gap.code,
                    "outcome": "error",
                    "stream_outputs": _thought(
                        "恢复查询流程", "分级回退后确认存在语义模型缺口"
                    ),
                })
                return updates
            updates.update({
                "catalog": selected.catalog,
                "catalog_mode": selected.mode,
                "selected_models": selected.selected_models,
                "semantic_context": selected.context,
                "retrieval_trace": selected.trace,
                "semantic_model_gap": None,
                "steps": steps,
                "warnings": warnings,
                "stream_outputs": _thought(
                    "恢复查询流程",
                    f"已完成 component 分批扩展，保留 {len(selected.selected_models)} 个 Cube",
                ),
            })
            return updates
        except Exception as exc:
            steps.append(_step(
                "catalog_linking_retry",
                retry_started,
                attempt=state["attempts"],
                status=StepStatus.FAILED,
                summary=str(exc)[:1000],
            ))
            if state.get("retry_origin") != "empty_result":
                updates.update({
                    "steps": steps,
                    "stage": "catalog_linking_retry",
                    "error_message": str(exc)[:1000],
                    "error_code": exc.__class__.__name__,
                    "outcome": "error",
                    "stream_outputs": _thought(
                        "恢复查询流程", "语义目录重新关联失败，正在整理失败信息"
                    ),
                })
                return updates
            warnings.append(
                f"第 {state['attempts']} 次尝试在 catalog_linking_retry 阶段失败，已使用当前目录继续重试。"
            )
            updates.update({
                "steps": steps,
                "warnings": warnings,
                "error_message": str(exc)[:1000],
                "error_code": exc.__class__.__name__,
                "stream_outputs": _thought(
                    "恢复查询流程", "语义目录扩大失败，将使用当前目录继续重试"
                ),
            })
            return updates

    return recover


def make_finalize_node(runtime, services: HydrologySemanticQueryServices):
    async def finalize_result(state: HydrologySemanticQueryState) -> dict[str, Any]:
        started = time.perf_counter()
        steps = list(state["steps"])
        warnings = list(state["warnings"])
        executed = state.get("outcome") == "executed"
        error = None
        if not executed:
            gap = state.get("semantic_model_gap")
            error = SemanticQueryError(
                stage=state.get("stage", "unknown"),
                code=state.get("error_code") or "semantic_query_failed",
                message=state.get("error_message") or "语义查询失败",
                retryable=False,
                status_code=state.get("error_status_code"),
                details=(
                    gap.model_dump(mode="json") if gap is not None else {}
                ),
            )
        result = SemanticQueryResult(
            success=executed,
            executed=executed,
            semantic_query=state.get("semantic_query") or state.get("previous_query"),
            columns=state.get("columns", []),
            rows=state.get("rows", []),
            row_count=len(state.get("rows", [])),
            attempts=state.get("attempts", 0),
            catalog_mode=state.get("catalog_mode"),
            query_mode=(
                state["semantic_context"].query_mode
                if state.get("semantic_context")
                else None
            ),
            selected_models=state.get("selected_models", []),
            retrieval_trace=state.get("retrieval_trace"),
            semantic_model_gap=state.get("semantic_model_gap"),
            warnings=warnings,
            steps=steps,
            error=error,
        )
        request: RequestData | None = None
        try:
            request = _request(state, services.settings)
        except Exception:
            pass
        if executed:
            answer = f"查询完成，共返回 {result.row_count} 行数据。"
            if request and _boolean(request["report"], services.settings.enable_report):
                try:
                    answer = await generate_report(runtime, request["question"], result)
                except Exception:
                    warnings.append(REPORT_FAILURE_WARNING)
                    result.warnings = warnings
        elif result.semantic_model_gap:
            missing = "、".join(result.semantic_model_gap.missing_concepts)
            answer = f"查询失败：语义模型缺少所需概念{f'：{missing}' if missing else '。'}"
        else:
            answer = f"查询失败：{error.message}" if error else "查询失败。"
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
    return "finish" if state.get("outcome") == "error" else "understand"


def after_intent(state: HydrologySemanticQueryState) -> str:
    return "finish" if state.get("outcome") == "error" else "retrieve"


def after_retrieval(state: HydrologySemanticQueryState) -> str:
    return "finish" if state.get("outcome") == "error" else "generate"


def after_generation(state: HydrologySemanticQueryState) -> str:
    return "validate" if state.get("semantic_query") is not None else "recover"


def after_validation(state: HydrologySemanticQueryState) -> str:
    return "execute" if not state.get("error_message") else "recover"


def after_execution(state: HydrologySemanticQueryState) -> str:
    return "finish" if state.get("outcome") == "executed" else "recover"


def after_recovery(state: HydrologySemanticQueryState) -> str:
    return "finish" if state.get("outcome") == "error" else "retry"
