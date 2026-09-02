from __future__ import annotations

import json
import logging
import math
import time
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
from typing import Any

from langchain.tools import ToolRuntime
from langgraph.types import Command

from ...contracts import (
    FailureKind,
    FilterOperator,
    QueryExecutionRecord,
    QueryMode,
    QueryOutcome,
    SemanticColumn,
    SemanticFilter,
    SemanticQuery,
    StepStatus,
)
from ..client import CubeClientError
from ..models import (
    CatalogMember,
    SemanticCatalog,
)
from ..runtime import (
    HydrologySemanticQueryServices,
    build_error,
    build_step,
    outcome_for_error,
    thought_output,
)
from ..state import QueryAgentState
from .common import error_payload, tool_message


class SemanticQueryValidationError(ValueError):
    def __init__(
        self,
        message: str,
        *,
        code: str = "semantic_query_validation_error",
        details: dict[str, object] | None = None,
    ) -> None:
        super().__init__(message)
        self.code = code
        self.details = details or {}


@dataclass(slots=True)
class ValidatedSemanticQuery:
    query: SemanticQuery
    warnings: list[str]


def _semantic_query_detail(query: SemanticQuery, summary: str) -> str:
    payload = query.model_dump(mode="json", by_alias=True, exclude_none=True)
    return f"{summary}\n\n```json\n{json.dumps(payload, ensure_ascii=False, indent=2)}\n```"


STANDARD_GRANULARITIES = frozenset(
    {"second", "minute", "hour", "day", "week", "month", "quarter", "year"}
)
EQUALITY_OPERATORS = frozenset({FilterOperator.EQUALS, FilterOperator.NOT_EQUALS})
STRING_OPERATORS = frozenset(
    {
        FilterOperator.CONTAINS,
        FilterOperator.NOT_CONTAINS,
        FilterOperator.STARTS_WITH,
        FilterOperator.NOT_STARTS_WITH,
        FilterOperator.ENDS_WITH,
        FilterOperator.NOT_ENDS_WITH,
    }
)
NUMBER_OPERATORS = frozenset(
    {FilterOperator.GT, FilterOperator.GTE, FilterOperator.LT, FilterOperator.LTE}
)
NULL_OPERATORS = frozenset({FilterOperator.SET, FilterOperator.NOT_SET})
DATE_RANGE_OPERATORS = frozenset(
    {FilterOperator.IN_DATE_RANGE, FilterOperator.NOT_IN_DATE_RANGE}
)
DATE_BOUND_OPERATORS = frozenset(
    {
        FilterOperator.BEFORE_DATE,
        FilterOperator.BEFORE_OR_ON_DATE,
        FilterOperator.AFTER_DATE,
        FilterOperator.AFTER_OR_ON_DATE,
    }
)


def _require_unique(values: list[str], label: str) -> None:
    if len(values) != len(set(values)):
        raise SemanticQueryValidationError(
            f"{label} 不能包含重复成员",
            code="query_shape",
        )


def _member(catalog: SemanticCatalog, query: SemanticQuery, name: str) -> CatalogMember:
    prefix, separator, _ = name.partition(".")
    if not separator:
        raise SemanticQueryValidationError(
            f"成员必须使用 model.member 全名：{name}",
            code="member_scope_mismatch",
        )
    if prefix not in query.models:
        raise SemanticQueryValidationError(
            f"成员前缀不属于 query models：{name}",
            code="member_scope_mismatch",
        )
    model = catalog.models.get(prefix)
    if model is None:
        raise SemanticQueryValidationError(
            f"未公开或不存在的水文 model：{prefix}",
            code="unknown_model",
        )
    member = model.members.get(name)
    if member is None:
        raise SemanticQueryValidationError(
            f"Cube 语义目录中不存在成员：{name}",
            code="unknown_member",
        )
    return member


def _require_type(
    catalog: SemanticCatalog,
    query: SemanticQuery,
    name: str,
    expected: str,
) -> CatalogMember:
    member = _member(catalog, query, name)
    if member.member_type != expected:
        raise SemanticQueryValidationError(
            f"成员 {name} 的类型是 {member.member_type}，不是 {expected}",
            code="member_type_mismatch",
        )
    return member


def _filter_types(
    catalog: SemanticCatalog,
    query: SemanticQuery,
    semantic_filter: SemanticFilter,
) -> set[str]:
    if semantic_filter.and_ or semantic_filter.or_:
        children = semantic_filter.and_ or semantic_filter.or_
        types: set[str] = set()
        for child in children:
            types.update(_filter_types(catalog, query, child))
        if len(types) > 1:
            raise SemanticQueryValidationError(
                "同一逻辑过滤组不能混合 measure 和 dimension",
                code="invalid_filter",
            )
        return types
    assert semantic_filter.member is not None
    member = _member(catalog, query, semantic_filter.member)
    if member.member_type not in {"measure", "dimension"}:
        raise SemanticQueryValidationError(
            f"过滤条件不能引用 {member.member_type} 成员：{member.name}",
            code="invalid_filter",
        )
    if member.data_type == "boolean":
        normalized_values: list[object] = []
        for value in semantic_filter.values:
            literal = value.strip().lower() if isinstance(value, str) else value
            if literal is True or literal == "true" or (
                not isinstance(literal, bool)
                and isinstance(literal, int | float)
                and literal == 1
            ):
                value = "1"
            elif literal is False or literal == "false" or (
                not isinstance(literal, bool)
                and isinstance(literal, int | float)
                and literal == 0
            ):
                value = "0"
            elif literal in {"0", "1"}:
                value = literal
            else:
                raise SemanticQueryValidationError(
                    f"boolean 成员只能使用 0 或 1 过滤：{member.name}",
                    code="invalid_filter",
                )
            normalized_values.append(value)
        semantic_filter.values = normalized_values
    assert semantic_filter.operator is not None
    _validate_filter_operator(member, semantic_filter.operator, semantic_filter.values)
    return {member.member_type}


def _filter_member_names(semantic_filter: SemanticFilter) -> list[str]:
    if semantic_filter.and_ or semantic_filter.or_:
        children = semantic_filter.and_ or semantic_filter.or_
        return [
            name
            for child in children
            for name in _filter_member_names(child)
        ]
    return [semantic_filter.member] if semantic_filter.member else []


def _referenced_member_names(query: SemanticQuery) -> list[str]:
    return [
        *query.measures,
        *query.dimensions,
        *query.segments,
        *[
            name
            for semantic_filter in query.filters
            for name in _filter_member_names(semantic_filter)
        ],
        *[item.dimension for item in query.time_dimensions],
        *[item.member for item in query.order],
    ]


def _validate_member_references(
    catalog: SemanticCatalog,
    query: SemanticQuery,
) -> None:
    invalid_members: list[dict[str, str]] = []
    seen: set[str] = set()
    for name in _referenced_member_names(query):
        if name in seen:
            continue
        seen.add(name)
        try:
            _member(catalog, query, name)
        except SemanticQueryValidationError as exc:
            invalid_members.append(
                {
                    "code": exc.code,
                    "member": name,
                    "message": str(exc),
                }
            )
    if invalid_members:
        raise SemanticQueryValidationError(
            "; ".join(item["message"] for item in invalid_members),
            code=invalid_members[0]["code"],
            details={"invalid_members": invalid_members},
        )


def _validate_filter_operator(
    member: CatalogMember,
    operator: FilterOperator,
    values: list[object],
) -> None:
    if operator in STRING_OPERATORS and member.data_type != "string":
        raise SemanticQueryValidationError(
            f"操作符 {operator.value} 只能用于 string 成员：{member.name}",
            code="invalid_filter",
        )
    if operator in NUMBER_OPERATORS and member.data_type != "number":
        raise SemanticQueryValidationError(
            f"操作符 {operator.value} 只能用于 number 成员：{member.name}",
            code="invalid_filter",
        )
    if (
        operator in DATE_RANGE_OPERATORS | DATE_BOUND_OPERATORS
        and (member.member_type != "dimension" or member.data_type != "time")
    ):
        raise SemanticQueryValidationError(
            f"操作符 {operator.value} 只能用于 time dimension：{member.name}",
            code="invalid_filter",
        )
    if operator in EQUALITY_OPERATORS | STRING_OPERATORS:
        if not values:
            raise SemanticQueryValidationError(
                f"操作符 {operator.value} 至少需要一个值",
                code="invalid_filter",
            )
        if member.data_type == "number":
            for value in values:
                try:
                    number = Decimal(str(value))
                except InvalidOperation as exc:
                    raise SemanticQueryValidationError(
                        f"操作符 {operator.value} 的值必须是数值",
                        code="invalid_filter",
                    ) from exc
                if not number.is_finite():
                    raise SemanticQueryValidationError(
                        f"操作符 {operator.value} 的值必须是有限数值",
                        code="invalid_filter",
                    )
        return
    if operator in NUMBER_OPERATORS:
        if len(values) != 1:
            raise SemanticQueryValidationError(
                f"操作符 {operator.value} 必须且只能提供一个数值",
                code="invalid_filter",
            )
        try:
            number = Decimal(str(values[0]))
        except InvalidOperation as exc:
            raise SemanticQueryValidationError(
                f"操作符 {operator.value} 的值必须是数值",
                code="invalid_filter",
            ) from exc
        if not number.is_finite():
            raise SemanticQueryValidationError(
                f"操作符 {operator.value} 的值必须是有限数值",
                code="invalid_filter",
            )
        return
    if operator in NULL_OPERATORS:
        if values:
            raise SemanticQueryValidationError(
                f"操作符 {operator.value} 不允许提供值",
                code="invalid_filter",
            )
        return
    if operator in DATE_RANGE_OPERATORS:
        if len(values) not in {1, 2} or not all(
            isinstance(value, str) and value.strip() for value in values
        ):
            raise SemanticQueryValidationError(
                f"操作符 {operator.value} 需要一个日期表达式或两个日期边界",
                code="invalid_filter",
            )
        return
    if operator in DATE_BOUND_OPERATORS and (
        len(values) != 1 or not isinstance(values[0], str) or not values[0].strip()
    ):
        raise SemanticQueryValidationError(
            f"操作符 {operator.value} 必须且只能提供一个日期",
            code="invalid_filter",
        )


def validate_query_shape(query: SemanticQuery) -> None:
    if not query.measures and not query.dimensions and not query.time_dimensions:
        raise SemanticQueryValidationError(
            "查询至少需要一个 measure、dimension 或 time dimension",
            code="query_shape",
        )
    if query.ungrouped:
        if query.measures:
            raise SemanticQueryValidationError(
                "明细查询不得携带聚合 measure",
                code="query_shape",
            )
        if not query.dimensions and not query.time_dimensions:
            raise SemanticQueryValidationError(
                "明细查询必须至少包含一个 dimension 或 time dimension",
                code="query_shape",
            )
        return
    if not query.measures and (query.dimensions or query.time_dimensions):
        raise SemanticQueryValidationError(
            "无 measure 的明细查询必须设置 ungrouped=true",
            code="query_shape",
        )


def validate_semantic_query(
    query: SemanticQuery,
    catalog: SemanticCatalog,
    *,
    requested_max_rows: int,
    hard_max_rows: int,
    timezone: str | None = None,
) -> ValidatedSemanticQuery:
    del timezone
    if requested_max_rows < 1 or hard_max_rows < 1:
        raise SemanticQueryValidationError(
            "结果行数上限必须大于 0",
            code="invalid_limit",
        )
    _require_unique(query.models, "query models")
    _require_unique(query.measures, "measures")
    _require_unique(query.dimensions, "dimensions")
    _require_unique(query.segments, "segments")
    _require_unique(
        [item.dimension for item in query.time_dimensions],
        "time_dimensions",
    )
    _require_unique([item.member for item in query.order], "order")
    missing_models = [name for name in query.models if name not in catalog.models]
    if missing_models:
        raise SemanticQueryValidationError(
            f"包含未公开或不存在的水文 model：{missing_models}",
            code="unknown_model",
        )
    model_types = {catalog.models[name].model_type for name in query.models}
    if len(model_types) != 1:
        raise SemanticQueryValidationError(
            "View 和 Cube namespace 禁止混合",
            code="query_shape",
        )
    model_type = next(iter(model_types))
    if query.query_mode == QueryMode.VIEW:
        if model_type != "view" or len(query.models) != 1:
            raise SemanticQueryValidationError(
                "View Mode 必须且只能查询一个公开 View",
                code="query_shape",
            )
    elif model_type != "cube":
        raise SemanticQueryValidationError(
            "Cube Mode 只能查询公开 Cube，禁止与 View 混合",
            code="query_shape",
        )
    elif len(query.models) > 1:
        components = {
            catalog.models[name].connected_component for name in query.models
        }
        if None in components or len(components) != 1:
            raise SemanticQueryValidationError(
                "Cube models 不属于同一 connectedComponent，无法证明连通",
                code="join_unreachable",
            )
    validate_query_shape(query)
    _validate_member_references(catalog, query)
    for name in query.measures:
        _require_type(catalog, query, name, "measure")
    for name in query.dimensions:
        _require_type(catalog, query, name, "dimension")
    for name in query.segments:
        _require_type(catalog, query, name, "segment")
    for semantic_filter in query.filters:
        _filter_types(catalog, query, semantic_filter)
    for time_dimension in query.time_dimensions:
        member = _require_type(
            catalog,
            query,
            time_dimension.dimension,
            "dimension",
        )
        if member.data_type != "time":
            raise SemanticQueryValidationError(
                f"时间范围只能引用 time dimension：{member.name}",
                code="member_type_mismatch",
            )
        if time_dimension.granularity:
            allowed = set(STANDARD_GRANULARITIES) | set(member.granularities)
            if time_dimension.granularity not in allowed:
                raise SemanticQueryValidationError(
                    f"不支持的时间粒度：{time_dimension.granularity}",
                    code="invalid_time_dimension",
                )
    for order in query.order:
        member = _member(catalog, query, order.member)
        if member.member_type == "segment":
            raise SemanticQueryValidationError(
                f"不能按 segment 排序：{order.member}",
                code="member_type_mismatch",
            )
    warnings: list[str] = []
    effective_max = min(requested_max_rows, hard_max_rows)
    if requested_max_rows > hard_max_rows:
        warnings.append(
            f"请求行数上限 {requested_max_rows} 超过硬上限 {hard_max_rows}，已按硬上限执行。"
        )
    if query.limit is None:
        query.limit = effective_max
    elif query.limit > effective_max:
        warnings.append(
            f"语义查询 limit {query.limit} 超过本次上限 {effective_max}，已自动收紧。"
        )
        query.limit = effective_max
    return ValidatedSemanticQuery(query=query, warnings=warnings)

_ANNOTATION_MEMBER_TYPES = {
    "dimensions": "dimension",
    "timeDimensions": "time_dimension",
    "measures": "measure",
}


def _annotation_map(payload: dict[str, Any]) -> dict[str, dict[str, Any]]:
    annotation = payload.get("annotation") or {}
    result: dict[str, dict[str, Any]] = {}
    if not isinstance(annotation, dict):
        return result
    for group, member_type in _ANNOTATION_MEMBER_TYPES.items():
        values = annotation.get(group) or {}
        if isinstance(values, dict):
            for name, item in values.items():
                metadata = dict(item) if isinstance(item, dict) else {}
                metadata["member_type"] = member_type
                result[str(name)] = metadata
    return result


def _number(value: Any) -> Any:
    if value is None or (
        isinstance(value, int | float) and not isinstance(value, bool)
    ):
        if isinstance(value, float) and not math.isfinite(value):
            return None
        return value
    try:
        number = Decimal(str(value))
    except (InvalidOperation, ValueError):
        return value
    if not number.is_finite():
        return None
    if number == number.to_integral_value():
        return int(number)
    converted = float(number)
    return converted if math.isfinite(converted) else None


def _typed(value: Any, data_type: str) -> Any:
    if value is None:
        return None
    if data_type == "number":
        return _number(value)
    if data_type == "boolean":
        if isinstance(value, bool):
            return value
        if isinstance(value, int | float) and value in {0, 1}:
            return bool(value)
        if isinstance(value, str):
            normalized = value.strip().lower()
            if normalized in {"1", "true", "yes"}:
                return True
            if normalized in {"0", "false", "no"}:
                return False
    return value


def _is_hidden_output_column(column: SemanticColumn) -> bool:
    short_name = column.name.rsplit(".", 1)[-1].lower()
    title = column.title.strip().lower()
    return (
        short_name in {"id", "relation_key", "code"}
        or short_name.endswith(("_id", "_ids", "_code"))
        or title.endswith(("id", "id列表", "编码"))
    )


def normalize_cube_response(
    payload: dict[str, Any],
) -> tuple[list[SemanticColumn], list[dict[str, Any]]]:
    if isinstance(payload.get("results"), list) and payload["results"]:
        first = payload["results"][0]
        if not isinstance(first, dict):
            raise ValueError("Cube /load 响应的 results 只能包含对象")
        payload = first
    raw_rows = payload.get("data") or []
    if not isinstance(raw_rows, list):
        raise ValueError("Cube /load 响应的 data 必须是列表")
    if any(not isinstance(row, dict) for row in raw_rows):
        raise ValueError("Cube /load 响应的 data 只能包含对象")
    rows = [dict(row) for row in raw_rows]
    annotations = _annotation_map(payload)
    names = list(annotations)
    for row in rows:
        for name in row:
            if name not in names:
                names.append(name)
    columns = [
        SemanticColumn(
            name=name,
            title=str(annotations.get(name, {}).get("title") or name),
            data_type=str(annotations.get(name, {}).get("type") or "string"),
            member_type=str(annotations.get(name, {}).get("member_type") or "unknown"),
        )
        for name in names
    ]
    columns = [column for column in columns if not _is_hidden_output_column(column)]
    visible_names = {column.name for column in columns}
    types = {column.name: column.data_type for column in columns}
    normalized_rows = [
        {
            name: _typed(value, types.get(name, "string"))
            for name, value in row.items()
            if name in visible_names
        }
        for row in rows
    ]
    return columns, normalized_rows

logger = logging.getLogger("uvicorn.error")


async def validate_query(
    state: QueryAgentState,
    services: HydrologySemanticQueryServices,
) -> dict[str, Any]:
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
        steps.append(build_step(
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
            "stream_outputs": thought_output(
                "生成语义查询",
                _semantic_query_detail(
                    validated.query,
                    "SemanticQuery 已生成，并通过可访问目录、成员类型、查询形态和行数限制校验",
                ),
            ),
        }
    except Exception as exc:
        validation_error = isinstance(exc, SemanticQueryValidationError)
        code = exc.code if validation_error else exc.__class__.__name__
        steps.append(build_step(
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
            "error": build_error(
                stage="semantic_validation",
                code=code,
                kind=FailureKind.VALIDATION if validation_error else FailureKind.SYSTEM,
                exc=exc,
                retryable=validation_error,
                details=exc.details if validation_error else None,
            ),
            "stream_outputs": thought_output(
                "生成语义查询",
                _semantic_query_detail(
                    state["semantic_query"],
                    "SemanticQuery 已生成，但未通过本地校验",
                ),
            ),
        }


async def compile_query(
    state: QueryAgentState,
    services: HydrologySemanticQueryServices,
) -> dict[str, Any]:
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
        steps.append(build_step(
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
            "stream_outputs": thought_output(
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
        steps.append(build_step(
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
            "error": build_error(
                stage="semantic_compilation",
                code=code,
                kind=kind,
                exc=exc,
                retryable=retryable,
                status_code=status,
            ),
            "stream_outputs": thought_output(
                "编译语义查询", "SemanticQuery 无法通过 Cube /sql 预检"
            ),
        }


async def execute_query(
    state: QueryAgentState,
    services: HydrologySemanticQueryServices,
) -> dict[str, Any]:
    started = time.perf_counter()
    steps = list(state["steps"])
    try:
        assert state["semantic_query"] is not None
        assert state["cube_query"] is not None
        response = await services.client.load(state["cube_query"])
        columns, rows = normalize_cube_response(response)
        rows = rows[: state["semantic_query"].limit]
        empty_result = not rows
        steps.append(build_step(
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
            "stream_outputs": thought_output(
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
        steps.append(build_step(
            "cube_execution",
            started,
            attempt=state["attempts"],
            status=StepStatus.FAILED,
            summary=str(exc)[:1000],
        ))
        return {
            "steps": steps,
            "stage": "cube_execution",
            "error": build_error(
                stage="cube_execution",
                code=code,
                kind=FailureKind.EXECUTION,
                exc=exc,
                retryable=retryable,
                status_code=status,
            ),
            "stream_outputs": thought_output("执行语义查询", "语义查询执行失败"),
        }

async def run_semantic_query_service(
    *,
    semantic_query: SemanticQuery,
    query_goal: str | None,
    runtime: ToolRuntime,
    services: HydrologySemanticQueryServices,
) -> Command:
    state: QueryAgentState = runtime.state
    query_history = list(state.get("query_history", []))
    if len(query_history) >= 1:
        warnings = list(state.get("warnings", []))
        warning = "当前查询任务已经形成业务查询结果，未执行额外查询。"
        if warning not in warnings:
            warnings.append(warning)
        observation = {
            "ok": True,
            "kind": "query_round_limit",
            "query_count": len(query_history),
            "max_query_rounds": 1,
            "terminal": True,
        }
        return Command(update={
            "messages": [tool_message(runtime, "run_semantic_query", observation)],
            "warnings": warnings,
            "last_tool_terminal": True,
            "stream_outputs": thought_output(
                "规划后续查询",
                "当前查询任务已经完成",
            ),
        })
    attempt = state.get("attempts", 0) + 1
    normalized_goal = (query_goal or "").strip() or str(
        state.get("standalone_question") or state.get("query") or f"第 {len(query_history) + 1} 轮查询"
    )
    working: QueryAgentState = dict(state)
    working.update({
        "semantic_query": semantic_query.model_copy(deep=True),
        "previous_query": state.get("semantic_query"),
        "selected_models": [],
        "cube_query": None,
        "compiled_sql": None,
        "compiled_params": [],
        "cube_response": None,
        "columns": [],
        "rows": [],
        "attempts": attempt,
        "stage": "semantic_validation",
        "error": None,
        "outcome": None,
    })
    if state.get("catalog") is None or state.get("search_count", 0) < 1:
        started = time.perf_counter()
        error = build_error(
            stage="semantic_validation",
            code="catalog_search_required",
            kind=FailureKind.VALIDATION,
            exc="首次执行 SemanticQuery 前必须先调用 search_semantic_catalog",
            retryable=True,
        )
        steps = list(state.get("steps", []))
        steps.append(build_step(
            "semantic_validation",
            started,
            attempt=attempt,
            status=StepStatus.FAILED,
            summary=error.internal_message,
            metadata={"code": error.code},
        ))
        observation = error_payload(error, terminal=False)
        return Command(update={
            "messages": [
                tool_message(runtime, "run_semantic_query", observation, error=True)
            ],
            "semantic_query": working["semantic_query"],
            "previous_query": working["previous_query"],
            "steps": steps,
            "attempts": attempt,
            "stage": error.stage,
            "error": error,
            "outcome": None,
            "last_tool_terminal": False,
            "stream_outputs": thought_output(
                "生成语义查询",
                _semantic_query_detail(
                    working["semantic_query"],
                    "SemanticQuery 已生成，但需要先检索受治理语义目录",
                ),
            ),
        })
    pipeline_outputs: list[dict[str, Any]] = []
    for pipeline_step in (validate_query, compile_query, execute_query):
        updates = await pipeline_step(working, services)
        pipeline_outputs.extend(updates.get("stream_outputs", []))
        working.update(updates)
        if working.get("error") is not None:
            break
    error = working.get("error")
    if error is not None:
        terminal = not error.retryable
        if terminal:
            working["outcome"] = outcome_for_error(error)
        observation = error_payload(error, terminal=terminal)
        return Command(update={
            "messages": [
                tool_message(runtime, "run_semantic_query", observation, error=True)
            ],
            "semantic_query": working.get("semantic_query"),
            "previous_query": working.get("previous_query"),
            "selected_models": working.get("selected_models", []),
            "cube_query": working.get("cube_query"),
            "compiled_sql": working.get("compiled_sql"),
            "compiled_params": working.get("compiled_params", []),
            "columns": [],
            "rows": [],
            "steps": working.get("steps", []),
            "warnings": working.get("warnings", []),
            "attempts": attempt,
            "stage": working.get("stage", error.stage),
            "error": error,
            "outcome": working.get("outcome"),
            "last_tool_terminal": terminal,
            "stream_outputs": pipeline_outputs + thought_output(
                "执行语义查询",
                "查询失败，正在根据结构化 Observation 决定下一步"
                if not terminal
                else "查询遇到不可重试错误",
            ),
        })
    outcome = working.get("outcome") or QueryOutcome.SYSTEM_ERROR
    rows = working.get("rows", [])
    columns = working.get("columns", [])
    query_number = len(query_history) + 1
    query_history.append(QueryExecutionRecord(
        query_number=query_number,
        task_id=(
            state["current_task"].task_id
            if state.get("current_task") is not None
            else None
        ),
        query_goal=normalized_goal,
        semantic_query=working["semantic_query"].model_copy(deep=True),
        outcome=outcome,
        columns=list(columns),
        rows=list(rows),
        row_count=len(rows),
        attempt=attempt,
        compiled_sql=working.get("compiled_sql"),
        compiled_params=working.get("compiled_params", []),
        selected_models=working.get("selected_models", []),
    ))
    terminal = True
    observation = {
        "ok": True,
        "kind": "semantic_query_result",
        "outcome": outcome.value,
        "query_number": query_number,
        "query_goal": normalized_goal,
        "semantic_query": working["semantic_query"].model_dump(
            mode="json", by_alias=True, exclude_none=True
        ),
        "compiled": bool(working.get("compiled_sql")),
        "row_count": len(rows),
        "columns": [column.model_dump(mode="json") for column in columns],
        "rows": rows[:50],
        "rows_truncated": len(rows) > 50,
        "terminal": terminal,
        "remaining_query_rounds": 0,
        "warnings": working.get("warnings", []),
    }
    return Command(update={
        "messages": [tool_message(runtime, "run_semantic_query", observation)],
        "semantic_query": working.get("semantic_query"),
        "previous_query": working.get("previous_query"),
        "selected_models": working.get("selected_models", []),
        "cube_query": working.get("cube_query"),
        "compiled_sql": working.get("compiled_sql"),
        "compiled_params": working.get("compiled_params", []),
        "cube_response": working.get("cube_response"),
        "columns": columns,
        "rows": rows,
        "steps": working.get("steps", []),
        "warnings": working.get("warnings", []),
        "attempts": attempt,
        "query_count": query_number,
        "query_history": query_history,
        "stage": working.get("stage", "cube_execution"),
        "error": None,
        "outcome": outcome,
        "last_tool_terminal": terminal,
        "stream_outputs": pipeline_outputs,
    })
