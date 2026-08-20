from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal, InvalidOperation

from .models import (
    CatalogMember,
    FilterOperator,
    ProjectionMode,
    QueryMode,
    SemanticCatalog,
    SemanticFilter,
    SemanticQuery,
)
from .semantic_context import SemanticJoinGraph


class SemanticQueryValidationError(ValueError):
    def __init__(
        self,
        message: str,
        *,
        code: str = "semantic_query_validation_error",
    ) -> None:
        super().__init__(message)
        self.code = code


@dataclass(slots=True)
class ValidatedSemanticQuery:
    query: SemanticQuery
    warnings: list[str]


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


def _member(catalog: SemanticCatalog, query: SemanticQuery, name: str) -> CatalogMember:
    prefix, separator, _ = name.partition(".")
    if not separator:
        raise SemanticQueryValidationError(f"成员必须使用 model.member 全名：{name}")
    if prefix not in query.models:
        raise SemanticQueryValidationError(f"成员前缀不属于 query models：{name}")
    model = catalog.models.get(prefix)
    if model is None:
        raise SemanticQueryValidationError(f"未公开、不存在或未召回的水文 model：{prefix}")
    member = model.members.get(name)
    if member is None:
        raise SemanticQueryValidationError(f"Cube 语义目录中不存在成员：{name}")
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
            f"成员 {name} 的类型是 {member.member_type}，不是 {expected}"
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
                "同一逻辑过滤组不能混合 measure 和 dimension"
            )
        return types
    assert semantic_filter.member is not None
    member = _member(catalog, query, semantic_filter.member)
    if member.member_type not in {"measure", "dimension"}:
        raise SemanticQueryValidationError(
            f"过滤条件不能引用 {member.member_type} 成员：{member.name}"
        )
    if member.data_type == "boolean":
        normalized_values: list[object] = []
        for value in semantic_filter.values:
            literal = value.strip().lower() if isinstance(value, str) else value
            if literal is True or literal == "true":
                value = "1"
            elif literal is False or literal == "false":
                value = "0"
            normalized_values.append(value)
        semantic_filter.values = normalized_values
    assert semantic_filter.operator is not None
    _validate_filter_operator(member, semantic_filter.operator, semantic_filter.values)
    return {member.member_type}


def _validate_filter_operator(
    member: CatalogMember,
    operator: FilterOperator,
    values: list[object],
) -> None:
    if operator in STRING_OPERATORS and member.data_type != "string":
        raise SemanticQueryValidationError(
            f"操作符 {operator.value} 只能用于 string 成员：{member.name}"
        )
    if operator in NUMBER_OPERATORS and member.data_type != "number":
        raise SemanticQueryValidationError(
            f"操作符 {operator.value} 只能用于 number 成员：{member.name}"
        )
    if operator in DATE_RANGE_OPERATORS | DATE_BOUND_OPERATORS:
        if member.member_type != "dimension" or member.data_type != "time":
            raise SemanticQueryValidationError(
                f"操作符 {operator.value} 只能用于 time dimension：{member.name}"
            )
    if operator in EQUALITY_OPERATORS | STRING_OPERATORS:
        if not values:
            raise SemanticQueryValidationError(f"操作符 {operator.value} 至少需要一个值")
        return
    if operator in NUMBER_OPERATORS:
        if len(values) != 1:
            raise SemanticQueryValidationError(f"操作符 {operator.value} 必须且只能提供一个数值")
        try:
            number = Decimal(str(values[0]))
        except InvalidOperation as exc:
            raise SemanticQueryValidationError(
                f"操作符 {operator.value} 的值必须是数值"
            ) from exc
        if not number.is_finite():
            raise SemanticQueryValidationError(f"操作符 {operator.value} 的值必须是有限数值")
        return
    if operator in NULL_OPERATORS:
        if values:
            raise SemanticQueryValidationError(f"操作符 {operator.value} 不允许提供值")
        return
    if operator in DATE_RANGE_OPERATORS:
        if len(values) not in {1, 2} or not all(
            isinstance(value, str) and value.strip() for value in values
        ):
            raise SemanticQueryValidationError(
                f"操作符 {operator.value} 需要一个日期表达式或两个日期边界"
            )
        return
    if operator in DATE_BOUND_OPERATORS and (
        len(values) != 1 or not isinstance(values[0], str) or not values[0].strip()
    ):
        raise SemanticQueryValidationError(f"操作符 {operator.value} 必须且只能提供一个日期")


def validate_projection_consistency(
    query: SemanticQuery,
    projection_mode: ProjectionMode,
) -> None:
    if projection_mode == ProjectionMode.DETAIL:
        if query.measures:
            reason = "detail 查询不得携带聚合 measure"
        elif not query.dimensions:
            reason = "detail 查询必须至少包含一个 dimension"
        elif not query.ungrouped:
            reason = "detail 查询必须设置 ungrouped=true"
        else:
            return
    elif projection_mode == ProjectionMode.AGGREGATE:
        if not query.measures:
            reason = "aggregate 查询必须至少包含一个 measure"
        elif query.ungrouped:
            reason = "aggregate 查询必须设置 ungrouped=false"
        else:
            return
    else:
        return
    raise SemanticQueryValidationError(reason, code="projection_mode_mismatch")


def validate_semantic_query(
    query: SemanticQuery,
    catalog: SemanticCatalog,
    *,
    requested_max_rows: int,
    hard_max_rows: int,
    timezone: str | None = None,
    projection_mode: ProjectionMode = ProjectionMode.DEFAULT,
) -> ValidatedSemanticQuery:
    del timezone
    if len(query.models) != len(set(query.models)):
        raise SemanticQueryValidationError("query models 不能重复")
    missing_models = [name for name in query.models if name not in catalog.models]
    if missing_models:
        raise SemanticQueryValidationError(
            f"包含未公开、不存在或未召回的水文 model：{missing_models}"
        )
    model_types = {catalog.models[name].model_type for name in query.models}
    if len(model_types) != 1:
        raise SemanticQueryValidationError("View 和 Cube namespace 禁止混合")
    model_type = next(iter(model_types))
    if query.query_mode == QueryMode.VIEW:
        if model_type != "view" or len(query.models) != 1:
            raise SemanticQueryValidationError("View Mode 必须且只能查询一个公开 View")
    elif model_type != "cube":
        raise SemanticQueryValidationError("Cube Mode 只能查询公开 Cube，禁止与 View 混合")
    else:
        graph = SemanticJoinGraph.from_catalog(catalog)
        ambiguous_pairs = graph.ambiguous_pairs(query.models)
        if ambiguous_pairs:
            raise SemanticQueryValidationError(
                f"Cube models 存在 Join Path 歧义：{ambiguous_pairs}"
            )
        subgraph = graph.minimal_subgraph(query.models)
        if subgraph is None:
            raise SemanticQueryValidationError("Cube models 位于断连的 Join Graph 中")
        path_models, _ = subgraph
        missing_path_models = set(path_models) - set(query.models)
        if missing_path_models:
            raise SemanticQueryValidationError(
                f"query models 缺少 Join Path 中间 Cube：{sorted(missing_path_models)}"
            )
    validate_projection_consistency(query, projection_mode)
    if not query.measures and not query.dimensions and not query.time_dimensions:
        raise SemanticQueryValidationError("查询至少需要一个 measure、dimension 或 time dimension")
    for name in query.measures:
        _require_type(catalog, query, name, "measure")
    for name in query.dimensions:
        _require_type(catalog, query, name, "dimension")
    for name in query.segments:
        _require_type(catalog, query, name, "segment")
    for semantic_filter in query.filters:
        _filter_types(catalog, query, semantic_filter)
    for time_dimension in query.time_dimensions:
        member = _require_type(catalog, query, time_dimension.dimension, "dimension")
        if member.data_type != "time":
            raise SemanticQueryValidationError(
                f"时间范围只能引用 time dimension：{member.name}"
            )
        if time_dimension.granularity:
            allowed = set(STANDARD_GRANULARITIES) | set(member.granularities)
            if time_dimension.granularity not in allowed:
                raise SemanticQueryValidationError(
                    f"不支持的时间粒度：{time_dimension.granularity}"
                )
    for order in query.order:
        member = _member(catalog, query, order.member)
        if member.member_type == "segment":
            raise SemanticQueryValidationError(f"不能按 segment 排序：{order.member}")
    if projection_mode == ProjectionMode.DEFAULT:
        if query.ungrouped and query.measures:
            raise SemanticQueryValidationError("明细查询不得携带聚合 measure")
        if not query.measures and query.dimensions and not query.ungrouped:
            raise SemanticQueryValidationError("明细查询必须设置 ungrouped=true")
    warnings: list[str] = []
    effective_max = min(max(1, requested_max_rows), max(1, hard_max_rows))
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
