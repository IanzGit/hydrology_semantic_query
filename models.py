from __future__ import annotations

from datetime import UTC, datetime
from enum import Enum
from typing import Any, Literal

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    field_validator,
    model_serializer,
    model_validator,
)


class StepStatus(str, Enum):
    SUCCESS = "success"
    SKIPPED = "skipped"
    FAILED = "failed"


class SemanticCatalogMode(str, Enum):
    AUTO = "auto"
    VECTOR = "vector"
    FULL = "full"


class QueryMode(str, Enum):
    VIEW = "view"
    CUBE = "cube"


class FilterOperator(str, Enum):
    EQUALS = "equals"
    NOT_EQUALS = "notEquals"
    CONTAINS = "contains"
    NOT_CONTAINS = "notContains"
    STARTS_WITH = "startsWith"
    NOT_STARTS_WITH = "notStartsWith"
    ENDS_WITH = "endsWith"
    NOT_ENDS_WITH = "notEndsWith"
    GT = "gt"
    GTE = "gte"
    LT = "lt"
    LTE = "lte"
    SET = "set"
    NOT_SET = "notSet"
    IN_DATE_RANGE = "inDateRange"
    NOT_IN_DATE_RANGE = "notInDateRange"
    BEFORE_DATE = "beforeDate"
    BEFORE_OR_ON_DATE = "beforeOrOnDate"
    AFTER_DATE = "afterDate"
    AFTER_OR_ON_DATE = "afterOrOnDate"


class SemanticFilter(BaseModel):
    member: str | None = None
    operator: FilterOperator | None = None
    values: list[Any] = Field(default_factory=list)
    and_: list[SemanticFilter] = Field(default_factory=list, alias="and")
    or_: list[SemanticFilter] = Field(default_factory=list, alias="or")

    model_config = ConfigDict(populate_by_name=True, extra="forbid")

    @field_validator("values")
    @classmethod
    def validate_scalar_values(cls, values: list[Any]) -> list[Any]:
        if any(not isinstance(value, (str, int, float, bool)) for value in values):
            raise ValueError("values 只能包含非空标量")
        return values

    @model_validator(mode="after")
    def validate_shape(self) -> SemanticFilter:
        logical_count = bool(self.and_) + bool(self.or_)
        leaf = self.member is not None or self.operator is not None or bool(self.values)
        if logical_count:
            if logical_count != 1 or leaf:
                raise ValueError("过滤器必须是单一 and/or 逻辑组")
        elif not self.member or not self.operator:
            raise ValueError("叶子过滤器必须包含 member 和 operator")
        return self

    def to_wire(self) -> dict[str, Any]:
        if self.and_:
            return {"and": [item.to_wire() for item in self.and_]}
        if self.or_:
            return {"or": [item.to_wire() for item in self.or_]}
        payload: dict[str, Any] = {
            "member": self.member,
            "operator": self.operator.value if self.operator else None,
        }
        if self.values:
            payload["values"] = [
                str(value).lower() if isinstance(value, bool) else str(value)
                for value in self.values
            ]
        return payload

    @model_serializer
    def serialize(self) -> dict[str, Any]:
        return self.to_wire()


class TimeDimension(BaseModel):
    dimension: str
    granularity: str | None = None
    date_range: str | list[str] | None = None

    model_config = ConfigDict(populate_by_name=True, extra="forbid")

    @model_validator(mode="before")
    @classmethod
    def normalize_input(cls, data: Any) -> Any:
        if isinstance(data, dict) and "dateRange" in data and "date_range" not in data:
            data = dict(data)
            data["date_range"] = data.pop("dateRange")
        return data

    @model_validator(mode="after")
    def validate_date_range(self) -> TimeDimension:
        if isinstance(self.date_range, list) and len(self.date_range) != 2:
            raise ValueError("date_range 数组必须包含开始和结束两个值")
        return self

    def to_wire(self) -> dict[str, Any]:
        payload: dict[str, Any] = {"dimension": self.dimension}
        if self.granularity:
            payload["granularity"] = self.granularity
        if self.date_range is not None:
            payload["dateRange"] = self.date_range
        return payload


class OrderItem(BaseModel):
    member: str
    direction: Literal["asc", "desc"] = "asc"

    model_config = ConfigDict(extra="forbid")


class SemanticIntent(BaseModel):
    metrics: list[str] = Field(default_factory=list)
    dimensions: list[str] = Field(default_factory=list)
    time: str | None = None
    filters: list[str] = Field(default_factory=list)
    sort: list[str] = Field(default_factory=list)
    limit: int | None = Field(default=None, ge=1)

    model_config = ConfigDict(extra="forbid")

    @model_validator(mode="before")
    @classmethod
    def normalize_input(cls, data: Any) -> Any:
        if not isinstance(data, dict):
            return data
        normalized = dict(data)
        for field in ("metrics", "dimensions", "filters", "sort"):
            value = normalized.get(field)
            if value is None:
                normalized[field] = []
            elif isinstance(value, str):
                normalized[field] = [value] if value.strip() else []
        time_value = normalized.get("time")
        if isinstance(time_value, list):
            values = [str(value).strip() for value in time_value if str(value).strip()]
            normalized["time"] = "；".join(values) if values else None
        elif time_value == "":
            normalized["time"] = None
        return normalized

    def concepts(self) -> list[tuple[str, str]]:
        values = [
            *((value, "measure") for value in self.metrics),
            *((value, "dimension") for value in self.dimensions),
            *((value, "time") for value in [self.time] if value),
            *((value, "filter") for value in self.filters),
            *((value, "sort") for value in self.sort),
        ]
        return list(dict.fromkeys(values))


class SemanticQuery(BaseModel):
    query_mode: QueryMode
    models: list[str] = Field(min_length=1, max_length=4)
    measures: list[str] = Field(default_factory=list)
    dimensions: list[str] = Field(default_factory=list)
    segments: list[str] = Field(default_factory=list)
    filters: list[SemanticFilter] = Field(default_factory=list)
    time_dimensions: list[TimeDimension] = Field(default_factory=list)
    order: list[OrderItem] = Field(default_factory=list)
    limit: int | None = Field(default=None, ge=1)
    offset: int = Field(default=0, ge=0)
    ungrouped: bool = False

    model_config = ConfigDict(populate_by_name=True, extra="forbid")

    @model_validator(mode="before")
    @classmethod
    def normalize_input(cls, data: Any) -> Any:
        if not isinstance(data, dict):
            return data
        if "timeDimensions" in data and "time_dimensions" not in data:
            data = dict(data)
            data["time_dimensions"] = data.pop("timeDimensions")
        if isinstance(data.get("order"), dict):
            data = dict(data)
            data["order"] = [
                {"member": member, "direction": direction}
                for member, direction in data["order"].items()
            ]
        return data

    def to_cube_query(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "measures": self.measures,
            "dimensions": self.dimensions,
            "segments": self.segments,
            "filters": [item.to_wire() for item in self.filters],
            "timeDimensions": [item.to_wire() for item in self.time_dimensions],
            "order": {item.member: item.direction for item in self.order},
            "limit": self.limit,
            "offset": self.offset,
            "ungrouped": self.ungrouped,
            "timezone": None,
        }
        return {key: value for key, value in payload.items() if value not in (None, [], {})}


class CatalogMember(BaseModel):
    name: str
    title: str
    member_type: Literal["measure", "dimension", "segment"]
    data_type: str
    description: str | None = None
    ai_context: str | None = None
    granularities: tuple[str, ...] = ()
    aliases: tuple[str, ...] = ()
    folder: str | None = None
    hierarchy: str | None = None
    primary_key: bool = False


class CatalogModel(BaseModel):
    name: str
    model_type: Literal["view", "cube"]
    title: str
    description: str | None = None
    ai_context: str | None = None
    members: dict[str, CatalogMember] = Field(default_factory=dict)
    folders: tuple[str, ...] = ()
    hierarchies: tuple[str, ...] = ()
    connected_component: str | int | None = None
    aliases: tuple[str, ...] = ()
    use_cases: tuple[str, ...] = ()
    business_priority: float = Field(default=0.5, ge=0, le=1)
    business_domain: str | None = None
    join_edges: tuple[str, ...] = ()


class SemanticCatalog(BaseModel):
    models: dict[str, CatalogModel] = Field(default_factory=dict)


class SemanticModelGap(BaseModel):
    code: Literal["semantic_model_gap"] = "semantic_model_gap"
    message: str
    missing_concepts: list[str] = Field(default_factory=list)
    disconnected_models: list[str] = Field(default_factory=list)
    ambiguous_model_pairs: list[list[str]] = Field(default_factory=list)


class RetrievalTrace(BaseModel):
    view_candidates: list[str] = Field(default_factory=list)
    cube_candidates: list[str] = Field(default_factory=list)
    member_hits: list[str] = Field(default_factory=list)
    member_parent_models: list[str] = Field(default_factory=list)
    view_coverage: dict[str, float] = Field(default_factory=dict)
    cube_coverage: dict[str, float] = Field(default_factory=dict)
    cube_connectivity: dict[str, float] = Field(default_factory=dict)
    rerank_scores: dict[str, float] = Field(default_factory=dict)
    join_paths: list[list[str]] = Field(default_factory=list)
    fallback_level: int = 0
    catalog_batches_analyzed: int = 0


class SemanticContext(BaseModel):
    intent: SemanticIntent
    query_mode: QueryMode
    models: list[str] = Field(min_length=1, max_length=4)
    allowed_members: list[str] = Field(default_factory=list)
    model_details: dict[str, dict[str, Any]] = Field(default_factory=dict)
    member_details: dict[str, dict[str, Any]] = Field(default_factory=dict)
    join_paths: list[list[str]] = Field(default_factory=list)
    fixed_business_context: dict[str, str] = Field(default_factory=dict)
    retrieval_level: int = Field(default=0, ge=0, le=3)


class SemanticColumn(BaseModel):
    name: str
    title: str
    data_type: str = "string"


class StepRecord(BaseModel):
    stage: str
    status: StepStatus
    duration_ms: float = 0.0
    attempt: int = 1
    summary: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)
    timestamp: datetime = Field(default_factory=lambda: datetime.now(UTC))


class SemanticQueryError(BaseModel):
    stage: str
    code: str
    message: str
    retryable: bool = False
    status_code: int | None = None
    details: dict[str, Any] = Field(default_factory=dict)


class SemanticQueryResult(BaseModel):
    success: bool
    executed: bool = False
    semantic_query: SemanticQuery | None = None
    columns: list[SemanticColumn] = Field(default_factory=list)
    rows: list[dict[str, Any]] = Field(default_factory=list)
    row_count: int = 0
    attempts: int = 0
    catalog_mode: SemanticCatalogMode | None = None
    query_mode: QueryMode | None = None
    selected_models: list[str] = Field(default_factory=list)
    retrieval_trace: RetrievalTrace | None = None
    semantic_model_gap: SemanticModelGap | None = None
    warnings: list[str] = Field(default_factory=list)
    steps: list[StepRecord] = Field(default_factory=list)
    error: SemanticQueryError | None = None


SemanticFilter.model_rebuild()
