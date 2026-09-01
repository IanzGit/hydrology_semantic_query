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


class SemanticCatalogMode(str, Enum):
    AUTO = "auto"
    VECTOR = "vector"
    FULL = "full"


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
    default_projection: tuple[str, ...] = ()


class SemanticCatalog(BaseModel):
    models: dict[str, CatalogModel] = Field(default_factory=dict)


class CatalogContextItem(BaseModel):
    item_type: Literal["model", "member", "view_folder", "join_component"]
    name: str
    model_name: str | None = None
    score: float | None = None
    payload: dict[str, Any] = Field(default_factory=dict)

    model_config = ConfigDict(extra="forbid")

class RetrievalHit(BaseModel):
    item_type: Literal["model", "member", "view_folder", "join_component"]
    name: str
    model_name: str | None = None
    score: float | None = None

    model_config = ConfigDict(extra="forbid")


class RetrievalTrace(BaseModel):
    strategy: SemanticCatalogMode
    queries: list[str] = Field(default_factory=list)
    hits: list[RetrievalHit] = Field(default_factory=list)
    index_source: str = "disabled"

    model_config = ConfigDict(extra="forbid")


class SemanticContext(BaseModel):
    strategy: SemanticCatalogMode
    items: list[CatalogContextItem] = Field(default_factory=list)
    retrieval_round: int = Field(default=1, ge=1)

    model_config = ConfigDict(extra="forbid")

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
        if {"and_", "or_"} <= self.model_fields_set:
            raise ValueError("过滤器不能同时包含 and 和 or")
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


SemanticFilter.model_rebuild()

class ColumnRole(str, Enum):
    MEASURE = "measure"
    TIME = "time"
    CATEGORY = "category"
    LATITUDE = "latitude"
    LONGITUDE = "longitude"
    STATUS = "status"
    IDENTIFIER = "identifier"
    UNKNOWN = "unknown"


class ResultShape(str, Enum):
    EMPTY = "empty"
    SCALAR = "scalar"
    TEMPORAL = "temporal"
    CATEGORICAL = "categorical"
    GEOSPATIAL = "geospatial"
    STATUS = "status"
    NUMERIC = "numeric"
    TABULAR = "tabular"


class ChartType(str, Enum):
    BAR = "BAR"
    LINE = "LINE"
    PIE = "PIE"


class PresentationBlockType(str, Enum):
    KPI = "kpi"
    STATUS = "status"
    CHART = "chart"
    MAP = "map"
    TABLE = "table"


class FieldRef(BaseModel):
    name: str
    title: str
    data_type: str


class ColumnProfile(FieldRef):
    role: ColumnRole
    member_type: str = "unknown"
    null_count: int = 0
    distinct_count: int = 0
    minimum: float | None = None
    maximum: float | None = None


class ResultProfile(BaseModel):
    row_count: int
    column_count: int
    shape: ResultShape
    columns: list[ColumnProfile] = Field(default_factory=list)
    measure_fields: list[str] = Field(default_factory=list)
    time_fields: list[str] = Field(default_factory=list)
    category_fields: list[str] = Field(default_factory=list)
    status_fields: list[str] = Field(default_factory=list)
    identifier_fields: list[str] = Field(default_factory=list)
    latitude_field: str | None = None
    longitude_field: str | None = None
    primary_measure: str | None = None
    primary_time: str | None = None
    primary_category: str | None = None


class KpiSpec(BaseModel):
    fields: list[FieldRef]
    mode: Literal["value", "latest"] = "value"
    time_field: FieldRef | None = None


class StatusSpec(BaseModel):
    field: FieldRef
    label_field: FieldRef | None = None
    max_items: int = Field(default=8, ge=1, le=20)


class ChartSpec(BaseModel):
    chart_type: ChartType
    x: FieldRef
    y: list[FieldRef] = Field(default_factory=list)
    series: FieldRef | None = None
    sort: Literal["asc", "desc", "none"] = "asc"
    limit: int = Field(default=200, ge=1, le=1000)


class MapSpec(BaseModel):
    latitude: FieldRef
    longitude: FieldRef
    value: FieldRef | None = None
    label: FieldRef | None = None
    limit: int = Field(default=500, ge=1, le=2000)


class TableSpec(BaseModel):
    fields: list[FieldRef]
    limit: int = Field(default=500, ge=1, le=2000)


PresentationSpec = KpiSpec | StatusSpec | ChartSpec | MapSpec | TableSpec


class PlannedBlock(BaseModel):
    id: str
    type: PresentationBlockType
    title: str
    description: str | None = None
    priority: int = 100
    config: PresentationSpec


class PresentationPlan(BaseModel):
    title: str
    profile: ResultProfile
    blocks: list[PlannedBlock] = Field(default_factory=list)


class ReportBlock(BaseModel):
    id: str
    type: PresentationBlockType
    title: str
    description: str | None = None
    data: Any
    config: dict[str, Any] = Field(default_factory=dict)
    priority: int = 100


class ReportSection(BaseModel):
    id: str
    title: str
    blocks: list[ReportBlock] = Field(default_factory=list)


class StructuredReport(BaseModel):
    protocol_version: Literal["1.0"] = "1.0"
    type: Literal["structured_report"] = "structured_report"
    title: str
    summary: str
    profile: ResultProfile
    sections: list[ReportSection] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)

class StepStatus(str, Enum):
    SUCCESS = "success"
    SKIPPED = "skipped"
    FAILED = "failed"


class FailureKind(str, Enum):
    PLANNER = "planner"
    VALIDATION = "validation"
    EXECUTION = "execution"
    SYSTEM = "system"


class QueryOutcome(str, Enum):
    SUCCESS = "success"
    NO_DATA = "no_data"
    PLANNER_ERROR = "planner_error"
    EXECUTION_ERROR = "execution_error"
    SYSTEM_ERROR = "system_error"


class SemanticColumn(BaseModel):
    name: str
    title: str
    data_type: str = "string"
    member_type: Literal["measure", "dimension", "time_dimension", "unknown"] = "unknown"


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
    kind: FailureKind
    internal_message: str
    internal_details: dict[str, Any] = Field(default_factory=dict)
    retryable: bool = False
    status_code: int | None = None


class SemanticQueryResult(BaseModel):
    outcome: QueryOutcome
    semantic_query: SemanticQuery | None = None
    columns: list[SemanticColumn] = Field(default_factory=list)
    rows: list[dict[str, Any]] = Field(default_factory=list)
    row_count: int = 0
    attempts: int = 0
    compiled_sql: str | None = None
    compiled_params: list[Any] = Field(default_factory=list)
    catalog_mode: SemanticCatalogMode | None = None
    query_mode: QueryMode | None = None
    selected_models: list[str] = Field(default_factory=list)
    retrieval_trace: RetrievalTrace | None = None
    warnings: list[str] = Field(default_factory=list)
    steps: list[StepRecord] = Field(default_factory=list)
    error: SemanticQueryError | None = None
    presentation: StructuredReport | None = None

    model_config = ConfigDict(extra="forbid")
