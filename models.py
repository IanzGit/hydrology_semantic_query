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


class MemberRequirement(BaseModel):
    phrase: str
    role: Literal["project", "aggregate", "group", "filter", "order"]
    required: bool = True
    operator: str | None = None
    values: list[Any] = Field(default_factory=list)
    direction: Literal["asc", "desc"] | None = None
    source: Literal["explicit", "legacy", "resolver"] = "explicit"

    model_config = ConfigDict(extra="forbid")

    @field_validator("phrase")
    @classmethod
    def validate_phrase(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("成员需求 phrase 不能为空")
        return value

    @field_validator("values")
    @classmethod
    def validate_values(cls, values: list[Any]) -> list[Any]:
        if any(not isinstance(value, (str, int, float, bool)) for value in values):
            raise ValueError("成员需求 values 只能包含标量")
        return values


class TemporalIntent(BaseModel):
    operator: Literal["current", "latest", "range", "before", "after", "as_of"]
    value: Any | None = None
    field_hint: str | None = None
    raw_phrase: str | None = None

    model_config = ConfigDict(extra="forbid")


class SemanticIntent(BaseModel):
    subjects: list[str] = Field(default_factory=list)
    member_requirements: list[MemberRequirement] = Field(default_factory=list)
    qualifiers: list[str] = Field(default_factory=list)
    temporal: TemporalIntent | None = None
    result_shape: Literal["detail", "aggregate", "trend"] = "detail"
    limit: int | None = Field(default=None, ge=1)

    model_config = ConfigDict(extra="forbid")

    @model_validator(mode="before")
    @classmethod
    def normalize_input(cls, data: Any) -> Any:
        if not isinstance(data, dict):
            return data
        normalized = dict(data)
        new_shape = any(
            key in normalized
            for key in (
                "subjects",
                "member_requirements",
                "qualifiers",
                "temporal",
                "result_shape",
            )
        )
        if new_shape:
            for field in ("subjects", "qualifiers"):
                value = normalized.get(field)
                if value is None:
                    normalized[field] = []
                elif isinstance(value, str):
                    normalized[field] = [value] if value.strip() else []
            requirements = normalized.get("member_requirements")
            if requirements is None:
                normalized["member_requirements"] = []
            elif isinstance(requirements, str):
                normalized["member_requirements"] = [{
                    "phrase": requirements,
                    "role": "project",
                    "source": "legacy",
                }]
            else:
                normalized["member_requirements"] = [
                    {
                        "phrase": value,
                        "role": "project",
                        "source": "legacy",
                    }
                    if isinstance(value, str)
                    else value
                    for value in requirements
                ]
            temporal = normalized.get("temporal")
            if isinstance(temporal, str) and temporal.strip():
                normalized["temporal"] = cls._temporal_from_phrase(temporal)
            if "result_shape" not in normalized:
                normalized["result_shape"] = cls._infer_result_shape(
                    normalized["member_requirements"]
                )
            return normalized

        def values(field: str) -> list[str]:
            value = normalized.get(field)
            if value is None:
                return []
            if isinstance(value, str):
                return [value] if value.strip() else []
            return [str(item).strip() for item in value if str(item).strip()]

        requirements: list[dict[str, Any]] = []
        for value in values("metrics"):
            requirements.append({
                "phrase": value,
                "role": "aggregate",
                "source": "legacy",
            })
        for value in values("dimensions"):
            requirements.append({
                "phrase": value,
                "role": "project",
                "source": "legacy",
            })
        qualifiers: list[str] = []
        for value in values("filters"):
            if any(term in value for term in ("有效", "启用", "活动中")):
                qualifiers.append(value)
            else:
                requirements.append({
                    "phrase": value,
                    "role": "filter",
                    "source": "legacy",
                })
        for value in values("sort"):
            requirements.append({
                "phrase": value,
                "role": "order",
                "source": "legacy",
            })
        time_value = normalized.get("time")
        temporal = None
        if isinstance(time_value, list):
            raw_phrase = "；".join(
                str(value).strip() for value in time_value if str(value).strip()
            )
            temporal = cls._temporal_from_phrase(raw_phrase) if raw_phrase else None
        elif isinstance(time_value, str) and time_value.strip():
            temporal = cls._temporal_from_phrase(time_value)
        normalized = {
            "subjects": [],
            "member_requirements": requirements,
            "qualifiers": qualifiers,
            "temporal": temporal,
            "result_shape": cls._infer_result_shape(requirements),
            "limit": normalized.get("limit"),
        }
        return normalized

    @staticmethod
    def _temporal_from_phrase(value: str) -> dict[str, Any]:
        phrase = value.strip()
        if "最新" in phrase or "最近一条" in phrase:
            operator = "latest"
        elif "当前" in phrase:
            operator = "current"
        elif "之前" in phrase or "以前" in phrase:
            operator = "before"
        elif "之后" in phrase or "以后" in phrase:
            operator = "after"
        else:
            operator = "range"
        return {"operator": operator, "raw_phrase": phrase}

    @staticmethod
    def _infer_result_shape(requirements: list[Any]) -> str:
        if any(
            (
                item.get("role")
                if isinstance(item, dict)
                else getattr(item, "role", None)
            )
            == "aggregate"
            for item in requirements
        ):
            return "aggregate"
        return "detail"

    def hard_requirements(self) -> list[MemberRequirement]:
        return [item for item in self.member_requirements if item.required]

    def scope_terms(self) -> list[str]:
        return list(dict.fromkeys(self.subjects))

    @property
    def metrics(self) -> list[str]:
        return [item.phrase for item in self.member_requirements if item.role == "aggregate"]

    @property
    def dimensions(self) -> list[str]:
        return [
            item.phrase
            for item in self.member_requirements
            if item.role in {"project", "group"}
        ]

    @property
    def filters(self) -> list[str]:
        return [item.phrase for item in self.member_requirements if item.role == "filter"]

    @property
    def sort(self) -> list[str]:
        return [item.phrase for item in self.member_requirements if item.role == "order"]

    @property
    def time(self) -> str | None:
        if self.temporal is None:
            return None
        return self.temporal.raw_phrase or self.temporal.operator


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
    scope_scores: dict[str, float] = Field(default_factory=dict)
    member_coverage: dict[str, float] = Field(default_factory=dict)
    resolved_requirements: dict[str, str] = Field(default_factory=dict)
    missing_requirements: list[str] = Field(default_factory=list)
    resolved_qualifiers: dict[str, str] = Field(default_factory=dict)
    operator_resolution: dict[str, Any] = Field(default_factory=dict)
    fallback_anchor: list[str] = Field(default_factory=list)
    suggested_members: list[str] = Field(default_factory=list)
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
    suggested_members: list[str] = Field(default_factory=list)
    projection_policy: Literal["explicit", "model_default", "summary"] = "explicit"
    fallback_anchor: list[str] = Field(default_factory=list)
    operator_resolution: dict[str, Any] = Field(default_factory=dict)
    resolved_qualifiers: dict[str, str] = Field(default_factory=dict)
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
