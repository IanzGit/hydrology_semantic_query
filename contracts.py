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


class QueryMode(str, Enum):
    VIEW = "view"
    CUBE = "cube"


class MainAgentAction(str, Enum):
    QUERY = "query"
    RESPOND = "respond"


class QueryTask(BaseModel):
    task_id: str = Field(min_length=1, max_length=64, pattern=r"^[A-Za-z0-9_-]+$")
    objective: str = Field(min_length=1, max_length=2000)
    depends_on: list[str] = Field(default_factory=list)
    condition: str | None = Field(default=None, max_length=1000)

    model_config = ConfigDict(extra="forbid")

    @field_validator("task_id", "objective", "condition")
    @classmethod
    def strip_text(cls, value: str | None) -> str | None:
        if value is None:
            return None
        stripped = value.strip()
        if not stripped:
            raise ValueError("文本字段不能为空")
        return stripped

    @field_validator("depends_on")
    @classmethod
    def validate_dependencies(cls, values: list[str]) -> list[str]:
        normalized = [value.strip() for value in values]
        if any(not value for value in normalized):
            raise ValueError("depends_on 不能包含空任务 ID")
        if len(set(normalized)) != len(normalized):
            raise ValueError("depends_on 不能包含重复任务 ID")
        return normalized


class ReportSectionRequirement(BaseModel):
    section_id: str = Field(
        min_length=1,
        max_length=64,
        pattern=r"^[A-Za-z0-9_-]+$",
    )
    title: str = Field(min_length=1, max_length=200)
    objective: str = Field(min_length=1, max_length=1000)
    source_task_ids: list[str] = Field(min_length=1)

    model_config = ConfigDict(extra="forbid")

    @field_validator("section_id", "title", "objective")
    @classmethod
    def strip_section_text(cls, value: str) -> str:
        stripped = value.strip()
        if not stripped:
            raise ValueError("报告章节字段不能为空")
        return stripped

    @field_validator("source_task_ids")
    @classmethod
    def validate_source_tasks(cls, values: list[str]) -> list[str]:
        normalized = [value.strip() for value in values]
        if any(not value for value in normalized):
            raise ValueError("source_task_ids 不能包含空任务 ID")
        if len(set(normalized)) != len(normalized):
            raise ValueError("source_task_ids 不能包含重复任务 ID")
        return normalized

class MainAgentDecision(BaseModel):
    action: MainAgentAction
    matched_playbook: str | None = Field(max_length=255)
    query_tasks: list[QueryTask]
    report_sections: list[ReportSectionRequirement]
    direct_answer: str | None = Field(max_length=4000)
    summary: str = Field(min_length=1, max_length=1000)

    model_config = ConfigDict(extra="forbid")

    @field_validator("matched_playbook", "direct_answer")
    @classmethod
    def strip_optional_decision_text(cls, value: str | None) -> str | None:
        if value is None:
            return None
        stripped = value.strip()
        return stripped or None

    @field_validator("summary")
    @classmethod
    def strip_summary(cls, value: str) -> str:
        stripped = value.strip()
        if not stripped:
            raise ValueError("summary 不能为空")
        return stripped

    @model_validator(mode="after")
    def validate_decision_shape(self) -> MainAgentDecision:
        task_ids = [task.task_id for task in self.query_tasks]
        section_ids = [section.section_id for section in self.report_sections]
        if len(set(task_ids)) != len(task_ids):
            raise ValueError("query_tasks 的 task_id 必须唯一")
        if len(set(section_ids)) != len(section_ids):
            raise ValueError("report_sections 的 section_id 必须唯一")
        if self.action == MainAgentAction.QUERY and not self.query_tasks:
            raise ValueError("query 动作必须包含查询任务")
        if self.action == MainAgentAction.QUERY and not self.report_sections:
            raise ValueError("query 动作必须包含报告章节")
        if self.action == MainAgentAction.RESPOND and not self.direct_answer:
            raise ValueError("respond 动作必须包含 direct_answer")
        if self.action == MainAgentAction.RESPOND and (
            self.query_tasks or self.report_sections
        ):
            raise ValueError("respond 动作不能包含查询任务或报告章节")
        if self.action != MainAgentAction.RESPOND and self.direct_answer:
            raise ValueError("只有 respond 动作可以包含 direct_answer")
        return self


class ExecutionPlanRevision(BaseModel):
    revision: int = Field(ge=1)
    action: MainAgentAction
    matched_playbook: str | None = None
    query_tasks: list[QueryTask] = Field(default_factory=list)
    report_sections: list[ReportSectionRequirement] = Field(default_factory=list)
    summary: str

    model_config = ConfigDict(extra="forbid")


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


class ChartType(str, Enum):
    BAR = "BAR"
    LINE = "LINE"
    PIE = "PIE"
    BAR_STACK = "BAR_STACK"


class ChartAggregation(str, Enum):
    NONE = "none"
    COUNT = "count"
    SUM = "sum"
    AVG = "avg"
    MIN = "min"
    MAX = "max"


class ChartFilterOperator(str, Enum):
    EQ = "eq"
    NE = "ne"
    IN = "in"
    NOT_IN = "not_in"
    GT = "gt"
    GTE = "gte"
    LT = "lt"
    LTE = "lte"
    BETWEEN = "between"
    IS_NULL = "is_null"
    NOT_NULL = "not_null"


class ChartFilter(BaseModel):
    field: str = Field(min_length=1)
    operator: ChartFilterOperator
    value: str | int | float | bool | list[str | int | float | bool] | None = None

    model_config = ConfigDict(extra="forbid")


class ChartSort(BaseModel):
    by: Literal["x", "value"] = "x"
    direction: Literal["asc", "desc"] = "asc"

    model_config = ConfigDict(extra="forbid")

    @model_validator(mode="before")
    @classmethod
    def normalize_sort_key(cls, data: Any) -> Any:
        if isinstance(data, dict) and "by" not in data:
            for alias in ("field", "target"):
                if alias in data:
                    data = dict(data)
                    data["by"] = data.pop(alias)
                    break
        return data


class DynamicChartPlan(BaseModel):
    title: str = Field(min_length=1, max_length=200)
    chart_type: ChartType
    priority: int = Field(ge=1, le=1000)
    x_field: str = Field(min_length=1)
    value_field: str | None = None
    series_field: str | None = None
    filters: list[ChartFilter] = Field(default_factory=list)
    group_by: list[str] = Field(default_factory=list)
    aggregation: ChartAggregation
    sort: ChartSort = Field(default_factory=ChartSort)
    unit_field: str | None = None
    limit: int = Field(default=200, ge=1, le=1000)

    model_config = ConfigDict(extra="forbid")

    @field_validator("title", "x_field", "value_field", "series_field", "unit_field")
    @classmethod
    def strip_chart_text(cls, value: str | None) -> str | None:
        if value is None:
            return None
        stripped = value.strip()
        if not stripped:
            raise ValueError("图表文本字段不能为空")
        return stripped

    @model_validator(mode="after")
    def validate_chart_shape(self) -> DynamicChartPlan:
        if self.aggregation == ChartAggregation.COUNT:
            if self.value_field is not None:
                raise ValueError("count 聚合不能提供 value_field")
        elif self.value_field is None:
            raise ValueError("非 count 图表必须提供 value_field")
        expected = [self.x_field]
        if self.series_field:
            expected.append(self.series_field)
        if len(self.group_by) != len(set(self.group_by)):
            raise ValueError("group_by 不能包含重复字段")
        if set(self.group_by) != set(expected):
            raise ValueError("group_by 必须只包含 x_field 和 series_field")
        if self.chart_type == ChartType.PIE and self.series_field is not None:
            raise ValueError("饼图不能提供 series_field")
        if self.chart_type == ChartType.BAR_STACK and self.series_field is None:
            raise ValueError("堆叠柱状图必须提供 series_field")
        return self


class TaskChartPlans(BaseModel):
    source_task_id: str = Field(min_length=1)
    charts: list[DynamicChartPlan] = Field(default_factory=list, max_length=4)
    no_chart_reason: str | None = None

    model_config = ConfigDict(extra="forbid")

    @model_validator(mode="after")
    def validate_task_plan(self) -> TaskChartPlans:
        if self.charts and self.no_chart_reason:
            raise ValueError("有图表计划时不能提供 no_chart_reason")
        if not self.charts and not (self.no_chart_reason or "").strip():
            raise ValueError("无图表计划时必须提供 no_chart_reason")
        return self


class ChartPlanningResponse(BaseModel):
    tasks: list[TaskChartPlans] = Field(default_factory=list)

    model_config = ConfigDict(extra="forbid")


class HighFrequencyValue(BaseModel):
    value: Any
    count: int = Field(ge=1)
    ratio: float = Field(ge=0, le=1)


class DataColumnProfile(BaseModel):
    name: str
    title: str
    declared_type: str
    inferred_type: Literal["numeric", "time", "category", "unknown"]
    row_count: int = Field(ge=0)
    null_count: int = Field(ge=0)
    null_rate: float = Field(ge=0, le=1)
    unique_count: int = Field(ge=0)
    finite_numeric_ratio: float = Field(ge=0, le=1)
    numeric_min: float | None = None
    numeric_max: float | None = None
    time_earliest: str | None = None
    time_latest: str | None = None
    top_values: list[HighFrequencyValue] = Field(default_factory=list)


class TaskDataProfile(BaseModel):
    source_task_id: str
    objective: str
    row_count: int = Field(ge=0)
    columns: list[DataColumnProfile] = Field(default_factory=list)
    first_rows: list[dict[str, Any]] = Field(default_factory=list)
    sampled_rows: list[dict[str, Any]] = Field(default_factory=list)
    has_chart_opportunity: bool = False


class ChartEvidence(BaseModel):
    chart_id: str
    title: str
    chart_type: ChartType
    source_task_id: str
    filters: list[ChartFilter] = Field(default_factory=list)
    aggregation: ChartAggregation
    unit: str
    input_row_count: int = Field(ge=0)
    filtered_row_count: int = Field(ge=0)
    valid_row_count: int = Field(ge=0)
    displayed_point_count: int = Field(ge=0)
    truncated: bool = False
    summary: str


class RenderedChart(BaseModel):
    plan: DynamicChartPlan
    evidence: ChartEvidence
    series_data: list[dict[str, Any]] = Field(default_factory=list)

    model_config = ConfigDict(extra="forbid")


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
    PARTIAL_SUCCESS = "partial_success"
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


class QueryExecutionRecord(BaseModel):
    query_number: int = Field(ge=1)
    task_id: str | None = None
    query_goal: str = Field(min_length=1)
    semantic_query: SemanticQuery
    outcome: QueryOutcome
    columns: list[SemanticColumn] = Field(default_factory=list)
    rows: list[dict[str, Any]] = Field(default_factory=list)
    row_count: int = 0
    attempt: int = Field(ge=1)
    compiled_sql: str | None = None
    compiled_params: list[Any] = Field(default_factory=list)
    selected_models: list[str] = Field(default_factory=list)

    model_config = ConfigDict(extra="forbid")


class TaskExecutionStatus(str, Enum):
    SUCCESS = "success"
    NO_DATA = "no_data"
    FAILED = "failed"
    SKIPPED = "skipped"


class TaskExecutionResult(BaseModel):
    task: QueryTask
    status: TaskExecutionStatus
    outcome: QueryOutcome | None = None
    query_record: QueryExecutionRecord | None = None
    error: SemanticQueryError | None = None
    warnings: list[str] = Field(default_factory=list)
    attempts: int = Field(default=0, ge=0)
    summary: str | None = None

    model_config = ConfigDict(extra="forbid")


class QueryTaskExecutionContext(BaseModel):
    original_question: str = Field(min_length=1)
    standalone_question: str = Field(min_length=1)
    plan: list[QueryTask] = Field(min_length=1)
    current_task: QueryTask
    completed_results: list[TaskExecutionResult] = Field(default_factory=list)

    model_config = ConfigDict(extra="forbid")


class ReportTask(BaseModel):
    original_question: str = Field(min_length=1)
    sections: list[ReportSectionRequirement] = Field(min_length=1)
    task_results: list[TaskExecutionResult] = Field(default_factory=list)

    model_config = ConfigDict(extra="forbid")


class SemanticQueryResult(BaseModel):
    outcome: QueryOutcome
    semantic_query: SemanticQuery | None = None
    columns: list[SemanticColumn] = Field(default_factory=list)
    rows: list[dict[str, Any]] = Field(default_factory=list)
    row_count: int = 0
    attempts: int = 0
    query_count: int = 0
    query_history: list[QueryExecutionRecord] = Field(default_factory=list)
    matched_playbook: str | None = None
    plan_revisions: list[ExecutionPlanRevision] = Field(default_factory=list)
    task_results: list[TaskExecutionResult] = Field(default_factory=list)
    compiled_sql: str | None = None
    compiled_params: list[Any] = Field(default_factory=list)
    catalog_mode: SemanticCatalogMode | None = None
    query_mode: QueryMode | None = None
    selected_models: list[str] = Field(default_factory=list)
    retrieval_trace: RetrievalTrace | None = None
    warnings: list[str] = Field(default_factory=list)
    steps: list[StepRecord] = Field(default_factory=list)
    error: SemanticQueryError | None = None
    model_config = ConfigDict(extra="forbid")


__all__ = [
    "SemanticCatalogMode",
    "RetrievalHit",
    "RetrievalTrace",
    "QueryMode",
    "MainAgentAction",
    "QueryTask",
    "ReportSectionRequirement",
    "MainAgentDecision",
    "ExecutionPlanRevision",
    "FilterOperator",
    "SemanticFilter",
    "TimeDimension",
    "OrderItem",
    "SemanticQuery",
    "ChartType",
    "ChartAggregation",
    "ChartFilterOperator",
    "ChartFilter",
    "ChartSort",
    "DynamicChartPlan",
    "TaskChartPlans",
    "ChartPlanningResponse",
    "HighFrequencyValue",
    "DataColumnProfile",
    "TaskDataProfile",
    "ChartEvidence",
    "RenderedChart",
    "StepStatus",
    "FailureKind",
    "QueryOutcome",
    "SemanticColumn",
    "StepRecord",
    "SemanticQueryError",
    "QueryExecutionRecord",
    "TaskExecutionStatus",
    "TaskExecutionResult",
    "QueryTaskExecutionContext",
    "ReportTask",
    "SemanticQueryResult",
]
