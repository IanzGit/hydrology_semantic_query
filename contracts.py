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
    REPORT = "report"
    RESPOND = "respond"


class ReportAnalysisMethod(str, Enum):
    OVERVIEW = "overview"
    TREND = "trend"
    ANOMALY = "anomaly"
    COMPARISON = "comparison"
    CORRELATION = "correlation"
    CONCLUSION = "conclusion"


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
    """主 Agent 下发的单个最终报告章节要求。"""

    section_id: str = Field(
        min_length=1,
        max_length=64,
        pattern=r"^[A-Za-z0-9_-]+$",
    )
    title: str = Field(min_length=1, max_length=200)
    objective: str = Field(min_length=1, max_length=1000)
    source_task_ids: list[str] = Field(min_length=1)
    analysis_methods: list[ReportAnalysisMethod] = Field(
        default_factory=lambda: [ReportAnalysisMethod.OVERVIEW]
    )

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

    @field_validator("analysis_methods")
    @classmethod
    def validate_analysis_methods(
        cls,
        values: list[ReportAnalysisMethod],
    ) -> list[ReportAnalysisMethod]:
        if not values:
            raise ValueError("analysis_methods 不能为空")
        if len(set(values)) != len(values):
            raise ValueError("analysis_methods 不能重复")
        return values


class MainAgentDecision(BaseModel):
    action: MainAgentAction
    matched_playbook: str | None = Field(default=None, max_length=255)
    query_tasks: list[QueryTask] = Field(default_factory=list)
    report_sections: list[ReportSectionRequirement] = Field(default_factory=list)
    direct_answer: str | None = Field(default=None, max_length=4000)
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
        if self.action == MainAgentAction.REPORT and not self.report_sections:
            raise ValueError("report 动作必须包含报告章节")
        if self.action == MainAgentAction.REPORT and self.query_tasks:
            raise ValueError("report 动作不能包含查询任务")
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


class ReportFactCategory(str, Enum):
    SCOPE = "scope"
    QUALITY = "quality"
    METRIC = "metric"
    TREND = "trend"
    DISTRIBUTION = "distribution"
    STATUS = "status"
    THRESHOLD = "threshold"
    ANOMALY = "anomaly"
    CORRELATION = "correlation"


class ReportFact(BaseModel):
    fact_id: str
    category: ReportFactCategory
    title: str
    display_text: str
    value: Any = None
    unit: str | None = None
    evidence_fields: list[str] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)

    model_config = ConfigDict(extra="forbid")


class ReportLimitation(BaseModel):
    code: str
    message: str

    model_config = ConfigDict(extra="forbid")


class ReportNarrativeInsight(BaseModel):
    title: str
    section_id: str | None = None
    fact_ids: list[str] = Field(min_length=1, max_length=6)
    interpretation: str
    impact: str
    possible_cause: str
    recommendation: str
    certainty: Literal["high", "medium", "low"]

    model_config = ConfigDict(extra="forbid")


class ReportSectionNarrative(BaseModel):
    """报告模型为单个规划章节生成的受事实约束叙事。"""

    section_id: str
    fact_ids: list[str] = Field(default_factory=list, max_length=8)
    analysis: str
    impact: str
    possible_cause: str
    conclusion: str
    recommendation: str
    certainty: Literal["high", "medium", "low"]

    model_config = ConfigDict(extra="forbid")


class ReportNarrativeDraft(BaseModel):
    """报告模型生成的全局摘要与章节化叙事草稿。"""

    title: str
    executive_summary: str
    insights: list[ReportNarrativeInsight] = Field(max_length=8)
    section_narratives: list[ReportSectionNarrative] = Field(default_factory=list)

    model_config = ConfigDict(extra="forbid")


class ReportAnalysis(BaseModel):
    profile: ResultProfile
    facts: list[ReportFact] = Field(default_factory=list)
    limitations: list[ReportLimitation] = Field(default_factory=list)

    model_config = ConfigDict(extra="forbid")


class SectionAnalysis(BaseModel):
    """按主 Agent 章节要求聚合的数据画像、事实和分析局限。"""

    requirement: ReportSectionRequirement
    source_profiles: dict[str, ResultProfile] = Field(default_factory=dict)
    available_source_task_ids: list[str] = Field(default_factory=list)
    unavailable_source_task_ids: list[str] = Field(default_factory=list)
    facts: list[ReportFact] = Field(default_factory=list)
    limitations: list[ReportLimitation] = Field(default_factory=list)

    model_config = ConfigDict(extra="forbid")


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


class ReportSectionContent(BaseModel):
    """章节中围绕结构化图表组织的文字内容。"""

    evidence: str
    analysis: str
    conclusion: str

    model_config = ConfigDict(extra="forbid")


class ReportSection(BaseModel):
    """最终结构化报告中可独立包含文字与展示块的章节。"""

    id: str
    title: str
    objective: str = ""
    source_task_ids: list[str] = Field(default_factory=list)
    analysis_methods: list[ReportAnalysisMethod] = Field(default_factory=list)
    fact_ids: list[str] = Field(default_factory=list)
    limitation_codes: list[str] = Field(default_factory=list)
    content: ReportSectionContent | None = None
    no_chart_reason: str | None = None
    blocks: list[ReportBlock] = Field(default_factory=list)


class StructuredReport(BaseModel):
    """支持旧版任务式和新版章节式组织的内部报告协议。"""

    protocol_version: Literal["1.0", "1.1"] = "1.0"
    type: Literal["structured_report"] = "structured_report"
    title: str
    summary: str
    profile: ResultProfile
    sections: list[ReportSection] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)
    facts: list[ReportFact] = Field(default_factory=list)
    insights: list[ReportNarrativeInsight] = Field(default_factory=list)
    limitations: list[ReportLimitation] = Field(default_factory=list)
    generated_at: datetime = Field(default_factory=lambda: datetime.now(UTC))


class VisualizationCandidate(BaseModel):
    """供可视化规划器选择的已校验、可直接渲染图表候选。"""

    candidate_id: str
    section_id: str
    title: str
    purpose: str
    source_task_ids: list[str] = Field(min_length=1)
    fact_ids: list[str] = Field(default_factory=list)
    block: ReportBlock

    model_config = ConfigDict(extra="forbid")


class VisualizationSelection(BaseModel):
    """模型对一个合法图表候选的章节级选择。"""

    candidate_id: str = Field(min_length=1, max_length=200)
    rationale: str = Field(min_length=1, max_length=1000)

    model_config = ConfigDict(extra="forbid")


class SectionVisualizationPlan(BaseModel):
    """单个报告章节的可视化决策。"""

    section_id: str
    charts: list[VisualizationSelection] = Field(default_factory=list, max_length=2)
    no_chart_reason: str | None = None

    model_config = ConfigDict(extra="forbid")

    @model_validator(mode="after")
    def validate_empty_plan_reason(self) -> SectionVisualizationPlan:
        if not self.charts and not str(self.no_chart_reason or "").strip():
            raise ValueError("无图表章节必须提供 no_chart_reason")
        if self.charts and self.no_chart_reason is not None:
            raise ValueError("已有图表的章节不能同时提供 no_chart_reason")
        return self


class VisualizationPlan(BaseModel):
    """整份报告按章节组织的可视化规划结果。"""

    sections: list[SectionVisualizationPlan]

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
    presentation: StructuredReport | None = None

    model_config = ConfigDict(extra="forbid")


__all__ = [
    "SemanticCatalogMode",
    "RetrievalHit",
    "RetrievalTrace",
    "QueryMode",
    "MainAgentAction",
    "ReportAnalysisMethod",
    "QueryTask",
    "ReportSectionRequirement",
    "MainAgentDecision",
    "ExecutionPlanRevision",
    "FilterOperator",
    "SemanticFilter",
    "TimeDimension",
    "OrderItem",
    "SemanticQuery",
    "ColumnRole",
    "ResultShape",
    "ChartType",
    "PresentationBlockType",
    "FieldRef",
    "ColumnProfile",
    "ResultProfile",
    "ReportFactCategory",
    "ReportFact",
    "ReportLimitation",
    "ReportNarrativeInsight",
    "ReportSectionNarrative",
    "ReportNarrativeDraft",
    "ReportAnalysis",
    "SectionAnalysis",
    "KpiSpec",
    "StatusSpec",
    "ChartSpec",
    "MapSpec",
    "TableSpec",
    "PresentationSpec",
    "PlannedBlock",
    "PresentationPlan",
    "ReportBlock",
    "ReportSection",
    "ReportSectionContent",
    "StructuredReport",
    "VisualizationCandidate",
    "VisualizationSelection",
    "SectionVisualizationPlan",
    "VisualizationPlan",
    "StepStatus",
    "FailureKind",
    "QueryOutcome",
    "SemanticColumn",
    "StepRecord",
    "SemanticQueryError",
    "QueryExecutionRecord",
    "TaskExecutionStatus",
    "TaskExecutionResult",
    "ReportTask",
    "SemanticQueryResult",
]
