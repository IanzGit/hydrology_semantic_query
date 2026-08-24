from __future__ import annotations

from enum import Enum
from typing import Any, Literal

from pydantic import BaseModel, Field


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
    AREA = "AREA"
    BAR = "BAR"
    HEATMAP = "HEATMAP"
    HISTOGRAM = "HISTOGRAM"
    LINE = "LINE"
    PIE = "PIE"
    SCATTER = "SCATTER"


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
