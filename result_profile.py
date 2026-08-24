from __future__ import annotations

import json
import math
import re
from decimal import Decimal, InvalidOperation
from typing import Any

from .models import SemanticColumn, SemanticQueryResult, is_internal_identifier
from .presentation_models import ColumnProfile, ColumnRole, ResultProfile, ResultShape

_LATITUDE_NAMES = {"lat", "latitude", "纬度"}
_LONGITUDE_NAMES = {"lng", "lon", "long", "longitude", "经度"}
_STATUS_MARKERS = ("status", "state", "severity", "level", "状态", "等级", "级别")
_TIME_MARKERS = ("time", "date", "day", "month", "year", "timestamp", "时间", "日期", "月份", "年份")


def _short_names(column: SemanticColumn) -> set[str]:
    short_name = column.name.rsplit(".", 1)[-1].lower()
    tokens = {item for item in re.split(r"[^a-z0-9\u4e00-\u9fff]+", short_name) if item}
    tokens.add(short_name)
    tokens.add(column.title.strip().lower())
    return tokens


def _contains_marker(column: SemanticColumn, markers: tuple[str, ...]) -> bool:
    text = f"{column.name.rsplit('.', 1)[-1]} {column.title}".lower()
    return any(marker in text for marker in markers)


def _role(column: SemanticColumn) -> ColumnRole:
    names = _short_names(column)
    if names & _LATITUDE_NAMES:
        return ColumnRole.LATITUDE
    if names & _LONGITUDE_NAMES:
        return ColumnRole.LONGITUDE
    if column.member_type == "time_dimension" or column.data_type.lower() in {
        "date",
        "datetime",
        "time",
        "timestamp",
    }:
        return ColumnRole.TIME
    if column.member_type == "measure":
        return ColumnRole.MEASURE
    if _contains_marker(column, _TIME_MARKERS):
        return ColumnRole.TIME
    if _contains_marker(column, _STATUS_MARKERS):
        return ColumnRole.STATUS
    if is_internal_identifier(column.name):
        return ColumnRole.IDENTIFIER
    if column.member_type == "dimension":
        return ColumnRole.CATEGORY
    if column.data_type.lower() in {"number", "integer", "float", "decimal"}:
        return ColumnRole.MEASURE
    if column.data_type.lower() in {"boolean", "string"}:
        return ColumnRole.CATEGORY
    return ColumnRole.UNKNOWN


def _distinct_key(value: Any) -> str:
    if isinstance(value, (dict, list, tuple, set)):
        try:
            return json.dumps(value, ensure_ascii=False, sort_keys=True, default=str)
        except TypeError:
            return str(value)
    return str(value)


def _finite_number(value: Any) -> float | None:
    if value is None or isinstance(value, bool):
        return None
    try:
        converted = float(Decimal(str(value)))
    except (InvalidOperation, ValueError, TypeError, OverflowError):
        return None
    return converted if math.isfinite(converted) else None


def _profile_column(
    column: SemanticColumn,
    rows: list[dict[str, Any]],
) -> ColumnProfile:
    values = [row.get(column.name) for row in rows]
    actual = [value for value in values if value is not None]
    numeric = [value for value in (_finite_number(item) for item in actual) if value is not None]
    return ColumnProfile(
        name=column.name,
        title=column.title,
        data_type=column.data_type,
        role=_role(column),
        member_type=column.member_type,
        null_count=len(values) - len(actual),
        distinct_count=len({_distinct_key(value) for value in actual}),
        minimum=min(numeric) if numeric else None,
        maximum=max(numeric) if numeric else None,
    )


def build_result_profile(result: SemanticQueryResult) -> ResultProfile:
    columns = [_profile_column(column, result.rows) for column in result.columns]
    by_role = {
        role: [column.name for column in columns if column.role == role]
        for role in ColumnRole
    }
    measures = by_role[ColumnRole.MEASURE]
    times = by_role[ColumnRole.TIME]
    categories = by_role[ColumnRole.CATEGORY]
    statuses = by_role[ColumnRole.STATUS]
    latitude = next(iter(by_role[ColumnRole.LATITUDE]), None)
    longitude = next(iter(by_role[ColumnRole.LONGITUDE]), None)
    if not result.rows:
        shape = ResultShape.EMPTY
    elif len(result.rows) == 1 and measures:
        shape = ResultShape.SCALAR
    elif latitude and longitude:
        shape = ResultShape.GEOSPATIAL
    elif statuses:
        shape = ResultShape.STATUS
    elif times and measures:
        shape = ResultShape.TEMPORAL
    elif categories and measures:
        shape = ResultShape.CATEGORICAL
    elif measures:
        shape = ResultShape.NUMERIC
    else:
        shape = ResultShape.TABULAR
    return ResultProfile(
        row_count=len(result.rows),
        column_count=len(result.columns),
        shape=shape,
        columns=columns,
        measure_fields=measures,
        time_fields=times,
        category_fields=categories,
        status_fields=statuses,
        identifier_fields=by_role[ColumnRole.IDENTIFIER],
        latitude_field=latitude,
        longitude_field=longitude,
        primary_measure=next(iter(measures), None),
        primary_time=next(iter(times), None),
        primary_category=next(iter(categories), None),
    )
