from __future__ import annotations

import json
import math
import re
from collections import Counter, defaultdict
from datetime import UTC, date, datetime, time
from decimal import Decimal, InvalidOperation
from enum import Enum
from itertools import combinations
from statistics import fmean, median
from typing import Any

from .models import (
    ColumnProfile,
    ColumnRole,
    ReportAnalysis,
    ReportFact,
    ReportFactCategory,
    ReportLimitation,
    ResultProfile,
    ResultShape,
    SemanticColumn,
    SemanticQueryResult,
)

_LATITUDE_NAMES = {"lat", "latitude", "纬度"}
_LONGITUDE_NAMES = {"lng", "lon", "long", "longitude", "经度"}
_STATUS_MARKERS = ("status", "state", "severity", "level", "状态", "等级", "级别")
_TIME_MARKERS = ("time", "date", "day", "month", "year", "timestamp", "时间", "日期", "月份", "年份")
_THRESHOLD_MARKERS = ("threshold", "limit", "阈值", "门限")
_UPPER_MARKERS = ("upper", "high", "maximum", "max_", "_max", "上限", "最大")
_LOWER_MARKERS = ("lower", "low", "minimum", "min_", "_min", "下限", "最小")
_UNIT_MARKERS = ("unit", "单位")


def is_internal_identifier(name: str) -> bool:
    short_name = name.partition(".")[2] or name
    return (
        short_name == "id"
        or short_name == "relation_key"
        or short_name.endswith("_id")
        or short_name.endswith("_ids")
    )


def finite_number(value: Any) -> float | None:
    if value is None or isinstance(value, bool):
        return None
    try:
        converted = float(Decimal(str(value)))
    except (InvalidOperation, ValueError, TypeError, OverflowError):
        return None
    return converted if math.isfinite(converted) else None


def cell_value(value: Any) -> Any:
    if value is None:
        return None
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, bool | int):
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, Decimal):
        converted = float(value)
        return converted if math.isfinite(converted) else None
    if isinstance(value, datetime | date):
        return value.isoformat()
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    if isinstance(value, dict):
        return {str(key): cell_value(item) for key, item in value.items()}
    if isinstance(value, list | tuple | set):
        return [cell_value(item) for item in value]
    if isinstance(value, str | bool | int):
        return value
    return str(value)


def display_value(value: Any) -> str:
    normalized = cell_value(value)
    if normalized is None:
        return "--"
    if isinstance(normalized, dict | list):
        return json.dumps(normalized, ensure_ascii=False, sort_keys=True)
    return str(normalized)


def time_key(value: Any) -> tuple[int, float | str]:
    if isinstance(value, datetime):
        parsed = value
    elif isinstance(value, date):
        parsed = datetime.combine(value, time.min)
    else:
        text = str(value or "")
        try:
            parsed = datetime.fromisoformat(text.replace("Z", "+00:00"))
        except ValueError:
            return 0, text
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=UTC)
    try:
        return 1, parsed.timestamp()
    except (OverflowError, OSError, ValueError):
        return 0, parsed.isoformat()


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
    if column.member_type == "time_dimension" or column.data_type.lower() in {"date", "datetime", "time", "timestamp"}:
        return ColumnRole.TIME
    if _contains_marker(column, _TIME_MARKERS):
        return ColumnRole.TIME
    if is_internal_identifier(column.name):
        return ColumnRole.IDENTIFIER
    if column.member_type == "measure" or column.data_type.lower() in {"number", "integer", "float", "decimal"}:
        return ColumnRole.MEASURE
    if _contains_marker(column, _STATUS_MARKERS):
        return ColumnRole.STATUS
    if column.member_type == "dimension" or column.data_type.lower() in {"boolean", "string"}:
        return ColumnRole.CATEGORY
    return ColumnRole.UNKNOWN


def _distinct_key(value: Any) -> str:
    if isinstance(value, (dict, list, tuple, set)):
        return json.dumps(cell_value(value), ensure_ascii=False, sort_keys=True, default=str)
    return str(value)


def _profile_column(column: SemanticColumn, rows: list[dict[str, Any]]) -> ColumnProfile:
    values = [row.get(column.name) for row in rows]
    actual = [value for value in values if value is not None]
    numeric = [number for value in actual if (number := finite_number(value)) is not None]
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
    by_role = {role: [column.name for column in columns if column.role == role] for role in ColumnRole}
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


def _number_text(value: float) -> str:
    if value == 0:
        return "0"
    if abs(value) >= 10000 or abs(value) < 0.001:
        return f"{value:.6g}"
    return f"{value:.4f}".rstrip("0").rstrip(".")


class _FactBuilder:
    def __init__(self) -> None:
        self.facts: list[ReportFact] = []
        self.counters: Counter[ReportFactCategory] = Counter()

    def add(
        self,
        category: ReportFactCategory,
        title: str,
        display_text: str,
        value: Any = None,
        unit: str | None = None,
        evidence_fields: list[str] | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        self.counters[category] += 1
        self.facts.append(ReportFact(
            fact_id=f"{category.value}-{self.counters[category]:03d}",
            category=category,
            title=title,
            display_text=display_text,
            value=cell_value(value),
            unit=unit,
            evidence_fields=evidence_fields or [],
            metadata=metadata or {},
        ))


def _column_map(result: SemanticQueryResult) -> dict[str, SemanticColumn]:
    return {column.name: column for column in result.columns}


def _unit_for_field(result: SemanticQueryResult, field: str, measure_fields: list[str]) -> str | None:
    unit_columns = [column for column in result.columns if _contains_marker(column, _UNIT_MARKERS)]
    short = field.rsplit(".", 1)[-1].lower()
    prefix = re.sub(r"(?:_?value|_?count|_?amount|_?level)$", "", short)
    candidates = [column for column in unit_columns if prefix and prefix in column.name.rsplit(".", 1)[-1].lower()]
    if not candidates and len(measure_fields) == 1 and len(unit_columns) == 1:
        candidates = unit_columns
    if len(candidates) != 1:
        return None
    values = {display_value(row.get(candidates[0].name)) for row in result.rows if row.get(candidates[0].name) not in (None, "")}
    return next(iter(values)) if len(values) == 1 else None


def _add_scope_facts(builder: _FactBuilder, result: SemanticQueryResult) -> None:
    query = result.semantic_query
    models = list(result.selected_models or (query.models if query else []))
    builder.add(ReportFactCategory.SCOPE, "查询模型", f"查询模型：{'、'.join(models) if models else '未提供'}。", models)
    filters = [item.to_wire() for item in query.filters] if query else []
    builder.add(ReportFactCategory.SCOPE, "过滤条件", f"过滤条件：{display_value(filters) if filters else '无显式过滤条件'}。", filters, evidence_fields=[item.member for item in query.filters if item.member] if query else [])
    time_ranges = [item.to_wire() for item in query.time_dimensions] if query else []
    builder.add(ReportFactCategory.SCOPE, "请求时间范围", f"请求时间范围：{display_value(time_ranges) if time_ranges else '未显式指定'}。", time_ranges, evidence_fields=[item.dimension for item in query.time_dimensions] if query else [])
    builder.add(ReportFactCategory.SCOPE, "返回规模", f"语义查询共返回 {len(result.rows)} 行、{len(result.columns)} 个字段。", {"rows": len(result.rows), "columns": len(result.columns)}, evidence_fields=[column.name for column in result.columns])
    field_titles = [f"{column.title}({column.name})" for column in result.columns]
    builder.add(ReportFactCategory.SCOPE, "结果字段", f"结果字段：{'、'.join(field_titles) if field_titles else '无'}。", field_titles, evidence_fields=[column.name for column in result.columns])
    if result.warnings:
        builder.add(ReportFactCategory.SCOPE, "查询警告", f"查询阶段记录 {len(result.warnings)} 条警告。", result.warnings)
    if result.query_history:
        history = [
            {
                "query_number": item.query_number,
                "query_goal": item.query_goal,
                "row_count": item.row_count,
                "models": item.selected_models or item.semantic_query.models,
            }
            for item in result.query_history
        ]
        goals = "；".join(f"第 {item['query_number']} 轮“{item['query_goal']}”返回 {item['row_count']} 行" for item in history)
        builder.add(ReportFactCategory.SCOPE, "语义查询轮次", f"语义查询阶段共完成 {len(history)} 轮业务查询：{goals}。", history)


def _add_quality_facts(builder: _FactBuilder, result: SemanticQueryResult, profile: ResultProfile) -> int:
    total_cells = len(result.rows) * len(result.columns)
    missing_cells = sum(column.null_count for column in profile.columns)
    coverage = ((total_cells - missing_cells) / total_cells * 100) if total_cells else 0.0
    builder.add(ReportFactCategory.QUALITY, "整体完整性", f"全部 {total_cells} 个数据单元中，空值 {missing_cells} 个，整体覆盖率为 {_number_text(coverage)}%。", {"total_cells": total_cells, "missing_cells": missing_cells, "coverage_percent": coverage}, evidence_fields=[column.name for column in result.columns])
    non_finite = 0
    for column in profile.columns:
        field_values = [row.get(column.name) for row in result.rows]
        field_coverage = ((len(field_values) - column.null_count) / len(field_values) * 100) if field_values else 0.0
        builder.add(ReportFactCategory.QUALITY, f"{column.title}覆盖率", f"字段“{column.title}”非空 {len(field_values) - column.null_count} 行，覆盖率 {_number_text(field_coverage)}%。", {"non_null": len(field_values) - column.null_count, "coverage_percent": field_coverage}, evidence_fields=[column.name])
        if column.role == ColumnRole.MEASURE:
            non_finite += sum(value is not None and finite_number(value) is None for value in field_values)
    builder.add(ReportFactCategory.QUALITY, "非有限数值", f"数值字段中共检出 {non_finite} 个无法用于计算的非有限值。", non_finite, evidence_fields=profile.measure_fields)
    return non_finite


def _add_result_time_scope(builder: _FactBuilder, result: SemanticQueryResult, profile: ResultProfile) -> None:
    columns = _column_map(result)
    for field in profile.time_fields:
        values = [row.get(field) for row in result.rows if row.get(field) is not None]
        if not values:
            continue
        ordered = sorted(values, key=time_key)
        builder.add(ReportFactCategory.SCOPE, f"{columns[field].title}结果时间范围", f"结果字段“{columns[field].title}”覆盖的时间范围为 {display_value(ordered[0])} 至 {display_value(ordered[-1])}。", {"start": cell_value(ordered[0]), "end": cell_value(ordered[-1])}, evidence_fields=[field])


def _metric_values(result: SemanticQueryResult, field: str) -> list[tuple[int, float]]:
    return [(index, number) for index, row in enumerate(result.rows) if (number := finite_number(row.get(field))) is not None]


def _add_metric_facts(builder: _FactBuilder, result: SemanticQueryResult, profile: ResultProfile) -> None:
    columns = _column_map(result)
    for field in profile.measure_fields:
        values = _metric_values(result, field)
        if not values:
            continue
        numbers = [value for _, value in values]
        column = columns[field]
        unit = _unit_for_field(result, field, profile.measure_fields)
        suffix = unit or ""
        stats = {
            "count": len(numbers),
            "mean": fmean(numbers),
            "median": median(numbers),
            "minimum": min(numbers),
            "maximum": max(numbers),
        }
        builder.add(ReportFactCategory.METRIC, f"{column.title}核心统计", f"“{column.title}”有效值 {len(numbers)} 个，均值 {_number_text(stats['mean'])}{suffix}，中位数 {_number_text(stats['median'])}{suffix}，最小值 {_number_text(stats['minimum'])}{suffix}，最大值 {_number_text(stats['maximum'])}{suffix}。", stats, unit=unit, evidence_fields=[field], metadata={"metric": field})
        if len(result.rows) == 1:
            builder.add(ReportFactCategory.METRIC, f"{column.title}单值", f"“{column.title}”当前值为 {_number_text(numbers[0])}{suffix}。", numbers[0], unit=unit, evidence_fields=[field], metadata={"metric": field})
        if profile.primary_time:
            timed = [(time_key(result.rows[index].get(profile.primary_time)), index, value) for index, value in values if result.rows[index].get(profile.primary_time) is not None]
            if timed:
                _, index, latest = max(timed, key=lambda item: item[0])
                timestamp = cell_value(result.rows[index].get(profile.primary_time))
                builder.add(ReportFactCategory.METRIC, f"{column.title}最新值", f"“{column.title}”最新有效值为 {_number_text(latest)}{suffix}，时间为 {display_value(timestamp)}。", {"latest": latest, "time": timestamp}, unit=unit, evidence_fields=[field, profile.primary_time], metadata={"metric": field})


def _add_one_trend(builder: _FactBuilder, result: SemanticQueryResult, profile: ResultProfile, field: str, rows: list[dict[str, Any]], series: Any = None) -> None:
    time_field = profile.primary_time
    if not time_field:
        return
    points = [(time_key(row.get(time_field)), row.get(time_field), number) for row in rows if row.get(time_field) is not None and (number := finite_number(row.get(field))) is not None]
    if len(points) < 2:
        return
    points.sort(key=lambda item: item[0])
    start_time, start = points[0][1], points[0][2]
    end_time, end = points[-1][1], points[-1][2]
    absolute_change = end - start
    change_rate = absolute_change / abs(start) * 100 if start != 0 else None
    peak = max(points, key=lambda item: item[2])
    trough = min(points, key=lambda item: item[2])
    columns = _column_map(result)
    unit = _unit_for_field(result, field, profile.measure_fields)
    suffix = unit or ""
    label = f"“{display_value(series)}”的" if series is not None else ""
    rate_text = f"，变化率 {_number_text(change_rate)}%" if change_rate is not None else "；起点为零，不计算变化率"
    builder.add(ReportFactCategory.TREND, f"{columns[field].title}趋势", f"{label}“{columns[field].title}”从 {display_value(start_time)} 的 {_number_text(start)}{suffix} 变化到 {display_value(end_time)} 的 {_number_text(end)}{suffix}，绝对变化 {_number_text(absolute_change)}{suffix}{rate_text}；峰值 {_number_text(peak[2])}{suffix}出现在 {display_value(peak[1])}，谷值 {_number_text(trough[2])}{suffix}出现在 {display_value(trough[1])}。", {"start": start, "end": end, "absolute_change": absolute_change, "change_rate_percent": change_rate, "peak": peak[2], "peak_time": cell_value(peak[1]), "trough": trough[2], "trough_time": cell_value(trough[1]), "series": cell_value(series)}, unit=unit, evidence_fields=[time_field, field] + ([profile.primary_category] if series is not None and profile.primary_category else []), metadata={"metric": field, "series": cell_value(series)})


def _add_trend_facts(builder: _FactBuilder, result: SemanticQueryResult, profile: ResultProfile) -> bool:
    if not profile.primary_time or not profile.measure_fields:
        return False
    before = len(builder.facts)
    category = profile.primary_category
    if category and next((column.distinct_count for column in profile.columns if column.name == category), 0) > 1:
        grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
        labels: dict[str, Any] = {}
        for row in result.rows:
            key = _distinct_key(row.get(category))
            grouped[key].append(row)
            labels[key] = row.get(category)
        for key in sorted(grouped):
            for field in profile.measure_fields:
                _add_one_trend(builder, result, profile, field, grouped[key], labels[key])
    else:
        for field in profile.measure_fields:
            _add_one_trend(builder, result, profile, field, result.rows)
    return len(builder.facts) > before


def _add_distribution_facts(builder: _FactBuilder, result: SemanticQueryResult, profile: ResultProfile) -> None:
    columns = _column_map(result)
    for field in profile.status_fields:
        counts = Counter(display_value(row.get(field)) for row in result.rows if row.get(field) is not None)
        if counts:
            builder.add(ReportFactCategory.STATUS, f"{columns[field].title}分布", f"“{columns[field].title}”共 {len(counts)} 类：" + "；".join(f"{label} {count} 条（{_number_text(count / sum(counts.values()) * 100)}%）" for label, count in counts.most_common()) + "。", dict(counts), evidence_fields=[field])
    category = profile.primary_category
    if category:
        counts = Counter(display_value(row.get(category)) for row in result.rows if row.get(category) is not None)
        if counts:
            builder.add(ReportFactCategory.DISTRIBUTION, f"{columns[category].title}分布", f"“{columns[category].title}”共 {len(counts)} 类，记录数最多的类别为“{counts.most_common(1)[0][0]}”，共 {counts.most_common(1)[0][1]} 条，占 {_number_text(counts.most_common(1)[0][1] / sum(counts.values()) * 100)}%。", dict(counts), evidence_fields=[category])
    if category and profile.primary_measure and not profile.primary_time:
        measure = profile.primary_measure
        groups: dict[str, list[float]] = defaultdict(list)
        for row in result.rows:
            number = finite_number(row.get(measure))
            if number is not None and row.get(category) is not None:
                groups[display_value(row.get(category))].append(number)
        means = {label: fmean(values) for label, values in groups.items() if values}
        if len(means) >= 2:
            ordered = sorted(means.items(), key=lambda item: (item[1], item[0]))
            bottom, top = ordered[0], ordered[-1]
            gap = top[1] - bottom[1]
            unit = _unit_for_field(result, measure, profile.measure_fields)
            suffix = unit or ""
            builder.add(ReportFactCategory.DISTRIBUTION, f"{columns[measure].title}组间对比", f"按“{columns[category].title}”分组后，“{columns[measure].title}”均值最高为“{top[0]}”的 {_number_text(top[1])}{suffix}，最低为“{bottom[0]}”的 {_number_text(bottom[1])}{suffix}，组间差异 {_number_text(gap)}{suffix}。", {"top": {"category": top[0], "value": top[1]}, "bottom": {"category": bottom[0], "value": bottom[1]}, "gap": gap, "ranking": dict(reversed(ordered))}, unit=unit, evidence_fields=[category, measure])


def _threshold_columns(result: SemanticQueryResult) -> list[SemanticColumn]:
    return [column for column in result.columns if _contains_marker(column, _THRESHOLD_MARKERS)]


def _add_threshold_facts(builder: _FactBuilder, result: SemanticQueryResult, profile: ResultProfile) -> bool:
    thresholds = _threshold_columns(result)
    value_fields = [field for field in profile.measure_fields if field not in {column.name for column in thresholds}]
    if not thresholds or not value_fields:
        return False
    columns = _column_map(result)
    value_field = next((field for field in value_fields if any(marker in f"{field} {columns[field].title}".lower() for marker in ("current", "value", "当前", "监测值"))), value_fields[0])
    added = False
    for threshold in thresholds:
        text = f"{threshold.name} {threshold.title}".lower()
        direction = "lower" if any(marker in text for marker in _LOWER_MARKERS) else "upper"
        comparable = []
        exceeded = []
        for index, row in enumerate(result.rows):
            value = finite_number(row.get(value_field))
            limit = finite_number(row.get(threshold.name))
            if value is None or limit is None:
                continue
            comparable.append(index)
            if (direction == "upper" and value > limit) or (direction == "lower" and value < limit):
                exceeded.append(index)
        if not comparable:
            continue
        relation = "低于下限" if direction == "lower" else "高于上限"
        builder.add(ReportFactCategory.THRESHOLD, f"{threshold.title}阈值对比", f"“{columns[value_field].title}”与明确字段“{threshold.title}”可比的记录有 {len(comparable)} 条，其中 {len(exceeded)} 条{relation}，占 {_number_text(len(exceeded) / len(comparable) * 100)}%。", {"comparable_count": len(comparable), "exceeded_count": len(exceeded), "exceeded_percent": len(exceeded) / len(comparable) * 100, "direction": direction, "row_indexes": exceeded}, evidence_fields=[value_field, threshold.name])
        added = True
    return added


def _quantile(values: list[float], probability: float) -> float:
    position = (len(values) - 1) * probability
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return values[lower]
    return values[lower] + (values[upper] - values[lower]) * (position - lower)


def _add_anomaly_fact(builder: _FactBuilder, result: SemanticQueryResult, profile: ResultProfile) -> bool:
    threshold_names = {column.name for column in _threshold_columns(result)}
    field = next((name for name in profile.measure_fields if name not in threshold_names), None)
    if not field:
        return False
    indexed_values = _metric_values(result, field)
    values = sorted(number for _, number in indexed_values)
    if len(values) < 8:
        return False
    q1 = _quantile(values, 0.25)
    q3 = _quantile(values, 0.75)
    iqr = q3 - q1
    lower = q1 - 1.5 * iqr
    upper = q3 + 1.5 * iqr
    outlier_pairs = [
        (index, value)
        for index, value in indexed_values
        if value < lower or value > upper
    ]
    outliers = sorted(value for _, value in outlier_pairs)
    title = _column_map(result)[field].title
    builder.add(
        ReportFactCategory.ANOMALY,
        f"{title}统计异常",
        (
            f"基于 {len(values)} 个有效样本的 IQR 统计规则，"
            f"“{title}”检出 {len(outliers)} 个统计异常值；"
            "该结果不等同于水害风险。"
        ),
        {
            "sample_count": len(values),
            "q1": q1,
            "q3": q3,
            "lower_bound": lower,
            "upper_bound": upper,
            "outliers": outliers,
            "row_indexes": [index for index, _ in outlier_pairs],
        },
        evidence_fields=[field],
    )
    return True


def _pearson(pairs: list[tuple[float, float]]) -> float | None:
    xs = [pair[0] for pair in pairs]
    ys = [pair[1] for pair in pairs]
    mean_x = fmean(xs)
    mean_y = fmean(ys)
    numerator = sum((x - mean_x) * (y - mean_y) for x, y in pairs)
    denominator = math.sqrt(sum((x - mean_x) ** 2 for x in xs) * sum((y - mean_y) ** 2 for y in ys))
    return numerator / denominator if denominator else None


def _add_correlation_facts(builder: _FactBuilder, result: SemanticQueryResult, profile: ResultProfile) -> bool:
    threshold_names = {column.name for column in _threshold_columns(result)}
    fields = [field for field in profile.measure_fields if field not in threshold_names]
    columns = _column_map(result)
    added = False
    for left, right in combinations(fields, 2):
        pairs = [(x, y) for row in result.rows if (x := finite_number(row.get(left))) is not None and (y := finite_number(row.get(right))) is not None]
        if len(pairs) < 8:
            continue
        coefficient = _pearson(pairs)
        if coefficient is None:
            continue
        builder.add(ReportFactCategory.CORRELATION, f"{columns[left].title}与{columns[right].title}相关性", f"在 {len(pairs)} 个同行有效配对样本上，“{columns[left].title}”与“{columns[right].title}”的 Pearson 相关系数为 {_number_text(coefficient)}；相关性不代表因果关系。", {"sample_count": len(pairs), "coefficient": coefficient}, evidence_fields=[left, right])
        added = True
    return added


def analyze_result(result: SemanticQueryResult) -> ReportAnalysis:
    profile = build_result_profile(result)
    builder = _FactBuilder()
    limitations: list[ReportLimitation] = []
    _add_scope_facts(builder, result)
    _add_result_time_scope(builder, result, profile)
    _add_quality_facts(builder, result, profile)
    _add_metric_facts(builder, result, profile)
    has_trend = _add_trend_facts(builder, result, profile)
    _add_distribution_facts(builder, result, profile)
    has_threshold = _add_threshold_facts(builder, result, profile)
    has_anomaly = _add_anomaly_fact(builder, result, profile)
    has_correlation = _add_correlation_facts(builder, result, profile)
    if not profile.primary_time:
        limitations.append(ReportLimitation(code="missing_time", message="结果中没有时间字段，无法验证趋势。"))
    elif not has_trend:
        limitations.append(ReportLimitation(code="insufficient_trend", message="时间字段存在，但同粒度有效数值点不足 2 个，无法验证趋势。"))
    if not profile.measure_fields:
        limitations.append(ReportLimitation(code="missing_numeric", message="结果中没有可用数值字段，无法计算核心指标、趋势、统计异常或相关性。"))
    else:
        for field in profile.measure_fields:
            if _unit_for_field(result, field, profile.measure_fields) is None:
                title = _column_map(result)[field].title
                limitations.append(ReportLimitation(code=f"missing_unit_{field}", message=f"结果未提供与“{title}”明确对应的单位字段，报告不推断单位。"))
    if not has_threshold:
        limitations.append(ReportLimitation(code="missing_threshold", message="结果未同时提供可比的监测值和明确阈值字段，不判定超限。"))
    if not has_anomaly:
        limitations.append(ReportLimitation(code="insufficient_anomaly_sample", message="单一数值指标的有效样本少于 8 个，不进行统计异常判定。"))
    if not has_correlation:
        limitations.append(ReportLimitation(code="insufficient_correlation_sample", message="结果不具备至少两个同粒度数值字段且有效配对样本不少于 8 个的条件，不计算相关性。"))
    for index, warning in enumerate(result.warnings, 1):
        limitations.append(ReportLimitation(code=f"query_warning_{index}", message=f"查询阶段警告：{warning}"))
    return ReportAnalysis(profile=profile, facts=builder.facts, limitations=limitations)


__all__ = [
    "analyze_result",
    "build_result_profile",
    "cell_value",
    "display_value",
    "finite_number",
    "is_internal_identifier",
    "time_key",
]
