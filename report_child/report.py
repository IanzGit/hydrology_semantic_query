from __future__ import annotations

import json
import math
from collections import Counter, defaultdict
from collections.abc import Sequence
from datetime import UTC, date, datetime, time
from decimal import Decimal, InvalidOperation
from enum import Enum
from typing import Any

from pydantic import ValidationError

from app.agents.streaming import (
    WorkflowOutputType,
    build_structured_output,
    table_output,
)

from .models import (
    ChartAggregation,
    ChartEvidence,
    ChartFilter,
    ChartFilterOperator,
    ChartPlanningResponse,
    ChartType,
    DataColumnProfile,
    DynamicChartPlan,
    HighFrequencyValue,
    RenderedChart,
    ReportTask,
    SemanticColumn,
    SemanticQueryResult,
    TaskChartPlans,
    TaskDataProfile,
)


def _cell_value(value: Any) -> Any:
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
        return {str(key): _cell_value(item) for key, item in value.items()}
    if isinstance(value, list | tuple | set):
        return [_cell_value(item) for item in value]
    if isinstance(value, str):
        return value
    return str(value)


def _display_value(value: Any) -> str:
    normalized = _cell_value(value)
    if normalized is None:
        return "--"
    if isinstance(normalized, dict | list):
        return json.dumps(normalized, ensure_ascii=False)
    return str(normalized)


def _finite_number(value: Any) -> float | None:
    if value is None or isinstance(value, bool):
        return None
    try:
        converted = float(Decimal(str(value)))
    except (InvalidOperation, ValueError, TypeError, OverflowError):
        return None
    return converted if math.isfinite(converted) else None


def _unique(values: list[Any]) -> list[Any]:
    result: list[Any] = []
    seen: set[str] = set()
    for value in values:
        key = json.dumps(_cell_value(value), ensure_ascii=False, sort_keys=True, default=str)
        if key not in seen:
            seen.add(key)
            result.append(value)
    return result


def _parsed_time(value: Any) -> datetime | None:
    if isinstance(value, datetime):
        parsed = value
    elif isinstance(value, date):
        parsed = datetime.combine(value, time.min)
    elif isinstance(value, str) and value.strip():
        try:
            parsed = datetime.fromisoformat(value.strip().replace("Z", "+00:00"))
        except ValueError:
            return None
    else:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=UTC)
    return parsed


def _value_key(value: Any) -> str:
    return json.dumps(
        _cell_value(value),
        ensure_ascii=False,
        sort_keys=True,
        default=str,
    )


def _uniform_indices(length: int, count: int) -> list[int]:
    if length <= 0 or count <= 0:
        return []
    if length <= count:
        return list(range(length))
    if count == 1:
        return [length - 1]
    return sorted({round(index * (length - 1) / (count - 1)) for index in range(count)})


def _profile_samples(rows: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    first_rows = [
        {str(key): _cell_value(value) for key, value in row.items()}
        for row in rows[:10]
    ]
    remaining = rows[10:]
    sampled_rows: list[dict[str, Any]] = []
    seen: set[str] = set()
    for index in _uniform_indices(len(remaining), min(40, len(remaining))):
        normalized = {
            str(key): _cell_value(value)
            for key, value in remaining[index].items()
        }
        key = _value_key(normalized)
        if key not in seen:
            seen.add(key)
            sampled_rows.append(normalized)
    return first_rows, sampled_rows


def _data_column_profile(
    column: SemanticColumn,
    rows: list[dict[str, Any]],
) -> DataColumnProfile:
    values = [row.get(column.name) for row in rows]
    non_null = [value for value in values if value is not None]
    numeric = [
        number
        for value in non_null
        if (number := _finite_number(value)) is not None
    ]
    parsed_times = [
        parsed
        for value in non_null
        if (parsed := _parsed_time(value)) is not None
    ]
    numeric_ratio = len(numeric) / len(non_null) if non_null else 0.0
    time_ratio = len(parsed_times) / len(non_null) if non_null else 0.0
    declared = column.data_type.lower()
    if non_null and numeric_ratio >= 0.8:
        inferred_type = "numeric"
    elif non_null and time_ratio >= 0.8:
        inferred_type = "time"
    elif non_null:
        inferred_type = "category"
    else:
        inferred_type = "unknown"
    counts: dict[str, list[Any]] = {}
    for position, value in enumerate(non_null):
        key = _value_key(value)
        if key not in counts:
            counts[key] = [value, 0, position]
        counts[key][1] += 1
    top_values = [
        HighFrequencyValue(
            value=_cell_value(item[0]),
            count=item[1],
            ratio=round(item[1] / len(non_null), 6),
        )
        for item in sorted(counts.values(), key=lambda item: (-item[1], item[2]))[:10]
    ] if non_null else []
    return DataColumnProfile(
        name=column.name,
        title=column.title,
        declared_type=declared,
        inferred_type=inferred_type,
        row_count=len(values),
        null_count=len(values) - len(non_null),
        null_rate=round((len(values) - len(non_null)) / len(values), 6) if values else 0.0,
        unique_count=len(counts),
        finite_numeric_ratio=round(numeric_ratio, 6),
        numeric_min=min(numeric) if numeric else None,
        numeric_max=max(numeric) if numeric else None,
        time_earliest=min(parsed_times).isoformat() if parsed_times else None,
        time_latest=max(parsed_times).isoformat() if parsed_times else None,
        top_values=top_values,
    )


def has_basic_chart_opportunity(columns: Sequence[DataColumnProfile]) -> bool:
    numeric = [
        column
        for column in columns
        if column.inferred_type == "numeric"
        and column.finite_numeric_ratio >= 0.8
    ]
    times = [
        column
        for column in columns
        if column.inferred_type == "time" and column.unique_count >= 2
    ]
    categories = [
        column
        for column in columns
        if column.inferred_type == "category" and column.unique_count >= 2
    ]
    usable_numeric = any(
        column.unique_count >= 1
        for column in numeric
    )
    return bool(
        (usable_numeric and times)
        or (usable_numeric and categories)
        or categories
    )


def build_task_data_profile(
    source_task_id: str,
    objective: str,
    result: SemanticQueryResult,
) -> TaskDataProfile:
    columns = [_data_column_profile(column, result.rows) for column in result.columns]
    first_rows, sampled_rows = _profile_samples(result.rows)
    return TaskDataProfile(
        source_task_id=source_task_id,
        objective=objective,
        row_count=len(result.rows),
        columns=columns,
        first_rows=first_rows,
        sampled_rows=sampled_rows,
        has_chart_opportunity=has_basic_chart_opportunity(columns),
    )


def build_task_data_profiles(
    datasets: Sequence[tuple[str, str, SemanticQueryResult]],
) -> list[TaskDataProfile]:
    return [
        build_task_data_profile(task_id, objective, result)
        for task_id, objective, result in datasets
    ]


class ChartPlanValidationError(ValueError):
    pass


def parse_chart_planning_response(
    raw: str,
) -> tuple[ChartPlanningResponse, list[dict[str, str]]]:
    try:
        payload = json.loads(raw)
    except (TypeError, json.JSONDecodeError) as exc:
        return ChartPlanningResponse(), [{
            "source_task_id": "*",
            "error": f"规划响应不是有效 JSON：{exc}",
        }]
    if not isinstance(payload, dict) or not isinstance(payload.get("tasks"), list):
        return ChartPlanningResponse(), [{
            "source_task_id": "*",
            "error": "规划响应必须包含 tasks 数组",
        }]
    root_extras = set(payload) - {"tasks"}
    errors: list[dict[str, str]] = []
    if root_extras:
        errors.append({
            "source_task_id": "*",
            "error": f"规划响应包含未知字段：{sorted(root_extras)}",
        })
    tasks: list[TaskChartPlans] = []
    for task_payload in payload["tasks"]:
        if not isinstance(task_payload, dict):
            errors.append({"source_task_id": "*", "error": "任务计划必须是对象"})
            continue
        source_task_id = str(task_payload.get("source_task_id") or "").strip()
        if not source_task_id:
            errors.append({"source_task_id": "*", "error": "任务计划缺少 source_task_id"})
            continue
        task_extras = set(task_payload) - {
            "source_task_id",
            "charts",
            "no_chart_reason",
        }
        if task_extras:
            errors.append({
                "source_task_id": source_task_id,
                "error": f"任务计划包含未知字段：{sorted(task_extras)}",
            })
        raw_charts = task_payload.get("charts")
        if not isinstance(raw_charts, list):
            errors.append({"source_task_id": source_task_id, "error": "charts 必须是数组"})
            raw_charts = []
        if len(raw_charts) > 4:
            errors.append({"source_task_id": source_task_id, "error": "每个任务最多规划 4 张图"})
        charts: list[DynamicChartPlan] = []
        for index, raw_chart in enumerate(raw_charts[:4]):
            try:
                charts.append(DynamicChartPlan.model_validate(raw_chart))
            except ValidationError as exc:
                errors.append({
                    "source_task_id": source_task_id,
                    "error": f"第 {index + 1} 张图结构无效：{exc}",
                })
        no_chart_reason = task_payload.get("no_chart_reason")
        if charts:
            no_chart_reason = None
        elif not isinstance(no_chart_reason, str) or not no_chart_reason.strip():
            errors.append({
                "source_task_id": source_task_id,
                "error": "无图表计划时必须提供 no_chart_reason",
            })
            no_chart_reason = "图表计划未通过结构校验"
        tasks.append(TaskChartPlans(
            source_task_id=source_task_id,
            charts=charts,
            no_chart_reason=no_chart_reason,
        ))
    return ChartPlanningResponse(tasks=tasks), errors


def _filter_comparison_mode(values: list[Any], expected: Any) -> str:
    non_null = [value for value in values if value is not None]
    if non_null and sum(_finite_number(value) is not None for value in non_null) / len(non_null) >= 0.8:
        if _finite_number(expected) is None:
            raise ChartPlanValidationError("数值过滤条件必须使用有限数值")
        return "numeric"
    if non_null and sum(_parsed_time(value) is not None for value in non_null) / len(non_null) >= 0.8:
        if _parsed_time(expected) is None:
            raise ChartPlanValidationError("时间过滤条件必须使用可解析时间")
        return "time"
    raise ChartPlanValidationError("比较过滤条件与字段实际值不兼容")


def _equal_value(left: Any, right: Any) -> bool:
    left_number = _finite_number(left)
    right_number = _finite_number(right)
    if left_number is not None and right_number is not None:
        return left_number == right_number
    return _value_key(left) == _value_key(right)


def _equality_mode(values: list[Any], expected_values: list[Any]) -> str:
    non_null = [value for value in values if value is not None]
    if non_null and sum(_finite_number(value) is not None for value in non_null) / len(non_null) >= 0.8:
        if any(_finite_number(value) is None for value in expected_values):
            raise ChartPlanValidationError("等值过滤条件与数值字段不兼容")
        return "numeric"
    if non_null and sum(_parsed_time(value) is not None for value in non_null) / len(non_null) >= 0.8:
        if any(_parsed_time(value) is None for value in expected_values):
            raise ChartPlanValidationError("等值过滤条件与时间字段不兼容")
        return "time"
    return "scalar"


def _mode_equal(left: Any, right: Any, mode: str) -> bool:
    if mode == "numeric":
        return _finite_number(left) == _finite_number(right)
    if mode == "time":
        return _parsed_time(left) == _parsed_time(right)
    return _equal_value(left, right)


def _filter_context(
    chart_filter: ChartFilter,
    values: list[Any],
) -> tuple[str, Any]:
    operator = chart_filter.operator
    expected = chart_filter.value
    if operator == ChartFilterOperator.IS_NULL:
        return "is_null", None
    if operator == ChartFilterOperator.NOT_NULL:
        return "not_null", None
    if operator in {ChartFilterOperator.IN, ChartFilterOperator.NOT_IN}:
        if not isinstance(expected, list) or not expected:
            raise ChartPlanValidationError(f"过滤操作符 {operator.value} 需要非空数组")
        mode = _equality_mode(values, expected)
        return mode, expected
    if operator in {ChartFilterOperator.EQ, ChartFilterOperator.NE}:
        if isinstance(expected, list) or expected is None:
            raise ChartPlanValidationError(f"过滤操作符 {operator.value} 需要单个非空值")
        mode = _equality_mode(values, [expected])
        return mode, expected
    bounds = expected if operator == ChartFilterOperator.BETWEEN else [expected]
    if operator == ChartFilterOperator.BETWEEN and (
        not isinstance(bounds, list) or len(bounds) != 2
    ):
        raise ChartPlanValidationError("between 过滤需要两个边界值")
    if operator != ChartFilterOperator.BETWEEN and (
        isinstance(expected, list) or expected is None
    ):
        raise ChartPlanValidationError(f"过滤操作符 {operator.value} 需要单个边界值")
    assert isinstance(bounds, list)
    mode = _filter_comparison_mode(values, bounds[0])
    if mode == "numeric":
        converted_bounds = [_finite_number(item) for item in bounds]
    else:
        converted_bounds = [_parsed_time(item) for item in bounds]
    if any(item is None for item in converted_bounds):
        raise ChartPlanValidationError("过滤边界值与字段实际值不兼容")
    return mode, converted_bounds


def _filter_matches(
    current: Any,
    chart_filter: ChartFilter,
    context: tuple[str, Any],
) -> bool:
    operator = chart_filter.operator
    mode, expected = context
    if operator == ChartFilterOperator.IS_NULL:
        return current is None
    if operator == ChartFilterOperator.NOT_NULL:
        return current is not None
    if operator in {ChartFilterOperator.IN, ChartFilterOperator.NOT_IN}:
        matched = any(_mode_equal(current, item, mode) for item in expected)
        return matched if operator == ChartFilterOperator.IN else not matched
    if operator in {ChartFilterOperator.EQ, ChartFilterOperator.NE}:
        matched = _mode_equal(current, expected, mode)
        return matched if operator == ChartFilterOperator.EQ else not matched
    converted = _finite_number(current) if mode == "numeric" else _parsed_time(current)
    if converted is None:
        return False
    low = expected[0]
    assert low is not None
    if operator == ChartFilterOperator.GT:
        return converted > low
    if operator == ChartFilterOperator.GTE:
        return converted >= low
    if operator == ChartFilterOperator.LT:
        return converted < low
    if operator == ChartFilterOperator.LTE:
        return converted <= low
    high = expected[1]
    assert high is not None
    return low <= converted <= high


def _apply_chart_filters(
    rows: list[dict[str, Any]],
    filters: list[ChartFilter],
    field_names: set[str],
) -> list[dict[str, Any]]:
    filtered = list(rows)
    for chart_filter in filters:
        if chart_filter.field not in field_names:
            raise ChartPlanValidationError(f"过滤字段不存在：{chart_filter.field}")
        if chart_filter.operator in {
            ChartFilterOperator.IS_NULL,
            ChartFilterOperator.NOT_NULL,
        } and chart_filter.value is not None:
            raise ChartPlanValidationError(f"过滤操作符 {chart_filter.operator.value} 的 value 必须为 null")
        values = [row.get(chart_filter.field) for row in filtered]
        context = _filter_context(chart_filter, values)
        filtered = [
            row
            for row in filtered
            if _filter_matches(row.get(chart_filter.field), chart_filter, context)
        ]
    if not filtered:
        raise ChartPlanValidationError("过滤后没有数据")
    return filtered


def _aggregate_chart_rows(
    plan: DynamicChartPlan,
    rows: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], int, int]:
    axis_rows = [
        row
        for row in rows
        if row.get(plan.x_field) is not None
        and (plan.series_field is None or row.get(plan.series_field) is not None)
    ]
    if not axis_rows:
        raise ChartPlanValidationError("横轴或系列字段没有可用值")
    invalid_numeric_count = 0
    if plan.aggregation != ChartAggregation.COUNT:
        assert plan.value_field is not None
        non_null = [row for row in axis_rows if row.get(plan.value_field) is not None]
        numeric_rows = [
            (row, number)
            for row in non_null
            if (number := _finite_number(row.get(plan.value_field))) is not None
        ]
        if len(numeric_rows) < 2:
            raise ChartPlanValidationError("数值图过滤后至少需要 2 个有限数值")
        if len(numeric_rows) / len(non_null) < 0.8:
            raise ChartPlanValidationError("数值字段有限数值比例低于 80%")
        invalid_numeric_count = len(non_null) - len(numeric_rows)
        value_rows = [(row, number) for row, number in numeric_rows]
    else:
        value_rows = [(row, 1.0) for row in axis_rows]
    groups: dict[tuple[str, str | None], dict[str, Any]] = {}
    for row, number in value_rows:
        x_value = row.get(plan.x_field)
        series_value = row.get(plan.series_field) if plan.series_field else None
        key = (_value_key(x_value), _value_key(series_value) if plan.series_field else None)
        if key not in groups:
            groups[key] = {
                "x": _cell_value(x_value),
                "series": _cell_value(series_value),
                "values": [],
            }
        groups[key]["values"].append(number)
    if plan.aggregation == ChartAggregation.NONE and any(
        len(group["values"]) != 1 for group in groups.values()
    ):
        raise ChartPlanValidationError("aggregation=none 时每个 (x, series) 组合必须唯一")
    aggregated: list[dict[str, Any]] = []
    for group in groups.values():
        values = group.pop("values")
        if plan.aggregation == ChartAggregation.COUNT:
            value: float | int = len(values)
        elif plan.aggregation == ChartAggregation.SUM:
            value = sum(values)
        elif plan.aggregation == ChartAggregation.AVG:
            value = sum(values) / len(values)
        elif plan.aggregation == ChartAggregation.MIN:
            value = min(values)
        elif plan.aggregation == ChartAggregation.MAX:
            value = max(values)
        else:
            value = values[0]
        aggregated.append({**group, "value": value})
    return aggregated, invalid_numeric_count, len(value_rows)


def _x_sort_key(value: Any) -> tuple[int, float | str]:
    parsed = _parsed_time(value)
    if parsed is not None:
        return 0, parsed.timestamp()
    number = _finite_number(value)
    if number is not None:
        return 1, number
    return 2, _display_value(value)


def _ordered_chart_rows(
    rows: list[dict[str, Any]],
    plan: DynamicChartPlan,
) -> list[dict[str, Any]]:
    if plan.chart_type == ChartType.LINE:
        return sorted(rows, key=lambda row: (_x_sort_key(row["x"]), _display_value(row.get("series"))))
    reverse = plan.sort.direction == "desc"
    if plan.sort.by == "value":
        return sorted(
            rows,
            key=lambda row: (row["value"], _x_sort_key(row["x"])),
            reverse=reverse,
        )
    return sorted(rows, key=lambda row: _x_sort_key(row["x"]), reverse=reverse)


def _uniform_rows(rows: list[dict[str, Any]], count: int) -> list[dict[str, Any]]:
    return [rows[index] for index in _uniform_indices(len(rows), count)]


def _line_rows(
    rows: list[dict[str, Any]],
    plan: DynamicChartPlan,
) -> tuple[list[dict[str, Any]], bool]:
    if any(_parsed_time(row["x"]) is None for row in rows):
        raise ChartPlanValidationError("折线图横轴必须全部是可解析时间")
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    series_labels: dict[str, Any] = {}
    for row in rows:
        key = _value_key(row.get("series")) if plan.series_field else "__single__"
        grouped[key].append(row)
        series_labels[key] = row.get("series")
    if len(grouped) > 10:
        raise ChartPlanValidationError("折线图系列数不能超过 10")
    for values in grouped.values():
        distinct_times = {_parsed_time(row["x"]) for row in values}
        if len(distinct_times) < 2:
            raise ChartPlanValidationError("折线图每个序列至少需要 2 个不同时间点")
    cap = min(plan.limit, 1000)
    if len(rows) <= cap:
        return rows, False
    if cap < 2 * len(grouped):
        raise ChartPlanValidationError("折线图 limit 不足以为每个序列保留首尾点")
    allocations = {key: 2 for key in grouped}
    remaining = cap - 2 * len(grouped)
    while remaining:
        candidates = [
            key for key, values in grouped.items()
            if allocations[key] < len(values)
        ]
        if not candidates:
            break
        key = max(candidates, key=lambda item: (len(grouped[item]) - allocations[item], item))
        allocations[key] += 1
        remaining -= 1
    sampled = [
        row
        for key, values in grouped.items()
        for row in _uniform_rows(values, allocations[key])
    ]
    return _ordered_chart_rows(sampled, plan), True


def _category_totals(rows: list[dict[str, Any]]) -> dict[str, float]:
    totals: dict[str, float] = defaultdict(float)
    for row in rows:
        totals[_value_key(row["x"])] += float(row["value"])
    return totals


def _limited_category_rows(
    rows: list[dict[str, Any]],
    plan: DynamicChartPlan,
    maximum_categories: int,
    maximum_series: int | None = None,
) -> tuple[list[dict[str, Any]], bool]:
    totals = _category_totals(rows)
    first_x = {_value_key(row["x"]): row["x"] for row in rows}
    if plan.sort.by == "value":
        category_keys = sorted(
            totals,
            key=lambda key: (totals[key], _x_sort_key(first_x[key])),
            reverse=plan.sort.direction == "desc",
        )
    else:
        category_keys = sorted(
            totals,
            key=lambda key: _x_sort_key(first_x[key]),
            reverse=plan.sort.direction == "desc",
        )
    series_keys: list[str] | None = None
    if maximum_series is not None and plan.series_field:
        series_totals: dict[str, float] = defaultdict(float)
        for row in rows:
            series_totals[_value_key(row["series"])] += float(row["value"])
        series_keys = sorted(series_totals, key=lambda key: (-series_totals[key], key))[
            :min(maximum_series, plan.limit)
        ]
    series_count = len(series_keys) if series_keys is not None else max(
        1,
        len({_value_key(row.get("series")) for row in rows}) if plan.series_field else 1,
    )
    point_category_cap = max(1, min(maximum_categories, plan.limit // series_count))
    kept_categories = set(category_keys[:point_category_cap])
    kept_series = set(series_keys) if series_keys is not None else None
    limited = [
        row
        for row in rows
        if _value_key(row["x"]) in kept_categories
        and (kept_series is None or _value_key(row["series"]) in kept_series)
    ]
    truncated = len(kept_categories) < len(category_keys) or (
        kept_series is not None
        and len(kept_series) < len({_value_key(row["series"]) for row in rows})
    )
    ordering = {key: index for index, key in enumerate(category_keys)}
    limited.sort(key=lambda row: (ordering[_value_key(row["x"])], _display_value(row.get("series"))))
    return limited, truncated


def _chart_series_data(
    rows: list[dict[str, Any]],
    plan: DynamicChartPlan,
    value_title: str,
    unit: str,
) -> list[dict[str, Any]]:
    series_name = "记录数" if plan.aggregation == ChartAggregation.COUNT else value_title
    series_name = f"{series_name}（{unit}）"
    if plan.chart_type == ChartType.PIE:
        return [{
            "name": series_name,
            "data": [
                {"name": _display_value(row["x"]), "value": _cell_value(row["value"])}
                for row in rows
            ],
        }]
    if plan.series_field:
        x_values = _unique([row["x"] for row in rows])
        series_values = _unique([row["series"] for row in rows])
        by_key = {
            (_value_key(row["x"]), _value_key(row["series"])): row["value"]
            for row in rows
        }
        return [
            {
                "name": _display_value(series),
                "data": [
                    {
                        "name": _display_value(x_value),
                        "value": _cell_value(by_key.get((_value_key(x_value), _value_key(series)), 0))
                        if plan.chart_type == ChartType.BAR_STACK
                        else _cell_value(by_key.get((_value_key(x_value), _value_key(series)))),
                    }
                    for x_value in x_values
                    if plan.chart_type == ChartType.BAR_STACK
                    or (_value_key(x_value), _value_key(series)) in by_key
                ],
            }
            for series in series_values
        ]
    return [{
        "name": series_name,
        "data": [
            {"name": _display_value(row["x"]), "value": _cell_value(row["value"])}
            for row in rows
        ],
    }]


def _chart_summary(
    rows: list[dict[str, Any]],
    plan: DynamicChartPlan,
    unit: str,
    invalid_numeric_count: int,
) -> str:
    invalid = f"；剔除 {invalid_numeric_count} 个无效数值" if invalid_numeric_count else ""
    if plan.chart_type == ChartType.PIE:
        total = sum(float(row["value"]) for row in rows)
        maximum = max(rows, key=lambda row: row["value"])
        ratio = float(maximum["value"]) / total * 100
        return f"总量为 {total:g} {unit}，最大类别为{_display_value(maximum['x'])}，占比 {ratio:.2f}%{invalid}。"
    if plan.chart_type == ChartType.LINE:
        groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for row in rows:
            groups[_display_value(row.get("series")) if plan.series_field else "全部数据"].append(row)
        descriptions = []
        for label, values in groups.items():
            ordered = sorted(values, key=lambda row: _parsed_time(row["x"]) or datetime.min.replace(tzinfo=UTC))
            start = ordered[0]
            end = ordered[-1]
            peak = max(ordered, key=lambda row: row["value"])
            trough = min(ordered, key=lambda row: row["value"])
            descriptions.append(
                f"{label}在{_display_value(start['x'])}至{_display_value(end['x'])}从{start['value']:g}变为{end['value']:g}，变化量{end['value'] - start['value']:g}，峰值{peak['value']:g}，谷值{trough['value']:g}"
            )
        return "；".join(descriptions) + f"，单位：{unit}{invalid}。"
    if plan.chart_type == ChartType.BAR_STACK:
        categories: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for row in rows:
            categories[_display_value(row["x"])].append(row)
        descriptions = []
        for category, values in categories.items():
            total = sum(float(row["value"]) for row in values)
            contributor = max(values, key=lambda row: row["value"])
            descriptions.append(
                f"{category}总量{total:g}，最大贡献序列为{_display_value(contributor['series'])}"
            )
        return "；".join(descriptions) + f"，单位：{unit}{invalid}。"
    maximum = max(rows, key=lambda row: row["value"])
    minimum = min(rows, key=lambda row: row["value"])
    maximum_label = _display_value(maximum["x"])
    minimum_label = _display_value(minimum["x"])
    if plan.series_field:
        maximum_label = f"{maximum_label}/{_display_value(maximum['series'])}"
        minimum_label = f"{minimum_label}/{_display_value(minimum['series'])}"
    return f"最高项为{maximum_label}（{maximum['value']:g} {unit}），最低项为{minimum_label}（{minimum['value']:g} {unit}），差值为{maximum['value'] - minimum['value']:g} {unit}{invalid}。"


def validate_and_render_chart_plan(
    plan: DynamicChartPlan,
    source_task_id: str,
    result: SemanticQueryResult,
    *,
    chart_id: str = "pending",
) -> RenderedChart:
    columns = {column.name: column for column in result.columns}
    referenced = {
        plan.x_field,
        *plan.group_by,
        *[chart_filter.field for chart_filter in plan.filters],
    }
    if plan.value_field:
        referenced.add(plan.value_field)
    if plan.series_field:
        referenced.add(plan.series_field)
    if plan.unit_field:
        referenced.add(plan.unit_field)
    missing = sorted(referenced - set(columns))
    if missing:
        raise ChartPlanValidationError(f"引用字段不存在：{missing}")
    filtered = _apply_chart_filters(result.rows, plan.filters, set(columns))
    aggregated, invalid_numeric_count, valid_row_count = _aggregate_chart_rows(plan, filtered)
    if plan.aggregation == ChartAggregation.COUNT:
        unit = "条"
    elif plan.unit_field:
        units = _unique([
            row.get(plan.unit_field)
            for row in filtered
            if row.get(plan.unit_field) not in (None, "")
        ])
        if len(units) > 1:
            raise ChartPlanValidationError(f"过滤后存在多个单位：{[_display_value(item) for item in units]}")
        unit = _display_value(units[0]) if units else "单位未提供"
    else:
        unit = "单位未提供"
    ordered = _ordered_chart_rows(aggregated, plan)
    truncated = False
    if plan.chart_type == ChartType.LINE:
        displayed, truncated = _line_rows(ordered, plan)
    elif plan.chart_type == ChartType.PIE:
        if plan.aggregation not in {ChartAggregation.COUNT, ChartAggregation.SUM}:
            raise ChartPlanValidationError("饼图只支持 count 或 sum 聚合")
        if not 2 <= len(ordered) <= 8:
            raise ChartPlanValidationError("饼图必须包含 2 至 8 个类别")
        if any(row["value"] < 0 for row in ordered):
            raise ChartPlanValidationError("饼图不能包含负值")
        if sum(row["value"] for row in ordered) <= 0:
            raise ChartPlanValidationError("饼图总值必须大于 0")
        displayed = ordered
    elif plan.chart_type == ChartType.BAR_STACK:
        displayed, truncated = _limited_category_rows(
            ordered,
            plan,
            maximum_categories=20,
            maximum_series=8,
        )
    else:
        displayed, truncated = _limited_category_rows(
            ordered,
            plan,
            maximum_categories=30,
        )
    if not displayed:
        raise ChartPlanValidationError("图表没有可展示的数据点")
    value_title = columns[plan.value_field].title if plan.value_field else "记录数"
    series_data = _chart_series_data(displayed, plan, value_title, unit)
    displayed_point_count = sum(len(series["data"]) for series in series_data)
    if displayed_point_count > 1000:
        raise ChartPlanValidationError("单图总数据点不能超过 1000")
    evidence = ChartEvidence(
        chart_id=chart_id,
        title=plan.title,
        chart_type=plan.chart_type,
        source_task_id=source_task_id,
        filters=plan.filters,
        aggregation=plan.aggregation,
        unit=unit,
        input_row_count=len(result.rows),
        filtered_row_count=len(filtered),
        valid_row_count=valid_row_count,
        displayed_point_count=displayed_point_count,
        truncated=truncated,
        summary=_chart_summary(displayed, plan, unit, invalid_numeric_count),
    )
    return RenderedChart(plan=plan, evidence=evidence, series_data=series_data)


def chart_plan_fingerprint(source_task_id: str, plan: DynamicChartPlan) -> str:
    normalized_filters = []
    for chart_filter in plan.filters:
        value = chart_filter.value
        if chart_filter.operator in {
            ChartFilterOperator.IN,
            ChartFilterOperator.NOT_IN,
        } and isinstance(value, list):
            value = sorted(value, key=_value_key)
        normalized_filters.append({
            "field": chart_filter.field,
            "operator": chart_filter.operator.value,
            "value": value,
        })
    payload = {
        "source_task_id": source_task_id,
        "chart_type": plan.chart_type.value,
        "x_field": plan.x_field,
        "value_field": plan.value_field,
        "series_field": plan.series_field,
        "filters": sorted(normalized_filters, key=_value_key),
        "aggregation": plan.aggregation.value,
    }
    return json.dumps(payload, ensure_ascii=False, sort_keys=True, default=str)


def validate_chart_planning_response(
    response: ChartPlanningResponse,
    datasets: Sequence[tuple[str, str, SemanticQueryResult]],
    profiles: Sequence[TaskDataProfile],
    *,
    existing: Sequence[RenderedChart] = (),
    global_limit: int = 8,
    required_task_ids: set[str] | None = None,
) -> tuple[list[RenderedChart], list[dict[str, str]]]:
    dataset_by_id = {task_id: result for task_id, _, result in datasets}
    task_order = {task_id: index for index, (task_id, _, _) in enumerate(datasets)}
    profile_by_id = {profile.source_task_id: profile for profile in profiles}
    required = required_task_ids or set(dataset_by_id)
    plans_by_id: dict[str, TaskChartPlans] = {}
    errors: list[dict[str, str]] = []
    for task_plan in response.tasks:
        task_id = task_plan.source_task_id
        if task_id not in dataset_by_id or task_id not in required:
            errors.append({"source_task_id": task_id, "error": "source_task_id 不属于本次待规划任务"})
            continue
        if task_id in plans_by_id:
            errors.append({"source_task_id": task_id, "error": "同一任务不能重复返回"})
            continue
        plans_by_id[task_id] = task_plan
    existing_counts = Counter(chart.evidence.source_task_id for chart in existing)
    seen = {
        chart_plan_fingerprint(chart.evidence.source_task_id, chart.plan)
        for chart in existing
    }
    candidates = sorted(
        [
            (plan.priority, task_order[task_id], plan_index, task_id, plan)
            for task_id, task_plan in plans_by_id.items()
            for plan_index, plan in enumerate(task_plan.charts)
        ],
        key=lambda item: item[:3],
    )
    accepted: list[RenderedChart] = []
    for _, _, plan_index, task_id, plan in candidates:
        if len(existing) + len(accepted) >= global_limit:
            errors.append({"source_task_id": task_id, "error": f"第 {plan_index + 1} 张图超过全局图表预算"})
            continue
        if existing_counts[task_id] + sum(
            chart.evidence.source_task_id == task_id for chart in accepted
        ) >= 4:
            errors.append({"source_task_id": task_id, "error": f"第 {plan_index + 1} 张图超过单任务预算"})
            continue
        fingerprint = chart_plan_fingerprint(task_id, plan)
        if fingerprint in seen:
            errors.append({"source_task_id": task_id, "error": f"第 {plan_index + 1} 张图与已有计划重复"})
            continue
        try:
            chart = validate_and_render_chart_plan(plan, task_id, dataset_by_id[task_id])
        except (ChartPlanValidationError, ValueError) as exc:
            errors.append({"source_task_id": task_id, "error": f"第 {plan_index + 1} 张图校验失败：{exc}"})
            continue
        seen.add(fingerprint)
        accepted.append(chart)
    for task_id in required:
        task_accepted = existing_counts[task_id] + sum(
            chart.evidence.source_task_id == task_id for chart in accepted
        )
        if task_id not in plans_by_id:
            errors.append({"source_task_id": task_id, "error": "规划响应缺少该成功任务"})
        elif task_accepted == 0 and profile_by_id[task_id].has_chart_opportunity:
            errors.append({"source_task_id": task_id, "error": "数据存在基本绘图机会，但没有有效图表计划"})
    return accepted, errors


def assign_chart_ids(
    charts: Sequence[RenderedChart],
    datasets: Sequence[tuple[str, str, SemanticQueryResult]],
) -> list[RenderedChart]:
    task_order = {task_id: index for index, (task_id, _, _) in enumerate(datasets)}
    ordered = sorted(
        charts,
        key=lambda chart: (
            chart.plan.priority,
            task_order[chart.evidence.source_task_id],
            list(charts).index(chart),
        ),
    )
    counters: Counter[str] = Counter()
    assigned = []
    for chart in ordered:
        task_id = chart.evidence.source_task_id
        counters[task_id] += 1
        chart_id = f"chart-{task_id}-{counters[task_id]:02d}"
        assigned.append(chart.model_copy(update={
            "evidence": chart.evidence.model_copy(update={"chart_id": chart_id}),
        }))
    return assigned


_FRONTEND_CHART_TYPES = {"BAR", "LINE", "PIE", "BAR_STACK"}


def _frontend_chart_output(
    *,
    chart_type: str,
    chart_name: str,
    series_data: list[dict[str, Any]],
    stack_config: dict[str, Any] | None = None,
    chart_id: str | None = None,
) -> dict[str, Any]:
    normalized_type = chart_type.upper()
    if normalized_type not in _FRONTEND_CHART_TYPES:
        raise ValueError(f"前端不支持图表类型: {chart_type}")
    payload: dict[str, Any] = {
        "chartType": normalized_type,
        "chartName": chart_name,
        "hasData": bool(series_data),
        "seriesData": series_data,
    }
    if chart_id is not None:
        payload["chartId"] = chart_id
    if normalized_type == "BAR_STACK":
        payload["stackConfig"] = stack_config or {"stackGroups": ["总量"]}
    return build_structured_output(
        output_type=WorkflowOutputType.CHART_OUTPUT,
        data=payload,
    )


def build_report_outputs(
    datasets: Sequence[tuple[str, str, SemanticQueryResult]],
    charts: Sequence[RenderedChart],
) -> list[dict[str, Any]]:
    if not datasets:
        raise ValueError("报告至少需要一个成功查询结果")
    outputs = [
        _frontend_chart_output(
            chart_type=chart.plan.chart_type.value,
            chart_name=chart.plan.title,
            series_data=chart.series_data,
            chart_id=chart.evidence.chart_id,
        )
        for chart in charts
    ]
    for _, objective, result in datasets:
        headers = [
            {
                "field": column.name,
                "title": column.title,
                "dataType": column.data_type,
            }
            for column in result.columns
        ]
        outputs.append(table_output(
            table_name="详细数据" if len(datasets) == 1 else f"{objective} · 详细数据",
            headers=headers,
            rows=[
                {
                    column.name: _cell_value(row.get(column.name))
                    for column in result.columns
                }
                for row in result.rows[:500]
            ],
        ))
    return outputs

REPORT_FAILURE_WARNING = "Markdown 分析报告生成失败，已保留查询摘要。"


def rows_to_markdown(result: SemanticQueryResult) -> str:
    if not result.columns:
        return ""
    lines = [
        "| " + " | ".join(column.title for column in result.columns) + " |",
        "| " + " | ".join("---" for _ in result.columns) + " |",
    ]
    for row in result.rows:
        values = []
        for column in result.columns:
            value = row.get(column.name)
            text = json.dumps(value, ensure_ascii=False) if isinstance(value, (dict, list)) else str(value if value is not None else "")
            values.append(text.replace("|", "\\|").replace("\n", "<br>"))
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines)


def report_task_to_markdown(
    report_task: ReportTask,
    datasets: Sequence[tuple[str, str, SemanticQueryResult]],
    charts: Sequence[RenderedChart] = (),
) -> str:
    successful_ids = {task_id for task_id, _, _ in datasets}
    results_by_id = {
        result.task.task_id: result for result in report_task.task_results
    }
    lines = ["报告章节要求："]
    for section in report_task.sections:
        sources = []
        for task_id in section.source_task_ids:
            result = results_by_id[task_id]
            availability = "数据可用" if task_id in successful_ids else "数据不可用"
            sources.append(
                f"{task_id}（{result.task.objective}，{availability}）"
            )
        lines.extend([
            f"## {section.title}",
            f"章节目标：{section.objective}",
            f"数据来源：{'、'.join(sources)}",
            "",
        ])
    lines.append("数据库查询数据：")
    for task_id, objective, result in datasets:
        lines.extend([
            "",
            f"### {task_id}：{objective}",
            rows_to_markdown(result),
        ])
    if charts:
        lines.extend(["", "图表证据："])
        for chart in charts:
            evidence = chart.evidence
            lines.extend([
                "",
                f"### {evidence.title}（{evidence.chart_id}）",
                f"来源任务：{evidence.source_task_id}",
                f"图表类型：{evidence.chart_type.value}",
                f"过滤条件：{json.dumps([item.model_dump(mode='json') for item in evidence.filters], ensure_ascii=False, default=str)}",
                f"聚合方式：{evidence.aggregation.value}",
                f"单位：{evidence.unit}",
                f"数据规模：输入 {evidence.input_row_count} 行，过滤后 {evidence.filtered_row_count} 行，有效 {evidence.valid_row_count} 行，展示 {evidence.displayed_point_count} 个点，截断：{'是' if evidence.truncated else '否'}",
                f"确定性摘要：{evidence.summary}",
            ])
    return "\n".join(lines).strip()


def ensure_chart_references(
    answer: str,
    charts: Sequence[RenderedChart],
) -> str:
    missing = [chart for chart in charts if chart.evidence.title not in answer]
    if not missing:
        return answer
    lines = [answer.rstrip(), "", "## 图表解读"] if answer.strip() else ["## 图表解读"]
    for chart in missing:
        lines.extend([
            "",
            f"### {chart.evidence.title}",
            chart.evidence.summary,
        ])
    return "\n".join(lines).strip()


__all__ = [
    "REPORT_FAILURE_WARNING",
    "ChartPlanValidationError",
    "assign_chart_ids",
    "build_report_outputs",
    "build_task_data_profile",
    "build_task_data_profiles",
    "chart_plan_fingerprint",
    "ensure_chart_references",
    "has_basic_chart_opportunity",
    "parse_chart_planning_response",
    "report_task_to_markdown",
    "rows_to_markdown",
    "validate_and_render_chart_plan",
    "validate_chart_planning_response",
]
