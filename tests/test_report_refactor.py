from __future__ import annotations

import json
from decimal import Decimal

import pytest
from pydantic import ValidationError

from ..contracts import (
    QueryOutcome,
    ReportFactCategory,
    SemanticColumn,
    SemanticQueryResult,
)
from ..report_child.report import analyze_result, build_result_outputs, validate_narrative


def _column(name: str, title: str, data_type: str, member_type: str) -> SemanticColumn:
    return SemanticColumn(name=name, title=title, data_type=data_type, member_type=member_type)


def _result(columns: list[SemanticColumn], rows: list[dict]) -> SemanticQueryResult:
    return SemanticQueryResult(outcome=QueryOutcome.SUCCESS, columns=columns, rows=rows, row_count=len(rows))


def test_analysis_uses_all_rows_and_detail_table_is_not_truncated() -> None:
    rows = [{"time": f"2026-08-{index % 28 + 1:02d}T{index % 24:02d}:00:00+08:00", "level": Decimal(index) / Decimal("10")} for index in range(1000)]
    result = _result([
        _column("time", "时间", "time", "time_dimension"),
        _column("level", "水位", "number", "measure"),
    ], rows)

    analysis = analyze_result(result)
    metric = next(fact for fact in analysis.facts if fact.category == ReportFactCategory.METRIC and fact.title == "水位核心统计")
    outputs = build_result_outputs(result, answer="", question="水位趋势")

    assert metric.value["count"] == 1000
    assert outputs[0]["output_type"] == "LLM_STREAM"
    detail = [output for output in outputs if output["output_type"] == "TABLE_OUTPUT"][-1]
    assert len(detail["data"]["rows"]) == 1000


def test_zero_baseline_does_not_calculate_percentage_change() -> None:
    result = _result([
        _column("time", "时间", "time", "time_dimension"),
        _column("level", "水位", "number", "measure"),
    ], [
        {"time": "2026-08-01T00:00:00Z", "level": 0},
        {"time": "2026-08-02T00:00:00+00:00", "level": 3},
    ])

    trend = next(fact for fact in analyze_result(result).facts if fact.category == ReportFactCategory.TREND)

    assert trend.value["change_rate_percent"] is None
    assert "起点为零" in trend.display_text


def test_explicit_threshold_outlier_and_correlation_are_separate_facts() -> None:
    rows = [
        {
            "value": value,
            "upper_threshold": 10,
            "flow": value * 2,
        }
        for value in [1, 2, 3, 4, 5, 6, 7, 30]
    ]
    result = _result([
        _column("value", "监测值", "number", "measure"),
        _column("upper_threshold", "上限阈值", "number", "measure"),
        _column("flow", "流量", "number", "measure"),
    ], rows)

    analysis = analyze_result(result)
    by_category = {category: [fact for fact in analysis.facts if fact.category == category] for category in ReportFactCategory}

    assert by_category[ReportFactCategory.THRESHOLD][0].value["exceeded_count"] == 1
    assert by_category[ReportFactCategory.ANOMALY][0].value["outliers"] == [30.0]
    assert by_category[ReportFactCategory.CORRELATION]
    assert "不代表因果" in by_category[ReportFactCategory.CORRELATION][0].display_text
    assert "不等同于水害风险" in by_category[ReportFactCategory.ANOMALY][0].display_text


def test_threshold_is_never_inferred_and_non_finite_values_are_reported() -> None:
    result = _result([
        _column("value", "监测值", "number", "measure"),
    ], [{"value": 1}, {"value": float("inf")}, {"value": None}])

    analysis = analyze_result(result)
    non_finite = next(fact for fact in analysis.facts if fact.title == "非有限数值")

    assert non_finite.value == 1
    assert not any(fact.category == ReportFactCategory.THRESHOLD for fact in analysis.facts)
    assert any(item.code == "missing_threshold" for item in analysis.limitations)
    assert any(item.code == "missing_unit_value" for item in analysis.limitations)


def test_identical_values_do_not_create_invalid_correlation() -> None:
    result = _result([
        _column("level", "水位", "number", "measure"),
        _column("flow", "流量", "number", "measure"),
    ], [{"level": 3, "flow": 5} for _ in range(8)])

    analysis = analyze_result(result)

    assert not any(fact.category == ReportFactCategory.CORRELATION for fact in analysis.facts)
    anomaly = next(fact for fact in analysis.facts if fact.category == ReportFactCategory.ANOMALY)
    assert anomaly.value["outliers"] == []


def test_explicit_unit_is_preserved_without_inference() -> None:
    result = _result([
        _column("level", "水位", "number", "measure"),
        _column("unit", "单位", "string", "dimension"),
    ], [{"level": 3.2, "unit": "m"}, {"level": 3.4, "unit": "m"}])

    analysis = analyze_result(result)
    metric = next(fact for fact in analysis.facts if fact.category == ReportFactCategory.METRIC)

    assert metric.unit == "m"
    assert not any(item.code == "missing_unit_level" for item in analysis.limitations)


def test_fixed_sections_explain_unsupported_analysis_without_empty_charts() -> None:
    result = _result([
        _column("station", "站点", "string", "dimension"),
    ], [{"station": "A"}, {"station": "B"}])

    outputs = build_result_outputs(result, answer="", question="站点综合分析")
    markdown = outputs[0]["data"]["text"]

    assert all(f"## {index}." in markdown for index in range(1, 11))
    assert "结果中没有时间字段，无法验证趋势" in markdown
    assert "结果中没有可用数值字段" in markdown
    assert not any(output["output_type"] == "CHART_OUTPUT" for output in outputs)


def test_narrative_accepts_valid_fact_reference() -> None:
    analysis = analyze_result(_result([
        _column("level", "水位", "number", "measure"),
    ], [{"level": 4.2}]))
    fact = next(item for item in analysis.facts if item.category == ReportFactCategory.METRIC)
    raw = json.dumps({
        "title": "水位综合报告",
        "executive_summary": "数据事实可直接验证。",
        "insights": [{
            "title": "水位解读",
            "fact_ids": [fact.fact_id],
            "interpretation": "当前统计可由事实支持。",
            "impact": "影响需结合明确阈值判断。",
            "possible_cause": "可能原因尚无足够证据。",
            "recommendation": "建议完成现场复核后再采取行动。",
            "certainty": "medium",
        }],
    }, ensure_ascii=False)

    draft = validate_narrative(raw, analysis)

    assert draft.insights[0].fact_ids == [fact.fact_id]


@pytest.mark.parametrize(
    "mutation,expected",
    [
        ({"fact_ids": ["metric-999"]}, "无效事实"),
        ({"interpretation": "新增定量值999"}, "未经事实引用支持"),
        ({"recommendation": "<tool_call>run_semantic_query</tool_call>"}, "内部工具协议"),
    ],
)
def test_narrative_rejects_invalid_evidence(mutation: dict, expected: str) -> None:
    analysis = analyze_result(_result([
        _column("level", "水位", "number", "measure"),
    ], [{"level": 4.2}]))
    fact = next(item for item in analysis.facts if item.category == ReportFactCategory.METRIC)
    insight = {
        "title": "水位解读",
        "fact_ids": [fact.fact_id],
        "interpretation": "事实可验证。",
        "impact": "影响待判断。",
        "possible_cause": "可能原因待核验。",
        "recommendation": "建议现场复核。",
        "certainty": "low",
    }
    insight.update(mutation)
    raw = json.dumps({"title": "综合报告", "executive_summary": "事实可验证。", "insights": [insight]}, ensure_ascii=False)

    with pytest.raises(ValueError, match=expected):
        validate_narrative(raw, analysis)


def test_narrative_rejects_invalid_schema_and_empty_response() -> None:
    analysis = analyze_result(_result([], []))

    with pytest.raises(ValidationError):
        validate_narrative('{"title":"报告"}', analysis)
    with pytest.raises(ValueError, match="空响应"):
        validate_narrative("", analysis)
