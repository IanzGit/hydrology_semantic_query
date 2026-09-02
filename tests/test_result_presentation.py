from __future__ import annotations

import pytest

from app.agents.scenarios.hydrology_semantic_query.query_child.tools.run_semantic_query import (
    normalize_cube_response,
)

from ..contracts import (
    ChartType,
    ColumnRole,
    PresentationBlockType,
    QueryOutcome,
    SemanticColumn,
    SemanticQueryResult,
)
from ..report_child.report import build_presentation_plan, build_result_outputs


def _result(
    columns: list[SemanticColumn],
    rows: list[dict],
    outcome: QueryOutcome = QueryOutcome.SUCCESS,
) -> SemanticQueryResult:
    return SemanticQueryResult(
        outcome=outcome,
        columns=columns,
        rows=rows,
        row_count=len(rows),
    )


def _column(
    name: str,
    title: str,
    data_type: str,
    member_type: str,
) -> SemanticColumn:
    return SemanticColumn(
        name=name,
        title=title,
        data_type=data_type,
        member_type=member_type,
    )


def _blocks(result: SemanticQueryResult):
    return [
        block
        for section in result.presentation.sections
        for block in section.blocks
    ]


def _chart_payload(output: dict) -> dict:
    assert output["output_type"] == "CHART_OUTPUT"
    assert set(output["data"]) >= {
        "chartType",
        "chartName",
        "hasData",
        "seriesData",
    }
    assert "chartData" not in output["data"]
    assert output["data"]["chartType"] in {"BAR", "LINE", "PIE", "BAR_STACK"}
    return output["data"]


def test_report_chart_types_are_supported_by_frontend() -> None:
    assert {chart_type.value for chart_type in ChartType} == {"BAR", "LINE", "PIE"}


def test_normalize_cube_response_preserves_member_semantics() -> None:
    columns, rows = normalize_cube_response({
        "data": [{"station": "A", "time": "2026-08-23", "level": "3.2"}],
        "annotation": {
            "dimensions": {"station": {"title": "站点", "type": "string"}},
            "timeDimensions": {"time": {"title": "时间", "type": "time"}},
            "measures": {"level": {"title": "水位", "type": "number"}},
        },
    })

    assert [column.member_type for column in columns] == [
        "dimension",
        "time_dimension",
        "measure",
    ]
    assert rows == [{"station": "A", "time": "2026-08-23", "level": 3.2}]


def test_normalize_cube_response_filters_identifier_and_code_columns() -> None:
    columns, rows = normalize_cube_response({
        "data": [{
            "device.id": "device-1",
            "sensor_ids": "sensor-1,sensor-2",
            "relation_key": "relation-1",
            "device.code": "D001",
            "alarm_source": "source-1",
            "device_name": "1 号泵站",
            "level": "3.2",
        }],
        "annotation": {
            "dimensions": {
                "device.id": {"title": "设备ID", "type": "string"},
                "sensor_ids": {"title": "传感器ID列表", "type": "string"},
                "relation_key": {"title": "关系键", "type": "string"},
                "device.code": {"title": "设备编码", "type": "string"},
                "alarm_source": {"title": "报警来源ID", "type": "string"},
                "device_name": {"title": "设备名称", "type": "string"},
            },
            "measures": {"level": {"title": "水位", "type": "number"}},
        },
    })

    assert [column.name for column in columns] == ["device_name", "level"]
    assert rows == [{"device_name": "1 号泵站", "level": 3.2}]


def test_normalize_cube_response_rejects_malformed_rows_and_non_finite_numbers() -> None:
    with pytest.raises(ValueError, match="data 只能包含对象"):
        normalize_cube_response({"data": [{"value": 1}, "invalid"]})
    with pytest.raises(ValueError, match="results 只能包含对象"):
        normalize_cube_response({"results": ["invalid"]})

    _, rows = normalize_cube_response({
        "data": [{"value": "Infinity", "enabled": "unknown"}],
        "annotation": {
            "measures": {"value": {"type": "number"}},
            "dimensions": {"enabled": {"type": "boolean"}},
        },
    })

    assert rows == [{"value": None, "enabled": "unknown"}]


def test_scalar_result_outputs_full_report_without_empty_chart() -> None:
    result = _result(
        [_column("level", "当前水位", "number", "measure")],
        [{"level": 4.26}],
    )

    outputs = build_result_outputs(result, answer="查询完成。", question="当前水位")

    assert [output["output_type"] for output in outputs] == [
        "LLM_STREAM",
        "TABLE_OUTPUT",
        "TABLE_OUTPUT",
    ]
    assert "## 10. 详细数据" in outputs[0]["data"]["text"]
    assert all(output["output_type"] != "CHART_OUTPUT" for output in outputs)
    assert result.presentation is not None
    assert result.presentation.profile.shape.value == "scalar"
    assert [block.type for block in _blocks(result)] == [
        PresentationBlockType.TABLE,
    ]


def test_categorical_result_automatically_outputs_bar_and_table() -> None:
    result = _result(
        [
            _column("station", "站点", "string", "dimension"),
            _column("flow", "流量", "number", "measure"),
        ],
        [{"station": "A", "flow": 3.2}, {"station": "B", "flow": 5.1}],
    )

    outputs = build_result_outputs(result, answer="查询完成。", question="各站点流量")

    assert [output["output_type"] for output in outputs] == [
        "LLM_STREAM",
        "CHART_OUTPUT",
        "TABLE_OUTPUT",
        "TABLE_OUTPUT",
    ]
    chart = _chart_payload(outputs[1])
    assert chart["chartType"] == "BAR"
    assert chart["seriesData"][0]["data"][0] == {
        "name": "B",
        "value": 5.1,
    }


def test_explicit_pie_overrides_default_bar_without_dropping_table() -> None:
    result = _result(
        [
            _column("type", "类型", "string", "dimension"),
            _column("count", "数量", "number", "measure"),
        ],
        [{"type": "雨量站", "count": 3}, {"type": "水位站", "count": 7}],
    )

    outputs = build_result_outputs(result, answer="", question="用饼图展示站点构成")

    assert [output["output_type"] for output in outputs] == [
        "LLM_STREAM",
        "CHART_OUTPUT",
        "TABLE_OUTPUT",
        "TABLE_OUTPUT",
    ]
    assert _chart_payload(outputs[1])["chartType"] == "PIE"


def test_temporal_multi_series_plan_uses_line_without_ambiguous_kpi() -> None:
    result = _result(
        [
            _column("time", "时间", "time", "time_dimension"),
            _column("station", "站点", "string", "dimension"),
            _column("level", "水位", "number", "measure"),
        ],
        [
            {"time": "2026-08-22", "station": "A", "level": 3.1},
            {"time": "2026-08-22", "station": "B", "level": 3.5},
            {"time": "2026-08-23", "station": "A", "level": 3.4},
            {"time": "2026-08-23", "station": "B", "level": 3.8},
        ],
    )

    plan = build_presentation_plan(result, "水位趋势")
    chart = next(block for block in plan.blocks if block.type == PresentationBlockType.CHART)

    assert chart.config.chart_type == ChartType.LINE
    assert chart.config.x.name == "time"
    assert chart.config.series.name == "station"
    assert not any(block.type == PresentationBlockType.KPI for block in plan.blocks)


def test_latest_kpi_compares_timezone_offsets_chronologically() -> None:
    result = _result(
        [
            _column("time", "时间", "time", "time_dimension"),
            _column("level", "水位", "number", "measure"),
        ],
        [
            {"time": "2026-01-01T01:00:00+08:00", "level": 1},
            {"time": "2025-12-31T18:00:00+00:00", "level": 2},
        ],
    )

    outputs = build_result_outputs(result, answer="", question="最新水位")

    chart = next(output for output in outputs if output["output_type"] == "CHART_OUTPUT")
    assert [point["value"] for point in _chart_payload(chart)["seriesData"][0]["data"]] == [1.0, 2.0]
    latest = next(fact for fact in result.presentation.facts if fact.title == "水位最新值")
    assert latest.value["latest"] == 2.0


def test_time_station_matrix_automatically_outputs_line() -> None:
    rows = [
        {"time": "2026-08-23", "station": f"S{index}", "level": index + 0.5}
        for index in range(9)
    ]
    result = _result(
        [
            _column("time", "时间", "time", "time_dimension"),
            _column("station", "站点", "string", "dimension"),
            _column("level", "水位", "number", "measure"),
        ],
        rows,
    )

    outputs = build_result_outputs(result, answer="", question="站点水位概览")

    assert [output["output_type"] for output in outputs] == [
        "LLM_STREAM",
        "CHART_OUTPUT",
        "TABLE_OUTPUT",
        "TABLE_OUTPUT",
    ]
    chart = _chart_payload(outputs[1])
    assert chart["chartType"] == "LINE"
    assert len(chart["seriesData"]) == 9


def test_status_field_outputs_status_cards_and_table() -> None:
    result = _result(
        [
            _column("device", "设备", "string", "dimension"),
            _column("alarm_status", "告警状态", "string", "dimension"),
        ],
        [
            {"device": "A", "alarm_status": "正常"},
            {"device": "B", "alarm_status": "严重告警"},
            {"device": "C", "alarm_status": "严重告警"},
        ],
    )

    outputs = build_result_outputs(result, answer="", question="设备状态")

    assert [output["output_type"] for output in outputs] == [
        "LLM_STREAM",
        "CHART_OUTPUT",
        "TABLE_OUTPUT",
    ]
    chart = _chart_payload(outputs[1])
    assert chart["chartType"] == "PIE"
    assert chart["seriesData"][0]["data"][0] == {
        "name": "严重告警",
        "value": 2,
    }
    status_block = next(
        block for block in _blocks(result) if block.type == PresentationBlockType.STATUS
    )
    assert status_block.data["items"][0] == {"label": "严重告警", "count": 2}


def test_alarm_names_remain_categories_and_alarm_levels_render_separate_pies() -> None:
    result = _result(
        [
            _column("alarm_name", "报警名称", "string", "dimension"),
            _column("current_level", "当前报警等级", "string", "dimension"),
            _column("highest_level", "历史最高报警等级", "string", "dimension"),
        ],
        [
            {
                "alarm_name": "下游水位",
                "current_level": "解除预警",
                "highest_level": "红色预警",
            },
            {
                "alarm_name": "井口温度",
                "current_level": "解除预警",
                "highest_level": "蓝色预警",
            },
        ],
    )

    outputs = build_result_outputs(result, answer="", question="查询单因素报警信息")

    charts = [output for output in outputs if output["output_type"] == "CHART_OUTPUT"]
    assert [chart["data"]["chartName"] for chart in charts] == ["历史最高报警等级分布"]
    assert all(_chart_payload(chart)["chartType"] == "PIE" for chart in charts)
    assert result.presentation.profile.primary_category == "alarm_name"
    assert result.presentation.profile.status_fields == ["current_level", "highest_level"]


def test_coordinates_remain_facts_and_table_without_map_block() -> None:
    result = _result(
        [
            _column("station", "站点", "string", "dimension"),
            _column("longitude", "经度", "number", "dimension"),
            _column("latitude", "纬度", "number", "dimension"),
            _column("flow", "流量", "number", "measure"),
        ],
        [
            {"station": "A", "longitude": 116.4, "latitude": 39.9, "flow": 5.2},
            {"station": "B", "longitude": 121.5, "latitude": 31.2, "flow": 3.8},
        ],
    )

    outputs = build_result_outputs(result, answer="", question="站点空间分布")

    assert [output["output_type"] for output in outputs] == [
        "LLM_STREAM",
        "CHART_OUTPUT",
        "TABLE_OUTPUT",
        "TABLE_OUTPUT",
    ]
    assert _chart_payload(outputs[1])["chartType"] == "BAR"
    assert all(block.type != PresentationBlockType.MAP for block in _blocks(result))
    assert result.presentation.profile.latitude_field == "latitude"
    assert result.presentation.profile.longitude_field == "longitude"
    assert next(
        column.role
        for column in result.presentation.profile.columns
        if column.name == "latitude"
    ) == ColumnRole.LATITUDE


def test_failed_result_only_outputs_answer_without_presentation() -> None:
    result = _result([], [], QueryOutcome.EXECUTION_ERROR)

    outputs = build_result_outputs(result, answer="查询失败。", question="查询水位")

    assert [output["output_type"] for output in outputs] == ["LLM_STREAM"]
    assert result.presentation is None


def test_success_outputs_never_use_unverified_echarts_stream_blocks() -> None:
    result = _result(
        [
            _column("time", "时间", "time", "time_dimension"),
            _column("station", "站点", "string", "dimension"),
            _column("level", "水位", "number", "measure"),
        ],
        [
            {"time": "2026-08-22", "station": "A", "level": 3.1},
            {"time": "2026-08-23", "station": "B", "level": 3.8},
        ],
    )

    outputs = build_result_outputs(result, answer="已生成报告。", question="水位热力图")

    assert outputs[0]["output_type"] == "LLM_STREAM"
    chart = next(output for output in outputs if output["output_type"] == "CHART_OUTPUT")
    assert _chart_payload(chart)["chartType"] == "LINE"
    for output in outputs:
        if output["output_type"] == "CHART_OUTPUT":
            _chart_payload(output)
