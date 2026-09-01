from __future__ import annotations

from ..tool_call_parser import (
    contains_internal_protocol,
    parse_hydrology_tool_calls,
)

ALLOWED_NAMES = {"search_semantic_catalog", "run_semantic_query"}


def test_xml_run_semantic_query_is_parsed() -> None:
    content = """<tool_call>
<function=run_semantic_query>
<parameter=semantic_query>
{"query_mode":"view","models":["view_his_record"],"dimensions":["view_his_record.point_name","view_his_record.value_type_name","view_his_record.value","view_his_record.display_value","view_his_record.observed_at"],"filters":[{"member":"view_his_record.point_name","operator":"equals","values":["ZL5水温"]}],"limit":100,"order":[{"member":"view_his_record.observed_at","direction":"desc"}],"ungrouped":true}
</parameter>
</function>
</tool_call>"""

    calls = parse_hydrology_tool_calls(content, ALLOWED_NAMES)

    assert calls == [{
        "name": "run_semantic_query",
        "args": {
            "semantic_query": {
                "query_mode": "view",
                "models": ["view_his_record"],
                "dimensions": [
                    "view_his_record.point_name",
                    "view_his_record.value_type_name",
                    "view_his_record.value",
                    "view_his_record.display_value",
                    "view_his_record.observed_at",
                ],
                "filters": [{
                    "member": "view_his_record.point_name",
                    "operator": "equals",
                    "values": ["ZL5水温"],
                }],
                "limit": 100,
                "order": [{
                    "member": "view_his_record.observed_at",
                    "direction": "desc",
                }],
                "ungrouped": True,
            }
        },
        "id": "call_xml_0",
    }]


def test_xml_search_query_preserves_string_parameter() -> None:
    content = """<tool_call>
<function=search_semantic_catalog>
<parameter=query>查询ZL5水温</parameter>
<parameter=limit>20</parameter>
</function>
</tool_call>"""

    calls = parse_hydrology_tool_calls(content, ALLOWED_NAMES)

    assert calls[0]["name"] == "search_semantic_catalog"
    assert calls[0]["args"] == {"query": "查询ZL5水温", "limit": 20}


def test_json_tool_calls_supports_nested_function_arguments() -> None:
    content = r"""{
      "tool_calls": [{
        "id": "query_1",
        "function": {
          "name": "search_semantic_catalog",
          "arguments": "{\"query\":\"查询水位\"}"
        }
      }]
    }"""

    calls = parse_hydrology_tool_calls(content, ALLOWED_NAMES)

    assert calls == [{
        "name": "search_semantic_catalog",
        "args": {"query": "查询水位"},
        "id": "query_1",
    }]


def test_multiple_action_blocks_are_parsed() -> None:
    content = """Action: search_semantic_catalog
Action Input: {"query":"查询水位"}
Action: search_semantic_catalog
Action Input: {"query":"查询水温","limit":10}"""

    calls = parse_hydrology_tool_calls(content, ALLOWED_NAMES)

    assert [call["args"] for call in calls] == [
        {"query": "查询水位"},
        {"query": "查询水温", "limit": 10},
    ]


def test_generic_tool_call_format_remains_supported() -> None:
    calls = parse_hydrology_tool_calls(
        'search_semantic_catalog(query="查询水位")',
        ALLOWED_NAMES,
    )

    assert calls[0]["name"] == "search_semantic_catalog"
    assert calls[0]["args"] == {"query": "查询水位"}


def test_unknown_or_malformed_xml_is_not_executed() -> None:
    unknown = """<tool_call><function=unknown_tool></function></tool_call>"""
    malformed = """<tool_call><function=run_semantic_query><parameter=semantic_query>{}</function></tool_call>"""

    assert parse_hydrology_tool_calls(unknown, ALLOWED_NAMES) == []
    assert parse_hydrology_tool_calls(malformed, ALLOWED_NAMES) == []
    assert contains_internal_protocol(unknown)
    assert contains_internal_protocol(malformed)


def test_invalid_json_tool_calls_and_actions_are_not_executed() -> None:
    invalid_json_call = '{"tool_calls":[{"name":"run_semantic_query","arguments":"{"}]}'
    invalid_action = "Action: run_semantic_query\nAction Input: {"

    assert parse_hydrology_tool_calls(invalid_json_call, ALLOWED_NAMES) == []
    assert parse_hydrology_tool_calls(invalid_action, ALLOWED_NAMES) == []
    assert contains_internal_protocol(invalid_json_call)
    assert contains_internal_protocol(invalid_action)


def test_internal_protocol_detection_covers_query_observation_and_sql() -> None:
    samples = [
        '{"query_mode":"view","models":["view_his_record"]}',
        '{"kind":"semantic_query_result","rows_truncated":false}',
        '{"kind":"error","stage":"semantic_validation"}',
        "Action: run_semantic_query\nAction Input: {}",
        "Observation: {\"ok\":true}",
        "SELECT * FROM dbo.Dev_HisRecord",
    ]

    assert all(contains_internal_protocol(sample) for sample in samples)
    assert not contains_internal_protocol("未查询到符合当前条件的数据。")
