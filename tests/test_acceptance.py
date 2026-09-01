from __future__ import annotations

import json
from typing import Any

import pytest
from langchain_core.messages import AIMessage

from ..client import CubeClientError
from ..config import HydrologySemanticQuerySettings
from ..graph import build_hydrology_semantic_query_graph
from ..models import QueryOutcome, SemanticCatalogMode
from ..runtime import HydrologySemanticQueryServices


def _meta() -> dict[str, Any]:
    return {
        "cubes": [
            {
                "name": "base_device_info",
                "type": "cube",
                "public": True,
                "title": "设备信息",
                "description": "水文设备基础信息，设备名称属于此实体",
                "connectedComponent": 1,
                "meta": {
                    "business_domain": "hydrology",
                    "default_projection": ["name", "code"],
                },
                "measures": [],
                "dimensions": [
                    {
                        "name": "base_device_info.name",
                        "title": "设备名称",
                        "type": "string",
                        "public": True,
                    },
                    {
                        "name": "base_device_info.code",
                        "title": "设备编码",
                        "type": "string",
                        "public": True,
                    },
                ],
                "segments": [],
                "folders": [],
                "hierarchies": [],
            },
            {
                "name": "base_device_x_value",
                "type": "cube",
                "public": True,
                "title": "传感器实时值",
                "description": "传感器当前值、状态和更新时间",
                "connectedComponent": 1,
                "meta": {"business_domain": "hydrology"},
                "measures": [
                    {
                        "name": "base_device_x_value.sample_count",
                        "title": "样本数",
                        "type": "number",
                        "public": True,
                    }
                ],
                "dimensions": [
                    {
                        "name": "base_device_x_value.sensor_current_value",
                        "title": "传感器当前值",
                        "type": "number",
                        "public": True,
                    },
                    {
                        "name": "base_device_x_value.sensor_status",
                        "title": "传感器状态",
                        "type": "string",
                        "public": True,
                    },
                    {
                        "name": "base_device_x_value.updated_at",
                        "title": "更新时间",
                        "type": "time",
                        "public": True,
                    },
                ],
                "segments": [],
                "folders": [],
                "hierarchies": [],
            },
            {
                "name": "hydrology_water_quality_view",
                "type": "view",
                "public": True,
                "title": "水质汇总",
                "meta": {"business_domain": "hydrology"},
                "measures": [
                    {
                        "name": "hydrology_water_quality_view.sample_count",
                        "title": "样本数",
                        "type": "number",
                        "public": True,
                    }
                ],
                "dimensions": [
                    {
                        "name": "hydrology_water_quality_view.device_name",
                        "title": "设备名称",
                        "type": "string",
                        "public": True,
                    }
                ],
                "segments": [],
                "folders": [],
                "hierarchies": [],
            },
        ]
    }


def _query(
    *,
    model: str = "base_device_x_value",
    dimensions: list[str] | None = None,
    measures: list[str] | None = None,
    models: list[str] | None = None,
    filters: list[dict[str, Any]] | None = None,
    ungrouped: bool | None = None,
) -> str:
    selected_models = models or [model]
    selected_dimensions = dimensions or []
    selected_measures = measures or []
    if ungrouped is None:
        ungrouped = bool(selected_dimensions and not selected_measures)
    return json.dumps({
        "query_mode": "view" if selected_models[0].startswith("hydrology_") else "cube",
        "models": selected_models,
        "measures": selected_measures,
        "dimensions": selected_dimensions,
        "segments": [],
        "filters": filters or [],
        "time_dimensions": [],
        "order": [],
        "limit": 10,
        "offset": 0,
        "ungrouped": ungrouped,
    }, ensure_ascii=False)


def _central_water_query() -> str:
    return _query(
        models=["base_device_info", "base_device_x_value"],
        dimensions=[
            "base_device_info.name",
            "base_device_x_value.sensor_current_value",
            "base_device_x_value.sensor_status",
            "base_device_x_value.updated_at",
        ],
        filters=[{
            "member": "base_device_info.name",
            "operator": "equals",
            "values": ["中央水仓"],
        }],
        ungrouped=True,
    )


class FakeModel:
    def __init__(self, responses: list[str | Exception | AIMessage]) -> None:
        self.responses = list(responses)
        self.messages: list[list[Any]] = []
        self.bindings: list[dict[str, Any]] = []
        self.bound_tools: list[list[Any]] = []

    def bind(self, **kwargs: Any) -> FakeModel:
        self.bindings.append(kwargs)
        return self

    def bind_tools(self, tools: list[Any], **kwargs: Any) -> FakeModel:
        self.bound_tools.append(tools)
        self.bindings.append(kwargs)
        return self

    async def ainvoke(self, messages: list[Any], config: Any = None) -> AIMessage:
        del config
        self.messages.append(messages)
        response = self.responses.pop(0)
        if isinstance(response, Exception):
            raise response
        if isinstance(response, AIMessage):
            return response
        return AIMessage(content=response)


class FakeRuntime:
    def __init__(
        self,
        responses: list[str | Exception | AIMessage],
        report_responses: list[str | Exception | AIMessage] | None = None,
    ) -> None:
        self.model = FakeModel(responses)
        self.report_model = FakeModel(
            report_responses if report_responses is not None else ["默认查询报告。"]
        )

    def get_chat_model(self, streaming: bool) -> FakeModel:
        return self.report_model if streaming else self.model


class FakeCubeClient:
    def __init__(
        self,
        *,
        sql_failures: list[Exception | None] | None = None,
        load_results: list[dict[str, Any] | Exception] | None = None,
    ) -> None:
        self.sql_failures = list(sql_failures or [])
        self.load_results = list(load_results or [])
        self.sql_queries: list[dict[str, Any]] = []
        self.load_queries: list[dict[str, Any]] = []
        self.events: list[str] = []

    async def get_meta(self) -> dict[str, Any]:
        self.events.append("meta")
        return _meta()

    async def get_sql(self, query: dict[str, Any]) -> tuple[str, list[Any]]:
        self.events.append("sql")
        self.sql_queries.append(query)
        if self.sql_failures:
            failure = self.sql_failures.pop(0)
            if failure is not None:
                raise failure
        return f"SELECT {len(self.sql_queries)} FROM governed_semantic_model", []

    async def load(self, query: dict[str, Any]) -> dict[str, Any]:
        self.events.append("load")
        self.load_queries.append(query)
        if self.load_results:
            result = self.load_results.pop(0)
            if isinstance(result, Exception):
                raise result
            return result
        dimensions = query.get("dimensions", [])
        measures = query.get("measures", [])
        row = {name: "value" for name in [*dimensions, *measures]}
        annotation = {
            "dimensions": {
                name: {"title": name, "type": "string"} for name in dimensions
            },
            "measures": {
                name: {"title": name, "type": "number"} for name in measures
            },
        }
        return {"data": [row], "annotation": annotation}


class TrackingServices(HydrologySemanticQueryServices):
    def __init__(self, settings: HydrologySemanticQuerySettings, client: Any) -> None:
        super().__init__(settings, client=client, embedding_client=None)

    @property
    def retrieval_limits(self) -> list[int | None]:
        return self.catalog_search.retrieval_limits if self.catalog_search else []

    @property
    def retrieval_questions(self) -> list[str]:
        return self.catalog_search.retrieval_questions if self.catalog_search else []


async def _invoke(
    responses: list[str | Exception | AIMessage],
    client: FakeCubeClient,
    *,
    question: str = "查询中央水仓水质",
    max_retries: int = 1,
    max_agent_iterations: int = 6,
    conversation_context: Any = None,
    report_responses: list[str | Exception | AIMessage] | None = None,
    captured_stream_outputs: list[dict[str, Any]] | None = None,
) -> tuple[dict[str, Any], FakeRuntime, TrackingServices]:
    runtime = FakeRuntime(responses, report_responses)
    settings = HydrologySemanticQuerySettings(
        cube_url="http://cube/cubejs-api/v1",
        cube_token=None,
        timeout_seconds=1,
        continue_wait_retries=0,
        meta_cache_ttl_seconds=60,
        max_retries=max_retries,
        max_rows=10,
        hard_max_rows=100,
        timezone="Asia/Shanghai",
        catalog_mode=SemanticCatalogMode.VECTOR,
        embedding_model=None,
        context_top_k=20,
        vector_index_path=None,
        embedding_batch_size=4,
        embedding_concurrency=1,
        auto_full_context_max_chars=30000,
        max_agent_iterations=max_agent_iterations,
    )
    services = TrackingServices(settings, client)
    graph = build_hydrology_semantic_query_graph(runtime, services).compile()
    metadata = {
        "business_knowledge": "中央水仓是设备名称",
    }
    if conversation_context is not None:
        metadata["conversation_context"] = conversation_context
    graph_input = {
        "query": question,
        "metadata": metadata,
    }
    if captured_stream_outputs is None:
        state = await graph.ainvoke(graph_input)
    else:
        state = None
        async for event in graph.astream_events(graph_input, version="v2"):
            if event.get("event") != "on_chain_end":
                continue
            output = event.get("data", {}).get("output")
            if event.get("name") == "LangGraph":
                state = output
                continue
            if not isinstance(output, dict):
                continue
            captured_stream_outputs.extend(output.get("stream_outputs", []))
        assert state is not None
    return state, runtime, services


def _search(query: str, *, limit: int | None = None) -> AIMessage:
    args: dict[str, Any] = {"query": query}
    if limit is not None:
        args["limit"] = limit
    return AIMessage(
        content="",
        tool_calls=[{
            "name": "search_semantic_catalog",
            "args": args,
            "id": f"search_{query}_{limit}",
        }],
    )


def _run(query: str | dict[str, Any]) -> AIMessage:
    semantic_query = json.loads(query) if isinstance(query, str) else query
    return AIMessage(
        content="",
        tool_calls=[{
            "name": "run_semantic_query",
            "args": {"semantic_query": semantic_query},
            "id": f"run_{len(json.dumps(semantic_query, ensure_ascii=False))}",
        }],
    )


def _answer(text: str = "查询完成。") -> AIMessage:
    return AIMessage(content=text)


def _xml_tool_call(name: str, args: dict[str, Any]) -> str:
    parameters = []
    for key, value in args.items():
        content = value if isinstance(value, str) else json.dumps(value, ensure_ascii=False)
        parameters.append(f"<parameter={key}>\n{content}\n</parameter>")
    return (
        "<tool_call>\n"
        f"<function={name}>\n"
        + "\n".join(parameters)
        + "\n</function>\n"
        "</tool_call>"
    )


async def test_central_water_quality_uses_device_name_and_sensor_fields() -> None:
    client = FakeCubeClient()
    state, runtime, services = await _invoke([
        _search("查询中央水仓水质"),
        _run(_central_water_query()),
        _answer("中央水仓水质查询完成。"),
    ], client)
    result = state["result"]

    assert result.outcome == QueryOutcome.SUCCESS
    assert result.selected_models == ["base_device_info", "base_device_x_value"]
    assert result.semantic_query.filters[0].member == "base_device_info.name"
    assert result.semantic_query.filters[0].values == ["中央水仓"]
    assert result.semantic_query.dimensions == [
        "base_device_info.name",
        "base_device_x_value.sensor_current_value",
        "base_device_x_value.sensor_status",
        "base_device_x_value.updated_at",
    ]
    assert client.events == ["meta", "sql", "load"]
    assert "query_mode" not in client.sql_queries[0]
    assert "models" not in client.sql_queries[0]
    assert client.sql_queries[0] == client.load_queries[0]
    assert services.retrieval_limits == [None]
    assert services.retrieval_questions == ["查询中央水仓水质"]
    assert state["standalone_question"] == "查询中央水仓水质"
    run_observation = runtime.model.messages[2][-1].content
    assert "compiled_sql" not in run_observation
    assert "SELECT" not in run_observation
    run_tool = next(
        tool
        for tools in runtime.model.bound_tools
        for tool in tools
        if tool.name == "run_semantic_query"
    )
    schema = run_tool.tool_call_schema.model_json_schema()
    assert schema["$defs"]["SemanticQuery"]["properties"]["models"]["items"] == {"type": "string"}


async def test_streaming_thoughts_include_semantic_query_and_compiled_sql() -> None:
    stream_outputs: list[dict[str, Any]] = []
    client = FakeCubeClient()
    await _invoke([
        _search("查询中央水仓水质"),
        _run(_central_water_query()),
        _answer(),
    ], client, captured_stream_outputs=stream_outputs)

    thoughts = [
        output["data"]
        for output in stream_outputs
        if output.get("output_type") == "CHAIN_OF_THOUGHT"
    ]
    semantic_thought = next(
        thought for thought in thoughts if thought["text"] == "生成语义查询"
    )
    sql_thought = next(
        thought for thought in thoughts if thought["text"] == "编译语义查询"
    )

    assert '"models": [' in semantic_thought["detail"]
    assert '"base_device_info"' in semantic_thought["detail"]
    assert '"values": [' in semantic_thought["detail"]
    assert "```sql\nSELECT 1 FROM governed_semantic_model\n```" in sql_thought["detail"]


async def test_followup_uses_standalone_question_for_retrieval_and_refresh() -> None:
    standalone_question = "查询中央水仓去年的水质"
    invalid = _query(
        dimensions=["base_device_x_value.unknown_status"],
        ungrouped=True,
    )
    client = FakeCubeClient()
    state, runtime, services = await _invoke(
        [
            json.dumps(
                {"standalone_question": standalone_question},
                ensure_ascii=False,
            ),
            _search(standalone_question),
            _run(invalid),
            _search(f"{standalone_question}\nunknown_member", limit=40),
            _run(_central_water_query()),
            _answer(),
        ],
        client,
        question="那去年呢",
        conversation_context={
            "messages": [
                {"role": "user", "content": "查询中央水仓今年的水质"},
                {"role": "assistant", "content": "已完成查询"},
            ]
        },
    )

    assert state["result"].outcome == QueryOutcome.SUCCESS
    assert state["standalone_question"] == standalone_question
    assert services.retrieval_limits == [None, 40]
    assert services.retrieval_questions[0] == standalone_question
    assert services.retrieval_questions[1].startswith(standalone_question)
    assert state["result"].retrieval_trace.queries[0] == standalone_question
    assert "unknown_member" in state["result"].retrieval_trace.queries[1]
    assert "那去年呢" in str(runtime.model.messages[0][1].content)
    assert "查询中央水仓今年的水质" in str(runtime.model.messages[0][1].content)


async def test_followup_rewrite_failure_falls_back_to_original_question() -> None:
    client = FakeCubeClient()
    state, _, services = await _invoke(
        [
            "not-json",
            _search("那去年呢"),
            _run(_central_water_query()),
            _answer(),
        ],
        client,
        question="那去年呢",
        conversation_context={"previous_question": "查询中央水仓今年的水质"},
    )

    assert state["result"].outcome == QueryOutcome.SUCCESS
    assert state["standalone_question"] == "那去年呢"
    assert services.retrieval_questions == ["那去年呢"]
    assert any("多轮问题改写失败" in warning for warning in state["result"].warnings)


async def test_planner_can_select_a_view_without_selector_routing() -> None:
    query = _query(
        model="hydrology_water_quality_view",
        measures=["hydrology_water_quality_view.sample_count"],
        dimensions=["hydrology_water_quality_view.device_name"],
        ungrouped=False,
    )
    client = FakeCubeClient()
    state, _, _ = await _invoke([
        _search("按设备统计水质样本数"),
        _run(query),
        _answer(),
    ], client, question="按设备统计水质样本数")

    assert state["result"].outcome == QueryOutcome.SUCCESS
    assert state["result"].query_mode.value == "view"
    assert state["result"].selected_models == ["hydrology_water_quality_view"]


async def test_invalid_tool_schema_can_be_corrected_with_same_context() -> None:
    client = FakeCubeClient()
    state, runtime, services = await _invoke([
        _search("查询中央水仓水质"),
        _run({}),
        _run(_central_water_query()),
        _answer(),
    ], client)

    assert state["result"].outcome == QueryOutcome.SUCCESS
    assert state["result"].attempts == 1
    assert services.retrieval_limits == [None]
    assert runtime.model.messages[2][-1].status == "error"
    assert runtime.model.messages[2][-1].name == "run_semantic_query"


async def test_unknown_member_refreshes_top_40_context_then_retries() -> None:
    invalid = _query(
        dimensions=["base_device_x_value.unknown_status"],
        ungrouped=True,
    )
    valid = _query(
        dimensions=["base_device_x_value.sensor_status"],
        ungrouped=True,
    )
    client = FakeCubeClient()
    state, runtime, services = await _invoke([
        _search("查询中央水仓水质"),
        _run(invalid),
        _search("查询中央水仓水质 unknown_member", limit=40),
        _run(valid),
        _answer(),
    ], client)

    assert state["result"].outcome == QueryOutcome.SUCCESS
    assert services.retrieval_limits == [None, 40]
    assert state["result"].retrieval_trace.queries[0] == "查询中央水仓水质"
    assert "unknown_member" in state["result"].retrieval_trace.queries[1]
    assert any(step.stage == "context_refresh" for step in state["result"].steps)
    assert "unknown_member" in str(runtime.model.messages[2])


async def test_multiple_unknown_members_are_refreshed_and_corrected_once() -> None:
    invalid = _query(
        models=["base_device_info", "base_device_x_value"],
        dimensions=[
            "base_device_info.location",
            "base_device_info.status",
        ],
        ungrouped=True,
    )
    client = FakeCubeClient()
    state, _, services = await _invoke([
        _search("查询中央水仓水质"),
        _run(invalid),
        _search(
            "base_device_info.location base_device_info.status unknown_member",
            limit=40,
        ),
        _run(_central_water_query()),
        _answer(),
    ], client)

    assert state["result"].outcome == QueryOutcome.SUCCESS
    assert services.retrieval_limits == [None, 40]
    assert "base_device_info.location" in services.retrieval_questions[1]
    assert "base_device_info.status" in services.retrieval_questions[1]


async def test_cube_400_compilation_error_refreshes_context_and_recompiles() -> None:
    cube_error = CubeClientError(
        "Unknown member in Cube query",
        code="cube_http_error",
        status_code=400,
        retryable_by_model=True,
    )
    query = _query(
        dimensions=["base_device_x_value.sensor_status"],
        ungrouped=True,
    )
    client = FakeCubeClient(sql_failures=[cube_error, None])
    state, runtime, services = await _invoke([
        _search("查询中央水仓水质"),
        _run(query),
        _search("查询中央水仓水质 cube_http_error", limit=40),
        _run(query),
        _answer(),
    ], client)

    assert state["result"].outcome == QueryOutcome.SUCCESS
    assert state["result"].compiled_sql == "SELECT 2 FROM governed_semantic_model"
    assert services.retrieval_limits == [None, 40]
    assert len(client.sql_queries) == 2
    assert len(client.load_queries) == 1
    assert "cube_http_error" in str(runtime.model.messages[2])


@pytest.mark.parametrize(
    "error",
    [
        CubeClientError(
            "Cube 认证失败",
            code="cube_auth_error",
            status_code=401,
        ),
        CubeClientError("Cube 网络请求失败", code="cube_network_error"),
        CubeClientError("连接 Cube 超时", code="cube_timeout"),
    ],
)
async def test_auth_network_and_timeout_errors_do_not_retry_planner(
    error: CubeClientError,
) -> None:
    query = _query(
        dimensions=["base_device_x_value.sensor_status"],
        ungrouped=True,
    )
    client = FakeCubeClient(sql_failures=[error])
    state, runtime, services = await _invoke([
        _search("查询中央水仓水质"),
        _run(query),
        _answer("查询服务暂时不可用。"),
    ], client)

    assert state["result"].outcome == QueryOutcome.EXECUTION_ERROR
    assert state["answer"] == "当前数据查询暂时未能完成，请稍后重试。"
    assert state["result"].attempts == 1
    assert services.retrieval_limits == [None]
    assert len(runtime.model.messages) == 3
    assert not client.load_queries


async def test_llm_timeout_is_system_error_without_retry() -> None:
    client = FakeCubeClient()
    state, runtime, services = await _invoke([TimeoutError("LLM timeout")], client)

    assert state["result"].outcome == QueryOutcome.SYSTEM_ERROR
    assert state["answer"] == "当前查询暂时无法完成。"
    assert state["result"].attempts == 0
    assert services.retrieval_limits == []
    assert len(runtime.model.messages) == 1
    assert not client.sql_queries


async def test_empty_result_returns_no_data_without_context_refresh() -> None:
    query = _query(
        dimensions=["base_device_x_value.sensor_status"],
        ungrouped=True,
    )
    client = FakeCubeClient(load_results=[{"data": [], "annotation": {}}])
    state, runtime, services = await _invoke([
        _search("查询中央水仓水质"),
        _run(query),
        _answer("未查询到符合条件的数据。"),
    ], client)

    assert state["result"].outcome == QueryOutcome.NO_DATA
    assert state["result"].attempts == 1
    assert services.retrieval_limits == [None]
    assert len(runtime.model.messages) == 3
    assert len(client.sql_queries) == 1
    assert len(client.load_queries) == 1
    assert all(step.stage != "context_refresh" for step in state["result"].steps)


async def test_execution_network_failure_preserves_compiled_sql_without_retry() -> None:
    query = _query(
        dimensions=["base_device_x_value.sensor_status"],
        ungrouped=True,
    )
    client = FakeCubeClient(load_results=[
        CubeClientError("Cube 网络请求失败", code="cube_network_error")
    ])
    state, runtime, services = await _invoke([
        _search("查询中央水仓水质"),
        _run(query),
        _answer("查询服务暂时不可用。"),
    ], client)

    assert state["result"].outcome == QueryOutcome.EXECUTION_ERROR
    assert state["answer"] == "当前数据查询暂时未能完成，请稍后重试。"
    assert state["result"].compiled_sql == "SELECT 1 FROM governed_semantic_model"
    assert services.retrieval_limits == [None]
    assert len(runtime.model.messages) == 3


async def test_final_answer_failure_preserves_successful_query_result() -> None:
    client = FakeCubeClient()
    state, _, _ = await _invoke([
        _search("查询中央水仓水质"),
        _run(_central_water_query()),
        TimeoutError("answer timeout"),
    ], client)

    assert state["result"].outcome == QueryOutcome.SUCCESS
    assert state["result"].row_count == 1
    assert state["answer"] == "默认查询报告。"
    assert any("最终回答生成失败" in warning for warning in state["result"].warnings)


async def test_successful_query_uses_dedicated_report_model() -> None:
    client = FakeCubeClient()
    state, runtime, _ = await _invoke([
        _search("查询中央水仓水质"),
        _run(_central_water_query()),
        _answer("Agent 查询结论。"),
    ], client, report_responses=["# 水质分析报告\n\n数据正常。"])

    assert state["answer"] == "# 水质分析报告\n\n数据正常。"
    assert state["result"].presentation.summary == state["answer"]
    assert runtime.report_model.bindings == [{
        "extra_body": {"enable_thinking": False},
    }]
    assert len(runtime.report_model.messages) == 1
    assert "不超过 800 字" in runtime.report_model.messages[0][0].content
    assert "中央水仓" in runtime.report_model.messages[0][1].content


@pytest.mark.parametrize("report_response", ["", TimeoutError("report timeout")])
async def test_dedicated_report_failure_falls_back_to_query_summary(
    report_response: str | Exception,
) -> None:
    client = FakeCubeClient()
    state, runtime, _ = await _invoke([
        _search("查询中央水仓水质"),
        _run(_central_water_query()),
        _answer("Agent 查询结论。"),
    ], client, report_responses=[report_response])

    assert state["answer"] == "查询完成，共返回 1 行数据。"
    assert len(runtime.report_model.messages) == 1
    assert any("Markdown 分析报告生成失败" in warning for warning in state["result"].warnings)


async def test_report_internal_protocol_falls_back_to_query_summary() -> None:
    internal_report = _xml_tool_call(
        "run_semantic_query",
        {"semantic_query": json.loads(_central_water_query())},
    )
    client = FakeCubeClient()
    state, _, _ = await _invoke([
        _search("查询中央水仓水质"),
        _run(_central_water_query()),
        _answer("Agent 查询结论。"),
    ], client, report_responses=[internal_report])

    assert state["answer"] == "查询完成，共返回 1 行数据。"
    assert "<tool_call>" not in state["answer"]
    assert any("Markdown 分析报告生成失败" in warning for warning in state["result"].warnings)


async def test_metadata_filter_limits_validation_catalog_not_just_prompt() -> None:
    query = _query(
        dimensions=["base_device_x_value.sensor_status"],
        ungrouped=True,
    )
    runtime = FakeRuntime([
        _search("查询状态"),
        _run(query),
        _answer("当前查询条件无法映射到可访问模型。"),
    ])
    client = FakeCubeClient()
    settings = HydrologySemanticQuerySettings(
        cube_url="http://cube/cubejs-api/v1",
        cube_token=None,
        timeout_seconds=1,
        continue_wait_retries=0,
        meta_cache_ttl_seconds=60,
        max_retries=1,
        max_rows=10,
        hard_max_rows=100,
        timezone="Asia/Shanghai",
        catalog_mode=SemanticCatalogMode.VECTOR,
        embedding_model=None,
        context_top_k=20,
        vector_index_path=None,
        embedding_batch_size=4,
        embedding_concurrency=1,
        auto_full_context_max_chars=30000,
    )
    services = TrackingServices(settings, client)
    graph = build_hydrology_semantic_query_graph(runtime, services).compile()
    state = await graph.ainvoke({
        "query": "查询状态",
        "metadata": {
            "catalog_metadata_filters": {"model_type": "view"},
        },
    })

    assert state["result"].outcome == QueryOutcome.PLANNER_ERROR
    assert state["answer"] == "当前问题暂时无法转换为有效的数据查询，请调整查询条件后重试。"
    assert state["result"].error.code == "unknown_model"
    assert services.retrieval_limits == [None]
    assert not client.sql_queries


async def test_direct_answer_does_not_create_semantic_query_result() -> None:
    client = FakeCubeClient()
    state, _, services = await _invoke([
        _answer("请提供具体的水文指标和时间范围。"),
    ], client, question="你能做什么")

    assert state["answer"] == "请提供具体的水文指标和时间范围。"
    assert state["result"] is None
    assert "hydrology_semantic_query_result" not in state["metadata"]
    assert services.retrieval_limits == []
    assert client.events == ["meta"]


def test_main_graph_contains_only_react_control_nodes() -> None:
    runtime = FakeRuntime([_answer()])
    client = FakeCubeClient()
    settings = HydrologySemanticQuerySettings(
        cube_url="http://cube/cubejs-api/v1",
        cube_token=None,
        timeout_seconds=1,
        continue_wait_retries=0,
        meta_cache_ttl_seconds=60,
        max_retries=1,
        max_rows=10,
        hard_max_rows=100,
        timezone="Asia/Shanghai",
        embedding_model=None,
        vector_index_path=None,
    )
    graph = build_hydrology_semantic_query_graph(
        runtime,
        TrackingServices(settings, client),
    )

    assert set(graph.nodes) == {"initialize", "agent", "tools", "finalize"}


async def test_run_requires_a_successful_catalog_search_first() -> None:
    client = FakeCubeClient()
    state, runtime, _ = await _invoke([
        _run(_central_water_query()),
        _search("查询中央水仓水质"),
        _run(_central_water_query()),
        _answer(),
    ], client)

    assert state["result"].outcome == QueryOutcome.SUCCESS
    assert state["result"].attempts == 2
    assert "catalog_search_required" in str(runtime.model.messages[1])
    assert client.events == ["meta", "sql", "load"]


async def test_action_input_text_fallback_runs_the_react_loop() -> None:
    semantic_query = _central_water_query()
    client = FakeCubeClient()
    state, _, _ = await _invoke([
        "Action: search_semantic_catalog\nAction Input: {\"query\":\"查询中央水仓水质\"}",
        f"Action: run_semantic_query\nAction Input: {{\"semantic_query\":{semantic_query}}}",
        "查询完成。",
    ], client)

    assert state["result"].outcome == QueryOutcome.SUCCESS
    assert client.events == ["meta", "sql", "load"]
    assert all(
        not message.content
        for message in state["messages"]
        if isinstance(message, AIMessage) and message.tool_calls
    )


async def test_xml_text_fallback_runs_the_react_loop() -> None:
    semantic_query = json.loads(_central_water_query())
    client = FakeCubeClient()
    state, _, _ = await _invoke([
        _xml_tool_call(
            "search_semantic_catalog",
            {"query": "查询中央水仓水质"},
        ),
        _xml_tool_call(
            "run_semantic_query",
            {"semantic_query": semantic_query},
        ),
        _answer(),
    ], client)

    assert state["result"].outcome == QueryOutcome.SUCCESS
    assert client.events == ["meta", "sql", "load"]


async def test_no_data_final_xml_tool_call_uses_safe_answer() -> None:
    query = _query(
        dimensions=["base_device_x_value.sensor_status"],
        ungrouped=True,
    )
    history_query = {
        "query_mode": "view",
        "models": ["view_his_record"],
        "dimensions": ["view_his_record.point_name"],
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
    client = FakeCubeClient(load_results=[{"data": [], "annotation": {}}])
    state, _, _ = await _invoke([
        _search("查询ZL5水温的当前值和历史值"),
        _run(query),
        _xml_tool_call(
            "run_semantic_query",
            {"semantic_query": history_query},
        ),
    ], client, question="查询ZL5水温的当前值和历史值")

    assert state["result"].outcome == QueryOutcome.NO_DATA
    assert state["answer"] == "未查询到符合当前条件的数据。"
    assert state["stream_outputs"] == [{
        "event_type": "INTENT_PROCESSING_PROGRESS",
        "output_type": "LLM_STREAM",
        "data": {
            "text": "未查询到符合当前条件的数据。",
            "progress": None,
        },
    }]
    assert client.events == ["meta", "sql", "load"]
    assert any("内部工具协议" in warning for warning in state["result"].warnings)
    assert all(
        "<tool_call>" not in str(message.content)
        for message in state["messages"]
    )


async def test_no_data_final_native_tool_call_uses_safe_answer() -> None:
    query = _query(
        dimensions=["base_device_x_value.sensor_status"],
        ungrouped=True,
    )
    client = FakeCubeClient(load_results=[{"data": [], "annotation": {}}])
    state, _, _ = await _invoke([
        _search("查询中央水仓水质"),
        _run(query),
        _run(_central_water_query()),
    ], client)

    assert state["result"].outcome == QueryOutcome.NO_DATA
    assert state["answer"] == "未查询到符合当前条件的数据。"
    assert client.events == ["meta", "sql", "load"]


async def test_unknown_native_tool_call_is_not_executed() -> None:
    client = FakeCubeClient()
    unknown_call = AIMessage(content="", tool_calls=[{
        "name": "unknown_tool",
        "args": {},
        "id": "unknown_1",
    }])
    state, _, _ = await _invoke([unknown_call], client, question="你能做什么")

    assert state["result"] is None
    assert state["answer"] == "请提供需要查询的水文数据、范围或条件。"
    assert client.events == ["meta"]


@pytest.mark.parametrize(
    "internal_response",
    [
        "<tool_call><function=run_semantic_query>",
        _central_water_query(),
        '{"kind":"semantic_query_result","rows_truncated":false}',
        "SELECT * FROM dbo.Dev_HisRecord",
    ],
)
async def test_internal_protocol_without_query_uses_safe_clarification(
    internal_response: str,
) -> None:
    client = FakeCubeClient()
    state, _, _ = await _invoke([internal_response], client, question="你能做什么")

    assert state["result"] is None
    assert state["answer"] == "请提供需要查询的水文数据、范围或条件。"
    assert "query_mode" not in state["answer"]


async def test_only_the_first_tool_call_is_executed_per_agent_turn() -> None:
    first = AIMessage(content="", tool_calls=[
        {
            "name": "search_semantic_catalog",
            "args": {"query": "查询中央水仓水质"},
            "id": "search_first",
        },
        {
            "name": "run_semantic_query",
            "args": {"semantic_query": json.loads(_central_water_query())},
            "id": "run_second",
        },
    ])
    client = FakeCubeClient()
    state, _, _ = await _invoke([
        first,
        _run(_central_water_query()),
        _answer(),
    ], client)

    first_agent_message = next(
        message
        for message in state["messages"]
        if isinstance(message, AIMessage) and message.tool_calls
    )
    assert len(first_agent_message.tool_calls) == 1
    assert first_agent_message.tool_calls[0]["name"] == "search_semantic_catalog"
    assert client.events == ["meta", "sql", "load"]


async def test_last_agent_iteration_disables_further_tools() -> None:
    client = FakeCubeClient()
    state, _, services = await _invoke([
        _search("中央水仓"),
        _search("中央水仓水质"),
        _run(_central_water_query()),
    ], client, max_agent_iterations=3)

    assert state["result"] is None
    assert state["metadata"]["agent_iterations"] == 3
    assert services.retrieval_limits == [None, None]
    assert client.events == ["meta"]
