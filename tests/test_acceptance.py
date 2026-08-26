from __future__ import annotations

import json
from typing import Any

import pytest
from langchain_core.messages import AIMessage

from ..config import HydrologySemanticQuerySettings
from ..graph import build_hydrology_semantic_query_graph
from ..models import QueryOutcome, SemanticCatalogMode
from ..nodes import HydrologySemanticQueryServices
from ..semantic_cube_client import CubeClientError


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
    def __init__(self, responses: list[str | Exception]) -> None:
        self.responses = list(responses)
        self.messages: list[list[Any]] = []
        self.bindings: list[dict[str, Any]] = []

    def bind(self, **kwargs: Any) -> FakeModel:
        self.bindings.append(kwargs)
        return self

    async def ainvoke(self, messages: list[Any], config: Any = None) -> AIMessage:
        del config
        self.messages.append(messages)
        response = self.responses.pop(0)
        if isinstance(response, Exception):
            raise response
        return AIMessage(content=response)


class FakeRuntime:
    def __init__(self, responses: list[str | Exception]) -> None:
        self.model = FakeModel(responses)

    def get_chat_model(self, streaming: bool) -> FakeModel:
        assert streaming is False
        return self.model


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
        self.retrieval_limits: list[int | None] = []

    async def retrieve_context(self, *args: Any, **kwargs: Any):
        self.retrieval_limits.append(kwargs.get("limit"))
        return await super().retrieve_context(*args, **kwargs)


async def _invoke(
    responses: list[str | Exception],
    client: FakeCubeClient,
    *,
    question: str = "查询中央水仓水质",
    max_retries: int = 1,
) -> tuple[dict[str, Any], FakeRuntime, TrackingServices]:
    runtime = FakeRuntime(responses)
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
        enable_report=False,
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
        "query": question,
        "metadata": {"report": False, "business_knowledge": "中央水仓是设备名称"},
    })
    return state, runtime, services


async def test_central_water_quality_uses_device_name_and_sensor_fields() -> None:
    client = FakeCubeClient()
    state, runtime, services = await _invoke([_central_water_query()], client)
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
    schema = runtime.model.bindings[0]["response_format"]["json_schema"]["schema"]
    assert schema["properties"]["models"]["items"] == {"type": "string"}


async def test_planner_can_select_a_view_without_selector_routing() -> None:
    query = _query(
        model="hydrology_water_quality_view",
        measures=["hydrology_water_quality_view.sample_count"],
        dimensions=["hydrology_water_quality_view.device_name"],
        ungrouped=False,
    )
    client = FakeCubeClient()
    state, _, _ = await _invoke([query], client, question="按设备统计水质样本数")

    assert state["result"].outcome == QueryOutcome.SUCCESS
    assert state["result"].query_mode.value == "view"
    assert state["result"].selected_models == ["hydrology_water_quality_view"]


async def test_json_error_retries_with_same_context_and_structured_feedback() -> None:
    client = FakeCubeClient()
    state, runtime, services = await _invoke(["{invalid", _central_water_query()], client)

    assert state["result"].outcome == QueryOutcome.SUCCESS
    assert state["result"].attempts == 2
    assert services.retrieval_limits == [None]
    retry_prompt = str(runtime.model.messages[1][1].content)
    assert "json_syntax_error" in retry_prompt
    assert "raw_response_excerpt" in retry_prompt


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
    state, runtime, services = await _invoke([invalid, valid], client)

    assert state["result"].outcome == QueryOutcome.SUCCESS
    assert services.retrieval_limits == [None, 40]
    assert state["result"].retrieval_trace.queries[0] == "查询中央水仓水质"
    assert "unknown_member" in state["result"].retrieval_trace.queries[1]
    assert any(step.stage == "context_refresh" for step in state["result"].steps)
    assert "unknown_member" in str(runtime.model.messages[1][1].content)


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
    state, runtime, services = await _invoke([query, query], client)

    assert state["result"].outcome == QueryOutcome.SUCCESS
    assert state["result"].compiled_sql == "SELECT 2 FROM governed_semantic_model"
    assert services.retrieval_limits == [None, 40]
    assert len(client.sql_queries) == 2
    assert len(client.load_queries) == 1
    assert "cube_http_error" in str(runtime.model.messages[1][1].content)


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
    state, runtime, services = await _invoke([query], client)

    assert state["result"].outcome == QueryOutcome.EXECUTION_ERROR
    assert state["result"].attempts == 1
    assert services.retrieval_limits == [None]
    assert len(runtime.model.messages) == 1
    assert not client.load_queries


async def test_llm_timeout_is_system_error_without_retry() -> None:
    client = FakeCubeClient()
    state, runtime, services = await _invoke([TimeoutError("LLM timeout")], client)

    assert state["result"].outcome == QueryOutcome.SYSTEM_ERROR
    assert state["result"].attempts == 1
    assert services.retrieval_limits == [None]
    assert len(runtime.model.messages) == 1
    assert not client.sql_queries


async def test_empty_result_returns_no_data_without_context_refresh() -> None:
    query = _query(
        dimensions=["base_device_x_value.sensor_status"],
        ungrouped=True,
    )
    client = FakeCubeClient(load_results=[{"data": [], "annotation": {}}])
    state, runtime, services = await _invoke([query], client)

    assert state["result"].outcome == QueryOutcome.NO_DATA
    assert state["result"].attempts == 1
    assert services.retrieval_limits == [None]
    assert len(runtime.model.messages) == 1
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
    state, runtime, services = await _invoke([query], client)

    assert state["result"].outcome == QueryOutcome.EXECUTION_ERROR
    assert state["result"].compiled_sql == "SELECT 1 FROM governed_semantic_model"
    assert services.retrieval_limits == [None]
    assert len(runtime.model.messages) == 1


async def test_metadata_filter_limits_validation_catalog_not_just_prompt() -> None:
    query = _query(
        dimensions=["base_device_x_value.sensor_status"],
        ungrouped=True,
    )
    runtime = FakeRuntime([query, query])
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
        enable_report=False,
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
            "report": False,
            "catalog_metadata_filters": {"model_type": "view"},
        },
    })

    assert state["result"].outcome == QueryOutcome.PLANNER_ERROR
    assert state["result"].error.code == "unknown_model"
    assert services.retrieval_limits == [None, 40]
    assert not client.sql_queries
