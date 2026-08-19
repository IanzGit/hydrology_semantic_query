from __future__ import annotations

import json
import logging
import sqlite3
from collections.abc import Sequence
from pathlib import Path

import pytest
from langchain_core.messages import AIMessage

from ..config import HydrologySemanticQuerySettings, load_hydrology_semantic_query_settings
from ..graph import build_hydrology_semantic_query_graph
from ..models import (
    CatalogMember,
    CatalogModel,
    QueryMode,
    RetrievalIntent,
    SemanticCatalog,
    SemanticCatalogMode,
    SemanticNeed,
    SemanticQuery,
)
from ..nodes import HydrologySemanticQueryServices, make_execution_node
from ..semantic_catalog_selector import SemanticCatalogSelector
from ..semantic_cube_client import CubeClientError


class KeywordEmbedding:
    model_path = "keyword-test"
    keywords = ("监测", "传感器", "报警", "设备", "酸碱度")

    def __init__(self, *, fail_documents: bool = False, fail_query: bool = False) -> None:
        self.fail_documents = fail_documents
        self.fail_query = fail_query
        self.document_calls = 0
        self.document_batch_sizes: list[int] = []

    async def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
        self.document_calls += 1
        self.document_batch_sizes.append(len(texts))
        if self.fail_documents:
            raise RuntimeError("不应重新生成目录向量")
        return [self._vector(text) for text in texts]

    async def embed_query(self, text: str) -> list[float]:
        if self.fail_query:
            raise RuntimeError("查询向量生成失败")
        return self._vector(text)

    def _vector(self, text: str) -> list[float]:
        values = [float(text.count(keyword)) for keyword in self.keywords]
        return values if any(values) else [0.0] * len(self.keywords)


def _catalog(count: int = 6) -> SemanticCatalog:
    models = {}
    for index in range(count):
        name = f"model_{index}"
        member_name = f"{name}.value"
        models[name] = CatalogModel(
            name=name,
            model_type="view",
            title=f"监测模型 {index}",
            members={
                member_name: CatalogMember(
                    name=member_name,
                    title="监测数量",
                    member_type="measure",
                    data_type="number",
                )
            },
        )
    return SemanticCatalog(models=models)


async def test_index_batches_persists_and_reuses_sqlite_cache(tmp_path: Path) -> None:
    cache_path = tmp_path / "semantic-vectors.sqlite3"
    embedding = KeywordEmbedding()
    selector = SemanticCatalogSelector(
        _catalog(),
        vector_index_path=str(cache_path),
        embedding_client=embedding,
        embedding_batch_size=3,
    )

    assert await selector.prepare() == "built_disk"
    assert embedding.document_batch_sizes == [3, 3, 3, 3]

    cached_embedding = KeywordEmbedding(fail_documents=True)
    cached_selector = SemanticCatalogSelector(
        _catalog(),
        vector_index_path=str(cache_path),
        embedding_client=cached_embedding,
        embedding_batch_size=3,
    )

    assert await cached_selector.prepare() == "disk_cache"
    assert cached_embedding.document_calls == 0


async def test_corrupted_or_old_cache_is_rebuilt_as_v3(tmp_path: Path) -> None:
    cache_path = tmp_path / "semantic-vectors.sqlite3"
    connection = sqlite3.connect(cache_path)
    connection.execute("CREATE TABLE metadata (key TEXT PRIMARY KEY, value TEXT NOT NULL)")
    connection.execute("INSERT INTO metadata VALUES ('version', '2')")
    connection.commit()
    connection.close()
    selector = SemanticCatalogSelector(
        _catalog(1),
        vector_index_path=str(cache_path),
        embedding_client=KeywordEmbedding(),
    )

    assert await selector.prepare() == "built_disk"
    connection = sqlite3.connect(cache_path)
    version = connection.execute(
        "SELECT value FROM metadata WHERE key = 'version'"
    ).fetchone()[0]
    connection.close()
    assert version == "3"


async def test_vector_failure_uses_bounded_batched_catalog_analysis() -> None:
    catalog = _catalog()
    selector = SemanticCatalogSelector(
        catalog,
        view_top_k=2,
        cube_top_k=2,
        member_top_k=3,
        vector_index_path=None,
        embedding_client=KeywordEmbedding(fail_query=True),
        context_member_limit=6,
        catalog_batch_size=2,
    )

    selected = await selector.select(
        "查询监测数量",
        retrieval_intent=RetrievalIntent(needs=[
            SemanticNeed(phrase="监测数量", usage="select", aggregate="count"),
        ]),
    )

    assert selected.warnings
    assert selected.context is not None
    assert len(selected.context.models) == 1
    assert len(selected.context.allowed_members) <= 6


@pytest.mark.parametrize(
    "name",
    [
        "VIEW_TOP_K",
        "CUBE_TOP_K",
        "MEMBER_TOP_K",
        "EMBEDDING_BATCH_SIZE",
        "EMBEDDING_CONCURRENCY",
        "RETRIEVAL_CONCURRENCY",
        "CONTEXT_MEMBER_LIMIT",
        "CATALOG_BATCH_SIZE",
        "MAX_CUBE_MODELS",
    ],
)
def test_catalog_positive_settings(name: str, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(f"HYDROLOGY_SEMANTIC_QUERY_{name}", "0")
    with pytest.raises(ValueError, match=name):
        load_hydrology_semantic_query_settings()


def test_catalog_caps_and_similarity_threshold(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("HYDROLOGY_SEMANTIC_QUERY_CONTEXT_MEMBER_LIMIT", "13")
    with pytest.raises(ValueError, match="CONTEXT_MEMBER_LIMIT"):
        load_hydrology_semantic_query_settings()
    monkeypatch.setenv("HYDROLOGY_SEMANTIC_QUERY_CONTEXT_MEMBER_LIMIT", "12")
    monkeypatch.setenv("HYDROLOGY_SEMANTIC_QUERY_MAX_CUBE_MODELS", "5")
    with pytest.raises(ValueError, match="MAX_CUBE_MODELS"):
        load_hydrology_semantic_query_settings()
    monkeypatch.setenv("HYDROLOGY_SEMANTIC_QUERY_MAX_CUBE_MODELS", "4")
    monkeypatch.setenv("HYDROLOGY_SEMANTIC_QUERY_MEMBER_MATCH_THRESHOLD", "1.1")
    with pytest.raises(ValueError, match="MEMBER_MATCH_THRESHOLD"):
        load_hydrology_semantic_query_settings()


def _meta() -> dict:
    return {
        "cubes": [
            {
                "name": "hydrology_monitoring_devices",
                "type": "view",
                "public": True,
                "title": "水文监测设备情况",
                "meta": {"priority": 1.0, "business_domain": "hydrology"},
                "connectedComponent": 1,
                "measures": [
                    {
                        "name": "hydrology_monitoring_devices.monitoring_sensor_count",
                        "title": "监测传感器数",
                        "type": "number",
                        "public": True,
                    }
                ],
                "dimensions": [],
                "segments": [],
                "folders": [],
                "hierarchies": [],
            },
            {
                "name": "base_device_x_value",
                "type": "cube",
                "public": True,
                "title": "传感器",
                "meta": {"priority": 0.8, "business_domain": "hydrology", "join_edges": []},
                "connectedComponent": 1,
                "measures": [
                    {
                        "name": "base_device_x_value.sensor_count",
                        "title": "监测传感器数",
                        "type": "number",
                        "public": True,
                    }
                ],
                "dimensions": [
                    {
                        "name": "base_device_x_value.id",
                        "title": "传感器ID",
                        "type": "string",
                        "primaryKey": True,
                        "public": True,
                    }
                ],
                "segments": [],
                "folders": [],
                "hierarchies": [],
            },
        ]
    }


def _intent() -> str:
    return json.dumps(
        {
            "needs": [
                {
                    "phrase": "监测传感器",
                    "usage": "select",
                    "aggregate": "count",
                }
            ],
        }
    )


def _query(mode: str = "view") -> str:
    model = (
        "hydrology_monitoring_devices"
        if mode == "view"
        else "base_device_x_value"
    )
    member = (
        "hydrology_monitoring_devices.monitoring_sensor_count"
        if mode == "view"
        else "base_device_x_value.sensor_count"
    )
    return json.dumps(
        {
            "query_mode": mode,
            "models": [model],
            "measures": [member],
            "dimensions": [],
            "segments": [],
            "filters": [],
            "timeDimensions": [],
            "order": [],
            "limit": 10,
            "offset": 0,
            "ungrouped": False,
        }
    )


class FakeModel:
    def __init__(self, responses: list[str]) -> None:
        self.responses = responses
        self.calls = []

    def bind(self, **kwargs):
        return self

    async def ainvoke(self, messages, config=None):
        self.calls.append(messages)
        return AIMessage(content=self.responses.pop(0))


class FakeRuntime:
    def __init__(self, responses: list[str]) -> None:
        self.model = FakeModel(responses)

    def get_chat_model(self, streaming: bool):
        return self.model


class FakeCubeClient:
    def __init__(self, *, empty_first: bool = False) -> None:
        self.empty_first = empty_first
        self.loads = []
        self.sql_queries = []

    async def get_meta(self) -> dict:
        return _meta()

    async def load(self, query: dict) -> dict:
        self.loads.append(query)
        member = query["measures"][0]
        rows = [] if self.empty_first and len(self.loads) == 1 else [{member: "1"}]
        return {
            "data": rows,
            "annotation": {
                "measures": {member: {"title": "数量", "type": "number"}}
            },
        }

    async def get_sql(self, query: dict) -> tuple[str, list]:
        self.sql_queries.append(query)
        return "SELECT count(*) FROM device_x_value", []


class AuthFailingCubeClient(FakeCubeClient):
    async def load(self, query: dict) -> dict:
        self.loads.append(query)
        raise CubeClientError(
            "Cube 认证失败",
            code="cube_auth_error",
            status_code=401,
        )


class SqlFailingCubeClient(FakeCubeClient):
    async def get_sql(self, query: dict) -> tuple[str, list]:
        self.sql_queries.append(query)
        raise CubeClientError("Cube /sql 调用失败", code="cube_network_error")


def _settings(
    *,
    max_retries: int = 0,
    retry_on_empty_result: bool = True,
) -> HydrologySemanticQuerySettings:
    return HydrologySemanticQuerySettings(
        cube_url="http://cube",
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
        view_top_k=1,
        cube_top_k=1,
        member_top_k=4,
        vector_index_path=None,
        retry_on_empty_result=retry_on_empty_result,
    )


async def _invoke(
    responses: list[str],
    client: FakeCubeClient,
    *,
    max_retries: int = 0,
    retry_on_empty_result: bool = True,
    question: str = "查询监测传感器数",
) -> tuple[dict, FakeRuntime]:
    runtime = FakeRuntime(responses)
    services = HydrologySemanticQueryServices(
        _settings(
            max_retries=max_retries,
            retry_on_empty_result=retry_on_empty_result,
        ),
        client=client,
        embedding_client=KeywordEmbedding(),
    )
    graph = build_hydrology_semantic_query_graph(runtime, services).compile()
    state = await graph.ainvoke(
        {"query": question, "metadata": {"report": False}}
    )
    return state, runtime


async def test_graph_view_fast_path_executes_without_sending_control_fields(caplog) -> None:
    client = FakeCubeClient()
    with caplog.at_level(logging.INFO, logger="uvicorn.error"):
        state, runtime = await _invoke([_intent(), _query()], client)

    result = state["result"]
    assert result.success is True
    assert result.query_mode == QueryMode.VIEW
    assert result.retrieval_trace.fallback_level == 0
    assert len(runtime.model.calls) == 2
    assert "query_mode" not in client.loads[0]
    assert "models" not in client.loads[0]
    assert "generated sql" in caplog.text


async def test_execution_stream_contains_generated_sql() -> None:
    client = FakeCubeClient()
    services = HydrologySemanticQueryServices(_settings(), client=client)
    node = make_execution_node(services)
    query = SemanticQuery.model_validate(json.loads(_query()))

    update = await node({
        "semantic_query": query,
        "steps": [],
        "warnings": [],
        "attempts": 1,
        "max_attempts": 1,
        "catalog_mode": SemanticCatalogMode.VECTOR,
    })

    assert "SELECT count(*)" in json.dumps(
        update["stream_outputs"], ensure_ascii=False
    )


async def test_empty_result_retries_in_cube_component_mode() -> None:
    client = FakeCubeClient(empty_first=True)
    state, runtime = await _invoke(
        [_intent(), _query(), _query("cube")],
        client,
        max_retries=1,
    )

    result = state["result"]
    assert result.success is True
    assert result.query_mode == QueryMode.CUBE
    assert result.retrieval_trace.fallback_level == 2
    assert len(runtime.model.calls) == 3
    assert len(client.loads) == 2


async def test_generation_failure_retries_with_same_bounded_context() -> None:
    state, runtime = await _invoke(
        [_intent(), "invalid", _query()],
        FakeCubeClient(),
        max_retries=1,
    )

    assert state["result"].success is True
    assert len(runtime.model.calls) == 3
    assert not any(step.stage == "catalog_linking_retry" for step in state["result"].steps)


async def test_semantic_model_gap_stops_before_query_generation() -> None:
    intent = json.dumps(
        {
            "needs": [
                {
                    "phrase": "酸碱度",
                    "usage": "select",
                    "aggregate": None,
                }
            ],
        }
    )
    state, runtime = await _invoke(
        [intent],
        FakeCubeClient(),
        question="查询水质酸碱度",
    )

    result = state["result"]
    assert result.success is False
    assert result.error.code == "semantic_model_gap"
    assert result.semantic_model_gap.missing_concepts == ["酸碱度"]
    assert len(runtime.model.calls) == 1


async def test_auth_error_does_not_trigger_catalog_fallback() -> None:
    state, _ = await _invoke([_intent(), _query()], AuthFailingCubeClient())

    result = state["result"]
    assert result.success is False
    assert result.error.code == "cube_auth_error"
    assert not any(step.stage == "catalog_linking_retry" for step in result.steps)


async def test_sql_failure_preserves_rows_and_returns_warning() -> None:
    state, _ = await _invoke([_intent(), _query()], SqlFailingCubeClient())

    result = state["result"]
    assert result.success is True
    assert result.row_count == 1
    assert "Cube 调试 SQL 获取失败" in result.warnings[0]
