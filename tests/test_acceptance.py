from __future__ import annotations

import json
from collections.abc import Sequence

from langchain_core.messages import AIMessage

from ..config import HydrologySemanticQuerySettings
from ..graph import build_hydrology_semantic_query_graph
from ..models import (
    CatalogMember,
    CatalogModel,
    QueryMode,
    QueryOutcome,
    RetrievalIntent,
    SemanticCatalog,
    SemanticCatalogMode,
    SemanticNeed,
    SemanticQuery,
)
from ..nodes import HydrologySemanticQueryServices
from ..semantic_catalog_selector import SemanticCatalogSelector
from ..semantic_cube_client import CubeClientError
from ..semantic_query_validator import validate_semantic_query


class KeywordEmbedding:
    model_path = "keyword-test"
    keywords = ("报警", "设备", "传感器", "阈值", "名称", "数量")

    def __init__(self) -> None:
        self.query_calls = 0

    async def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
        return [self._vector(text) for text in texts]

    async def embed_query(self, text: str) -> list[float]:
        self.query_calls += 1
        return self._vector(text)

    def _vector(self, text: str) -> list[float]:
        values = [float(text.count(keyword)) for keyword in self.keywords]
        return values if any(values) else [0.1] * len(self.keywords)


def _member(
    model: str,
    name: str,
    title: str,
    *,
    member_type: str = "dimension",
    data_type: str = "string",
) -> CatalogMember:
    return CatalogMember(
        name=f"{model}.{name}",
        title=title,
        member_type=member_type,
        data_type=data_type,
    )


def _model(
    name: str,
    model_type: str,
    title: str,
    members: Sequence[CatalogMember],
    *,
    joins: tuple[str, ...] = (),
    description: str = "",
) -> CatalogModel:
    return CatalogModel(
        name=name,
        model_type=model_type,
        title=title,
        description=description,
        members={member.name: member for member in members},
        join_edges=joins,
        connected_component=1,
    )


def _small_catalog() -> SemanticCatalog:
    view = _model(
        "hydrology_device_view",
        "view",
        "设备快捷场景",
        [
            _member("hydrology_device_view", "device_name", "设备名称"),
            _member(
                "hydrology_device_view",
                "sensor_count",
                "传感器数量",
                member_type="measure",
                data_type="number",
            ),
        ],
    )
    return SemanticCatalog(models={view.name: view})


def _large_catalog() -> SemanticCatalog:
    filler = "这是一段较长的业务描述文本，用于撑大语义目录的字符估算。"
    models = {}
    for index in range(12):
        name = f"base_cube_{index}"
        members = [
            _member(
                name,
                f"member_{member_index}",
                f"业务成员 {member_index} 的完整标题说明",
                member_type=(
                    "measure" if member_index % 2 == 0 else "dimension"
                ),
                data_type="number" if member_index % 2 == 0 else "string",
            )
            for member_index in range(10)
        ]
        models[name] = _model(
            name,
            "cube",
            f"原子实体 {index}",
            members,
            description=filler,
        )
    return SemanticCatalog(models=models)


def _mixed_catalog() -> SemanticCatalog:
    view = _model(
        "hydrology_device_view",
        "view",
        "设备快捷场景",
        [
            _member("hydrology_device_view", "device_name", "设备名称"),
        ],
    )
    filler = "这是一段较长的业务描述文本，用于撑大语义目录的字符估算。"
    cubes = {}
    for index in range(10):
        name = f"base_cube_{index}"
        cubes[name] = _model(
            name,
            "cube",
            f"原子实体 {index}",
            [
                _member(
                    name,
                    f"member_{member_index}",
                    f"业务成员 {member_index} 的完整标题说明",
                )
                for member_index in range(10)
            ],
            description=filler,
        )
    return SemanticCatalog(models={view.name: view, **cubes})


async def test_auto_small_catalog_selects_full_without_embedding() -> None:
    embedding = KeywordEmbedding()
    selector = SemanticCatalogSelector(
        _small_catalog(),
        vector_index_path=None,
        embedding_client=embedding,
    )

    selected = await selector.select(
        "查询设备名称",
        retrieval_intent=RetrievalIntent(needs=[
            SemanticNeed(phrase="设备名称", usage="select"),
        ]),
    )

    assert selected.mode == SemanticCatalogMode.FULL
    assert selected.context is not None
    assert embedding.query_calls == 0
    assert selected.context.candidate_models == ["hydrology_device_view"]
    assert len(selected.context.allowed_members) == 2


async def test_auto_large_catalog_selects_vector() -> None:
    embedding = KeywordEmbedding()
    selector = SemanticCatalogSelector(
        _large_catalog(),
        vector_index_path=None,
        embedding_client=embedding,
    )

    selected = await selector.select(
        "查询传感器数量",
        retrieval_intent=RetrievalIntent(needs=[
            SemanticNeed(phrase="传感器数量", usage="select", aggregate="count"),
        ]),
    )

    assert selected.mode == SemanticCatalogMode.VECTOR
    assert embedding.query_calls >= 1
    assert selected.context is not None
    assert len(selected.context.candidate_models) < 12


async def test_explicit_full_overrides_large_catalog_size() -> None:
    embedding = KeywordEmbedding()
    selector = SemanticCatalogSelector(
        _large_catalog(),
        vector_index_path=None,
        embedding_client=embedding,
    )

    selected = await selector.select(
        "查询传感器数量",
        retrieval_intent=RetrievalIntent(needs=[
            SemanticNeed(phrase="传感器数量", usage="select", aggregate="count"),
        ]),
        mode=SemanticCatalogMode.FULL,
    )

    assert selected.mode == SemanticCatalogMode.FULL
    assert embedding.query_calls == 0
    assert selected.context is not None
    assert len(selected.context.candidate_models) == 12


async def test_explicit_vector_overrides_small_catalog_size() -> None:
    embedding = KeywordEmbedding()
    selector = SemanticCatalogSelector(
        _small_catalog(),
        vector_index_path=None,
        embedding_client=embedding,
    )

    selected = await selector.select(
        "查询设备名称",
        retrieval_intent=RetrievalIntent(needs=[
            SemanticNeed(phrase="设备名称", usage="select"),
        ]),
        mode=SemanticCatalogMode.VECTOR,
    )

    assert selected.mode == SemanticCatalogMode.VECTOR
    assert embedding.query_calls >= 1


async def test_auto_after_metadata_filter_shrinks_to_full() -> None:
    embedding = KeywordEmbedding()
    selector = SemanticCatalogSelector(
        _mixed_catalog(),
        vector_index_path=None,
        embedding_client=embedding,
    )

    selected = await selector.select(
        "查询设备名称",
        retrieval_intent=RetrievalIntent(needs=[
            SemanticNeed(phrase="设备名称", usage="select"),
        ]),
        metadata_filters={"model_type": "view"},
    )

    assert selected.mode == SemanticCatalogMode.FULL
    assert embedding.query_calls == 0
    assert selected.context is not None
    assert selected.context.candidate_models == ["hydrology_device_view"]


async def test_non_top1_binding_member_passes_validation() -> None:
    embedding = KeywordEmbedding()
    selector = SemanticCatalogSelector(
        _small_catalog(),
        view_top_k=2,
        cube_top_k=2,
        member_top_k=4,
        vector_index_path=None,
        embedding_client=embedding,
    )

    selected = await selector.select(
        "查询设备名称和传感器数量",
        retrieval_intent=RetrievalIntent(needs=[
            SemanticNeed(phrase="设备名称", usage="select"),
            SemanticNeed(phrase="传感器数量", usage="select", aggregate="count"),
        ]),
        mode=SemanticCatalogMode.VECTOR,
    )

    assert selected.context is not None
    assert selected.context.binding_candidates
    top1 = selected.context.binding_candidates["select:设备名称"][0].member
    assert top1 == "hydrology_device_view.device_name"
    query = SemanticQuery(
        query_mode=QueryMode.VIEW,
        models=["hydrology_device_view"],
        measures=["hydrology_device_view.sensor_count"],
        dimensions=["hydrology_device_view.device_name"],
    )
    validated = validate_semantic_query(
        query,
        selected.catalog,
        requested_max_rows=50,
        hard_max_rows=1000,
    )
    assert validated.query.dimensions == ["hydrology_device_view.device_name"]


async def test_cube_join_middle_model_is_added_to_candidates() -> None:
    left = "base_alarm_event"
    middle = "base_alarm_record"
    right = "base_device"
    catalog = SemanticCatalog(models={
        left: _model(
            left,
            "cube",
            "报警事件",
            [
                _member(left, "alarm_count", "报警数", member_type="measure", data_type="number"),
                _member(left, "device_id", "设备ID"),
            ],
            joins=(middle,),
        ),
        middle: _model(
            middle,
            "cube",
            "报警记录",
            [_member(middle, "id", "记录ID")],
            joins=(right,),
        ),
        right: _model(
            right,
            "cube",
            "设备原子实体",
            [_member(right, "name", "设备名称")],
        ),
    })
    selector = SemanticCatalogSelector(
        catalog,
        view_top_k=2,
        cube_top_k=2,
        member_top_k=6,
        vector_index_path=None,
        embedding_client=KeywordEmbedding(),
    )

    selected = await selector.select(
        "查询报警事件与设备信息",
        retrieval_intent=RetrievalIntent(needs=[
            SemanticNeed(phrase="报警数", usage="select", aggregate="count"),
            SemanticNeed(phrase="设备名称", usage="select"),
        ]),
        mode=SemanticCatalogMode.VECTOR,
    )

    assert selected.gap is None
    assert selected.context is not None
    assert middle in selected.context.candidate_models
    assert {"base_alarm_event", "base_device"}.issubset(
        selected.context.candidate_models
    )


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
                "meta": {"priority": 0.8, "business_domain": "hydrology"},
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
    return json.dumps({
        "needs": [
            {"phrase": "监测传感器", "usage": "select", "aggregate": "count"},
        ],
    })


def _query(model: str = "hydrology_monitoring_devices") -> str:
    member = (
        "hydrology_monitoring_devices.monitoring_sensor_count"
        if model == "hydrology_monitoring_devices"
        else "base_device_x_value.sensor_count"
    )
    return json.dumps({
        "query_mode": "view" if model == "hydrology_monitoring_devices" else "cube",
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
    })


class FakeModel:
    def __init__(self, responses: list[str]) -> None:
        self.responses = responses

    def bind(self, **kwargs):
        return self

    async def ainvoke(self, messages, config=None):
        return AIMessage(content=self.responses.pop(0))


class FakeRuntime:
    def __init__(self, responses: list[str]) -> None:
        self.model = FakeModel(responses)

    def get_chat_model(self, streaming: bool):
        return self.model


class FakeCubeClient:
    def __init__(self) -> None:
        self.loads: list[dict] = []
        self.sql_calls = 0

    async def get_meta(self) -> dict:
        return _meta()

    async def load(self, query: dict) -> dict:
        self.loads.append(query)
        member = query["measures"][0]
        return {
            "data": [{member: "1"}],
            "annotation": {
                "measures": {member: {"title": "数量", "type": "number"}}
            },
        }

    async def get_sql(self, query: dict) -> tuple[str, list]:
        self.sql_calls += 1
        return f"SELECT {self.sql_calls} FROM device_x_value", []


class LoadFailingCubeClient(FakeCubeClient):
    def __init__(self, *, fail_first: bool = False) -> None:
        super().__init__()
        self.fail_first = fail_first

    async def load(self, query: dict) -> dict:
        self.loads.append(query)
        if self.fail_first and len(self.loads) == 1:
            raise CubeClientError(
                "Cube 执行失败",
                code="cube_execution_error",
                retryable_by_model=True,
            )
        member = query["measures"][0]
        return {
            "data": [{member: "1"}],
            "annotation": {
                "measures": {member: {"title": "数量", "type": "number"}}
            },
        }


async def _invoke(
    responses: list[str],
    client: FakeCubeClient,
    *,
    max_retries: int = 1,
) -> tuple[dict, FakeRuntime]:
    runtime = FakeRuntime(responses)
    services = HydrologySemanticQueryServices(
        HydrologySemanticQuerySettings(
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
            vector_index_path=None,
        ),
        client=client,
        embedding_client=KeywordEmbedding(),
    )
    graph = build_hydrology_semantic_query_graph(runtime, services).compile()
    state = await graph.ainvoke(
        {"query": "查询监测传感器数", "metadata": {"report": False}}
    )
    return state, runtime


async def test_execution_failure_preserves_compiled_sql() -> None:
    client = LoadFailingCubeClient(fail_first=True)
    state, _ = await _invoke([_intent(), _query()], client, max_retries=0)

    result = state["result"]
    assert result.outcome == QueryOutcome.EXECUTION_ERROR
    assert result.compiled_sql == "SELECT 1 FROM device_x_value"
    assert len(client.loads) == 1


async def test_retry_clears_old_compiled_artifacts() -> None:
    client = LoadFailingCubeClient(fail_first=True)
    state, _ = await _invoke([_intent(), _query(), _query("base_device_x_value")], client)

    result = state["result"]
    assert result.outcome == QueryOutcome.SUCCESS
    assert result.attempts == 2
    assert result.compiled_sql == "SELECT 2 FROM device_x_value"
    assert len(client.loads) == 2
