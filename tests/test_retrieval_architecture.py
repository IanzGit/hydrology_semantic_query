from __future__ import annotations

import json
from collections.abc import Sequence
from pathlib import Path

import pytest

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
from ..semantic_catalog import catalog_from_meta
from ..semantic_catalog_selector import SemanticCatalogSelector
from ..semantic_context import SemanticJoinGraph, context_for_prompt
from ..semantic_query_planner import parse_retrieval_intent
from ..semantic_query_validator import (
    SemanticQueryValidationError,
    validate_semantic_query,
)


class KeywordEmbedding:
    model_path = "keyword-test"
    keywords = (
        "报警",
        "设备",
        "阈值",
        "标签",
        "风险",
        "酸碱度",
        "配置",
        "事件",
        "指标",
    )

    async def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
        return [self._vector(text) for text in texts]

    async def embed_query(self, text: str) -> list[float]:
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
    component: int = 1,
    priority: float = 0.5,
) -> CatalogModel:
    return CatalogModel(
        name=name,
        model_type=model_type,
        title=title,
        members={member.name: member for member in members},
        join_edges=joins,
        connected_component=component,
        business_priority=priority,
    )


def _catalog() -> SemanticCatalog:
    alarm_view = "hydrology_alarm_view"
    device_view = "hydrology_device_view"
    event = "base_event"
    sensor = "base_sensor"
    device = "base_device"
    label = "base_label"
    models = {
        alarm_view: _model(
            alarm_view,
            "view",
            "报警快捷场景",
            [
                _member(alarm_view, "alarm_count", "报警数", member_type="measure", data_type="number"),
                _member(alarm_view, "device_name", "设备名称"),
            ],
            priority=1.0,
        ),
        device_view: _model(
            device_view,
            "view",
            "设备快捷场景",
            [_member(device_view, "device_name", "设备名称")],
            priority=0.9,
        ),
        event: _model(
            event,
            "cube",
            "报警事件",
            [
                _member(event, "alarm_count", "报警数", member_type="measure", data_type="number"),
                _member(event, "sensor_id", "传感器ID"),
            ],
            joins=(sensor,),
        ),
        sensor: _model(
            sensor,
            "cube",
            "传感器原子实体",
            [
                _member(sensor, "id", "传感器ID"),
                _member(sensor, "device_id", "设备ID"),
                _member(sensor, "maximum_threshold", "最大报警阈值"),
            ],
            joins=(device,),
        ),
        device: _model(
            device,
            "cube",
            "设备原子实体",
            [
                _member(device, "id", "设备ID"),
                _member(device, "name", "设备名称"),
            ],
        ),
        label: _model(
            label,
            "cube",
            "标签原子实体",
            [_member(label, "name", "标签名称")],
            component=2,
        ),
    }
    return SemanticCatalog(models=models)


def test_catalog_consumes_meta_type_rich_metadata_and_governed_join_edges() -> None:
    payload = {
        "cubes": [
            {
                "name": "hydrology_monitoring_devices",
                "type": "view",
                "public": True,
                "title": "监测设备",
                "connectedComponent": 1,
                "folders": [{"name": "状态字段", "members": ["sensor_state"]}],
                "hierarchies": [{"name": "设备层级"}],
                "meta": {
                    "aliases": ["监测点"],
                    "use_cases": ["查询设备状态"],
                    "priority": 0.9,
                    "business_domain": "hydrology",
                },
                "measures": [],
                "dimensions": [
                    {
                        "name": "hydrology_monitoring_devices.sensor_state",
                        "title": "传感器状态",
                        "type": "string",
                        "public": True,
                    }
                ],
                "segments": [],
            },
            {
                "name": "base_device_x_value",
                "type": "cube",
                "public": True,
                "connectedComponent": 1,
                "meta": {
                    "join_edges": [
                        {"target": "base_device_info", "relationship": "many_to_one"}
                    ]
                },
                "measures": [],
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
            },
            {
                "name": "base_private",
                "type": "cube",
                "public": False,
                "measures": [],
                "dimensions": [],
                "segments": [],
            },
        ]
    }

    catalog = catalog_from_meta(payload)

    view = catalog.models["hydrology_monitoring_devices"]
    cube = catalog.models["base_device_x_value"]
    assert view.model_type == "view"
    assert view.aliases == ("监测点",)
    assert view.use_cases == ("查询设备状态",)
    assert view.folders == ("状态字段",)
    assert view.hierarchies == ("设备层级",)
    assert view.business_priority == 0.9
    assert cube.join_edges == ("base_device_info",)
    assert cube.members["base_device_x_value.id"].primary_key is True
    assert "base_private" not in catalog.models


def test_real_cube_1_6_70_meta_fixture_exposes_three_views_and_seven_cubes() -> None:
    fixture = Path(__file__).with_name("fixtures") / "cube_meta_1_6_70.json"
    catalog = catalog_from_meta(json.loads(fixture.read_text(encoding="utf-8")))

    assert sum(model.model_type == "view" for model in catalog.models.values()) == 3
    assert sum(model.model_type == "cube" for model in catalog.models.values()) == 7
    assert catalog.models["base_warn_state_info"].join_edges == (
        "base_device_x_value",
        "base_water_warn_sensor_set",
    )
    assert catalog.models["base_multifactor_sensor"].members[
        "base_multifactor_sensor.multifactor_indicator_count"
    ].title == "多因素指标数"
    assert "base_water_warn_sensor_set.prediction_function" not in catalog.models[
        "base_water_warn_sensor_set"
    ].members


def test_join_graph_returns_shortest_path_and_rejects_disconnected_models() -> None:
    graph = SemanticJoinGraph.from_catalog(_catalog())

    assert graph.shortest_path("base_event", "base_device") == [
        "base_event",
        "base_sensor",
        "base_device",
    ]
    assert graph.minimal_subgraph(["base_event", "base_device"]) == (
        ["base_event", "base_sensor", "base_device"],
        [["base_event", "base_sensor", "base_device"]],
    )
    assert graph.shortest_path("base_event", "base_label") is None


def test_retrieval_intent_accepts_needs_only() -> None:
    intent = RetrievalIntent.model_validate({
        "needs": [
            {"phrase": "设备", "usage": "select", "aggregate": "count"},
            {"phrase": "启用设备", "usage": "filter", "aggregate": None},
        ]
    })

    assert intent.needs == [
        SemanticNeed(phrase="设备", usage="select", aggregate="count"),
        SemanticNeed(phrase="启用设备", usage="filter"),
    ]
    with pytest.raises(ValueError):
        RetrievalIntent.model_validate({"subjects": ["设备"]})


def test_parse_retrieval_intent_keeps_business_phrase_not_operation_word() -> None:
    intent = parse_retrieval_intent(json.dumps({
        "needs": [
            {"phrase": "设备", "usage": "select", "aggregate": "count"},
            {"phrase": "启用设备", "usage": "filter", "aggregate": None},
        ]
    }))

    assert [need.phrase for need in intent.needs] == ["设备", "启用设备"]
    assert all(need.phrase != "数量" for need in intent.needs)


async def test_scope_only_query_keeps_model_default_projection() -> None:
    view_name = "hydrology_multifactor_warnings"
    catalog = SemanticCatalog(models={
        view_name: _model(
            view_name,
            "view",
            "水文多因素预警情况",
            [
                _member(view_name, "warning_event_id", "预警事件ID"),
                _member(view_name, "warning_name", "预警名称"),
            ],
        ).model_copy(update={"aliases": ("多因素预警",)}),
    })
    selected = await SemanticCatalogSelector(
        catalog,
        view_top_k=1,
        cube_top_k=1,
        vector_index_path=None,
        embedding_client=None,
    ).select(
        "请查询当前水文多因素预警的具体情况",
        retrieval_intent=RetrievalIntent(),
    )

    assert selected.gap is None
    assert selected.selected_models == [view_name]
    assert selected.context is not None
    assert selected.context.projection_policy == "model_default"
    assert "retrieval_intent" in context_for_prompt(selected.context)


async def test_need_binding_uses_contextual_need_and_full_query_evidence() -> None:
    device = "base_device_info"
    monitor = "hydrology_monitoring_devices"
    catalog = SemanticCatalog(models={
        monitor: _model(
            monitor,
            "view",
            "监测设备情况",
            [
                _member(
                    monitor,
                    "monitoring_sensor_count",
                    "监测传感器数",
                    member_type="measure",
                    data_type="number",
                ),
            ],
        ),
        device: _model(
            device,
            "cube",
            "设备原子实体",
            [
                _member(
                    device,
                    "device_count",
                    "设备数",
                    member_type="measure",
                    data_type="number",
                ),
                _member(
                    device,
                    "enabled",
                    "已启用设备",
                    member_type="segment",
                ),
            ],
        ),
    })
    selected = await SemanticCatalogSelector(
        catalog,
        view_top_k=1,
        cube_top_k=1,
        member_top_k=4,
        vector_index_path=None,
        embedding_client=KeywordEmbedding(),
    ).select(
        "查询已启用设备数量",
        retrieval_intent=RetrievalIntent(needs=[
            SemanticNeed(phrase="设备", usage="select", aggregate="count"),
            SemanticNeed(phrase="启用设备", usage="filter"),
        ]),
        mode=SemanticCatalogMode.VECTOR,
    )

    assert selected.gap is None
    assert selected.context is not None
    assert device in selected.context.candidate_models
    assert selected.trace.member_hits
    assert selected.trace.binding_candidates["select:设备:count"][0].member == (
        f"{device}.device_count"
    )
    assert selected.trace.binding_candidates["filter:启用设备"][0].member == (
        f"{device}.enabled"
    )
    assert selected.trace.binding_scores["select:设备:count"][
        f"{device}.device_count"
    ] > selected.trace.binding_scores["select:设备:count"][
        f"{monitor}.monitoring_sensor_count"
    ]
    assert f"{device}.enabled" in selected.context.allowed_members


async def test_full_query_hit_is_not_vetoed_by_second_lookup() -> None:
    selector = SemanticCatalogSelector(
        _catalog(),
        view_top_k=1,
        cube_top_k=3,
        member_top_k=6,
        vector_index_path=None,
        embedding_client=KeywordEmbedding(),
    )

    selected = await selector.select(
        "查询传感器最大报警阈值",
        retrieval_intent=RetrievalIntent(needs=[
            SemanticNeed(phrase="最大报警阈值", usage="select"),
        ]),
        mode=SemanticCatalogMode.VECTOR,
    )

    assert selected.gap is None
    assert selected.context is not None
    assert "base_sensor.maximum_threshold" in selected.trace.member_hits
    assert "base_sensor" in selected.context.candidate_models
    assert "base_sensor.maximum_threshold" in selected.context.allowed_members


async def test_view_and_cube_select_from_need_bindings() -> None:
    selector = SemanticCatalogSelector(
        _catalog(),
        view_top_k=2,
        cube_top_k=3,
        member_top_k=10,
        vector_index_path=None,
        embedding_client=KeywordEmbedding(),
    )

    view_result = await selector.select(
        "查询报警数和设备名称",
        retrieval_intent=RetrievalIntent(needs=[
            SemanticNeed(phrase="报警数", usage="select", aggregate="count"),
            SemanticNeed(phrase="设备名称", usage="select"),
        ]),
        mode=SemanticCatalogMode.VECTOR,
    )
    cube_result = await selector.select(
        "查询设备名称和最大报警阈值",
        retrieval_intent=RetrievalIntent(needs=[
            SemanticNeed(phrase="设备名称", usage="select"),
            SemanticNeed(phrase="最大报警阈值", usage="select"),
        ]),
        mode=SemanticCatalogMode.VECTOR,
    )

    assert view_result.context is not None
    assert "hydrology_alarm_view" in view_result.context.candidate_models
    assert cube_result.context is not None
    assert {"base_sensor", "base_device"}.issubset(cube_result.context.candidate_models)


async def test_component_fallback_keeps_context_bounded() -> None:
    selector = SemanticCatalogSelector(
        _catalog(),
        view_top_k=1,
        cube_top_k=1,
        member_top_k=1,
        catalog_batch_size=1,
        context_member_limit=6,
        vector_index_path=None,
        embedding_client=KeywordEmbedding(),
    )

    selected = await selector.select(
        "查询设备风险信息",
        retrieval_intent=RetrievalIntent(needs=[
            SemanticNeed(phrase="设备名称", usage="select"),
            SemanticNeed(phrase="风险", usage="select"),
        ]),
        minimum_fallback_level=2,
        mode=SemanticCatalogMode.VECTOR,
    )

    assert selected.gap is None
    assert selected.context is not None
    prompt = context_for_prompt(selected.context)
    assert len(selected.context.candidate_models) <= 4
    assert len(selected.context.allowed_members) <= 6
    assert "base_label" not in prompt


async def test_low_score_need_still_produces_candidate_context_not_gap() -> None:
    selected = await SemanticCatalogSelector(
        _catalog(),
        view_top_k=2,
        cube_top_k=3,
        member_top_k=6,
        vector_index_path=None,
        embedding_client=KeywordEmbedding(),
    ).select(
        "查询水质酸碱度",
        retrieval_intent=RetrievalIntent(needs=[
            SemanticNeed(phrase="酸碱度", usage="select"),
        ]),
        mode=SemanticCatalogMode.VECTOR,
    )

    assert selected.gap is None
    assert selected.context is not None
    assert selected.context.candidate_models
    assert "select:酸碱度" in selected.context.binding_candidates
    assert not selected.trace.missing_needs


def test_cube_query_allows_connected_models_and_rejects_namespace_mix_or_disconnect() -> None:
    catalog = _catalog()
    connected = SemanticQuery(
        query_mode=QueryMode.CUBE,
        models=["base_event", "base_sensor", "base_device"],
        measures=["base_event.alarm_count"],
        dimensions=["base_device.name"],
    )

    validated = validate_semantic_query(
        connected,
        catalog,
        timezone="Asia/Shanghai",
        requested_max_rows=50,
        hard_max_rows=1000,
    )

    assert validated.query.to_cube_query()["dimensions"] == ["base_device.name"]
    assert "query_mode" not in validated.query.to_cube_query()
    assert "models" not in validated.query.to_cube_query()

    with pytest.raises(SemanticQueryValidationError, match="断连"):
        validate_semantic_query(
            connected.model_copy(update={"models": ["base_event", "base_label"], "dimensions": ["base_label.name"]}),
            catalog,
            timezone="Asia/Shanghai",
            requested_max_rows=50,
            hard_max_rows=1000,
        )
    with pytest.raises(SemanticQueryValidationError, match="混合"):
        validate_semantic_query(
            connected.model_copy(update={"models": ["base_event", "hydrology_alarm_view"]}),
            catalog,
            timezone="Asia/Shanghai",
            requested_max_rows=50,
            hard_max_rows=1000,
        )


def test_validator_rejects_ambiguous_diamond_join_before_load() -> None:
    models = {
        "base_a": _model(
            "base_a",
            "cube",
            "A",
            [_member("base_a", "count", "数量", member_type="measure", data_type="number")],
            joins=("base_b", "base_c"),
        ),
        "base_b": _model("base_b", "cube", "B", [_member("base_b", "id", "B ID")], joins=("base_d",)),
        "base_c": _model("base_c", "cube", "C", [_member("base_c", "id", "C ID")], joins=("base_d",)),
        "base_d": _model("base_d", "cube", "D", [_member("base_d", "name", "D 名称")]),
    }
    query = SemanticQuery(
        query_mode=QueryMode.CUBE,
        models=["base_a", "base_d"],
        measures=["base_a.count"],
        dimensions=["base_d.name"],
    )

    with pytest.raises(SemanticQueryValidationError, match="歧义"):
        validate_semantic_query(
            query,
            SemanticCatalog(models=models),
            timezone="Asia/Shanghai",
            requested_max_rows=50,
            hard_max_rows=1000,
        )
