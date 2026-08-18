from __future__ import annotations

import json
from collections.abc import Sequence
from pathlib import Path

import pytest

from ..models import (
    CatalogMember,
    CatalogModel,
    MemberRequirement,
    QueryMode,
    SemanticCatalog,
    SemanticIntent,
    SemanticQuery,
    TemporalIntent,
)
from ..nodes import _apply_query_defaults
from ..semantic_catalog import catalog_from_meta
from ..semantic_catalog_selector import SemanticCatalogSelector
from ..semantic_context import SemanticJoinGraph, context_for_prompt
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


def test_semantic_intent_normalizes_model_json_container_shapes() -> None:
    intent = SemanticIntent.model_validate({
        "metrics": "报警数",
        "dimensions": None,
        "time": ["当前", "最新"],
        "filters": None,
        "sort": None,
        "limit": None,
    })

    assert intent.metrics == ["报警数"]
    assert intent.dimensions == []
    assert intent.time == "当前；最新"
    assert intent.filters == []
    assert intent.sort == []


def test_semantic_intent_preserves_typed_requirements_qualifiers_and_temporal() -> None:
    intent = SemanticIntent.model_validate({
        "subjects": ["多因素预警"],
        "member_requirements": [{
            "phrase": "风险等级",
            "role": "filter",
            "operator": "gt",
            "values": [2],
        }],
        "qualifiers": ["有效"],
        "temporal": {
            "operator": "latest",
            "field_hint": "开始时间",
            "raw_phrase": "最新",
        },
        "result_shape": "detail",
    })

    assert intent.subjects == ["多因素预警"]
    assert intent.member_requirements[0] == MemberRequirement(
        phrase="风险等级",
        role="filter",
        operator="gt",
        values=[2],
    )
    assert intent.qualifiers == ["有效"]
    assert intent.temporal == TemporalIntent(
        operator="latest",
        field_hint="开始时间",
        raw_phrase="最新",
    )


async def test_scope_only_current_detail_uses_view_default_projection() -> None:
    view_name = "hydrology_multifactor_warnings"
    view = _model(
        view_name,
        "view",
        "水文多因素预警情况",
        [
            _member(view_name, "warning_event_id", "预警事件ID"),
            _member(view_name, "warning_name", "预警名称"),
            _member(view_name, "started_at", "开始时间", data_type="time"),
        ],
    ).model_copy(update={
        "ai_context": "多因素预警事件使用该 View，事件和配置固定为启用状态。",
        "aliases": ("多因素预警", "水害预警"),
    })
    catalog = SemanticCatalog(models={view_name: view})
    selector = SemanticCatalogSelector(
        catalog,
        view_top_k=1,
        cube_top_k=1,
        vector_index_path=None,
        embedding_client=None,
    )

    selected = await selector.select(
        "请查询当前水文多因素预警的具体情况",
        intent=SemanticIntent(
            subjects=["水文多因素预警"],
            temporal=TemporalIntent(operator="current", raw_phrase="当前"),
            result_shape="detail",
        ),
    )

    assert selected.gap is None
    assert selected.selected_models == [view_name]
    assert selected.context is not None
    assert selected.context.projection_policy == "model_default"
    assert selected.context.suggested_members
    assert selected.context.suggested_members[:2] == [
        f"{view_name}.warning_event_id",
        f"{view_name}.warning_name",
    ]
    assert selected.trace.member_coverage[view_name] == 1.0
    assert selected.trace.operator_resolution["resolution"] == "fixed_business_context"


async def test_real_scope_queries_resolve_view_and_fixed_qualifier_context() -> None:
    fixture = Path(__file__).with_name("fixtures") / "cube_meta_1_6_70.json"
    catalog = catalog_from_meta(json.loads(fixture.read_text(encoding="utf-8")))
    selector = SemanticCatalogSelector(
        catalog,
        view_top_k=3,
        cube_top_k=4,
        member_top_k=10,
        vector_index_path=None,
        embedding_client=None,
    )

    current = await selector.select(
        "请查询当前水文多因素预警的具体情况",
        intent=SemanticIntent(
            subjects=["水文多因素预警"],
            temporal=TemporalIntent(operator="current", raw_phrase="当前"),
            result_shape="detail",
        ),
    )
    single_factor = await selector.select(
        "统计当前有效的单因素报警事件数量",
        intent=SemanticIntent(
            subjects=["单因素报警事件"],
            member_requirements=[
                MemberRequirement(phrase="单因素报警事件数量", role="aggregate")
            ],
            qualifiers=["有效"],
            temporal=TemporalIntent(operator="current", raw_phrase="当前"),
            result_shape="aggregate",
        ),
    )

    assert current.selected_models == ["hydrology_multifactor_warnings"]
    assert current.gap is None
    assert current.trace.fallback_level == 0
    assert single_factor.selected_models == ["hydrology_single_factor_alarms"]
    assert single_factor.gap is None
    assert single_factor.trace.resolved_qualifiers == {"有效": "fixed_business_context"}


async def test_latest_scope_query_adds_time_order_and_limit_to_default_query() -> None:
    fixture = Path(__file__).with_name("fixtures") / "cube_meta_1_6_70.json"
    catalog = catalog_from_meta(json.loads(fixture.read_text(encoding="utf-8")))
    selector = SemanticCatalogSelector(
        catalog,
        view_top_k=3,
        cube_top_k=4,
        member_top_k=10,
        vector_index_path=None,
        embedding_client=None,
    )

    selected = await selector.select(
        "查询最新一条多因素预警",
        intent=SemanticIntent(
            subjects=["多因素预警"],
            temporal=TemporalIntent(operator="latest", raw_phrase="最新"),
            result_shape="detail",
        ),
    )

    assert selected.context is not None
    assert selected.context.operator_resolution == {
        "operator": "latest",
        "resolution": "time_dimension",
        "time_member": "hydrology_multifactor_warnings.started_at",
        "direction": "desc",
        "limit": 1,
    }
    query = _apply_query_defaults(
        SemanticQuery(
            query_mode=selected.context.query_mode,
            models=selected.context.models,
        ),
        selected.context,
    )
    assert query.dimensions
    assert query.order[0].member == "hydrology_multifactor_warnings.started_at"
    assert query.order[0].direction == "desc"
    assert query.limit == 1
    assert query.ungrouped is True


async def test_order_requirement_can_bind_a_measure() -> None:
    selector = SemanticCatalogSelector(
        _catalog(),
        view_top_k=1,
        cube_top_k=1,
        vector_index_path=None,
        embedding_client=None,
    )

    selected = await selector.select(
        "按报警数倒序排序",
        intent=SemanticIntent(
            subjects=["报警事件"],
            member_requirements=[
                MemberRequirement(
                    phrase="报警数",
                    role="order",
                    direction="desc",
                )
            ],
        ),
    )

    assert selected.context is not None
    assert selected.context.query_mode == QueryMode.VIEW
    assert selected.trace.resolved_requirements == {
        "报警数": "hydrology_alarm_view.alarm_count"
    }


async def test_explicit_project_is_not_absorbed_into_model_scope() -> None:
    view_name = "hydrology_multifactor_warnings"
    catalog = SemanticCatalog(models={
        view_name: _model(
            view_name,
            "view",
            "水文多因素预警情况",
            [_member(view_name, "warning_name", "预警名称")],
        ).model_copy(update={"aliases": ("多因素预警",)}),
    })
    selector = SemanticCatalogSelector(
        catalog,
        view_top_k=1,
        cube_top_k=1,
        vector_index_path=None,
        embedding_client=None,
    )

    selected = await selector.select(
        "显示预警名称",
        intent=SemanticIntent(
            subjects=["多因素预警"],
            member_requirements=[
                MemberRequirement(phrase="不存在字段X", role="project")
            ],
            result_shape="detail",
        ),
    )

    assert selected.context is None
    assert selected.gap is not None
    assert selected.gap.missing_concepts == ["不存在字段X"]


async def test_forced_cube_fallback_has_scope_anchor_without_requirements() -> None:
    selector = SemanticCatalogSelector(
        _catalog(),
        view_top_k=1,
        cube_top_k=1,
        vector_index_path=None,
        embedding_client=KeywordEmbedding(),
    )

    selected = await selector.select(
        "查询报警事件具体情况",
        intent=SemanticIntent(subjects=["报警事件"], result_shape="detail"),
        minimum_fallback_level=2,
    )

    assert selected.gap is None
    assert selected.context is not None
    assert selected.context.query_mode == QueryMode.CUBE
    assert selected.context.fallback_anchor
    assert selected.context.models


async def test_view_and_cube_have_separate_top_k_and_global_member_recalls_parent_cube() -> None:
    selector = SemanticCatalogSelector(
        _catalog(),
        view_top_k=1,
        cube_top_k=1,
        member_top_k=2,
        vector_index_path=None,
        embedding_client=KeywordEmbedding(),
    )

    selected = await selector.select(
        "查询传感器最大报警阈值",
        intent=SemanticIntent(dimensions=["最大报警阈值"]),
    )

    assert len(selected.trace.view_candidates) == 1
    assert len(selected.trace.cube_candidates) == 1
    assert "base_sensor" in selected.trace.member_parent_models
    assert selected.context is not None
    assert selected.context.query_mode == QueryMode.CUBE
    assert "base_sensor" in selected.context.models
    assert all(model.startswith("base_") for model in selected.context.models)


async def test_full_view_coverage_uses_view_and_partial_coverage_switches_to_cube() -> None:
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
        intent=SemanticIntent(metrics=["报警数"], dimensions=["设备名称"]),
    )
    cube_result = await selector.select(
        "查询设备名称和最大报警阈值",
        intent=SemanticIntent(dimensions=["设备名称", "最大报警阈值"]),
    )

    assert view_result.context is not None
    assert view_result.context.query_mode == QueryMode.VIEW
    assert view_result.context.models == ["hydrology_alarm_view"]
    assert view_result.trace.view_coverage["hydrology_alarm_view"] == 1.0
    assert cube_result.context is not None
    assert cube_result.context.query_mode == QueryMode.CUBE
    assert set(cube_result.context.models) == {"base_sensor", "base_device"}


async def test_rerank_scores_cover_view_and_cube_namespaces_with_cube_connectivity() -> None:
    selector = SemanticCatalogSelector(
        _catalog(),
        view_top_k=2,
        cube_top_k=3,
        member_top_k=10,
        vector_index_path=None,
        embedding_client=KeywordEmbedding(),
    )

    selected = await selector.select(
        "查询设备名称和最大报警阈值",
        intent=SemanticIntent(dimensions=["设备名称", "最大报警阈值"]),
    )

    assert selected.trace.view_coverage
    assert selected.trace.cube_coverage
    assert set(selected.trace.view_candidates) <= set(selected.trace.rerank_scores)
    assert set(selected.trace.cube_candidates) <= set(selected.trace.rerank_scores)
    assert selected.trace.cube_connectivity["base_sensor"] == 1.0
    assert selected.trace.cube_connectivity["base_label"] == 0.0


async def test_view_coverage_compares_against_global_member_match() -> None:
    view_name = "hydrology_warning_view"
    cube_name = "base_warning_configuration"
    catalog = SemanticCatalog(models={
        view_name: _model(
            view_name,
            "view",
            "多因素预警事件",
            [_member(view_name, "warning_count", "多因素预警事件数", member_type="measure", data_type="number")],
        ),
        cube_name: _model(
            cube_name,
            "cube",
            "多因素预警配置",
            [_member(cube_name, "configuration_count", "多因素预警配置数", member_type="measure", data_type="number")],
        ),
    })
    selector = SemanticCatalogSelector(
        catalog,
        view_top_k=1,
        cube_top_k=1,
        member_top_k=2,
        vector_index_path=None,
        embedding_client=KeywordEmbedding(),
    )

    selected = await selector.select(
        "统计多因素预警配置数量",
        intent=SemanticIntent(metrics=["多因素预警配置数量"]),
    )

    assert selected.trace.view_coverage[view_name] == 0.0
    assert selected.context is not None
    assert selected.context.query_mode == QueryMode.CUBE
    assert selected.context.models == [cube_name]


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

async def test_component_fallback_is_batched_and_context_never_contains_full_catalog() -> None:
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
        intent=SemanticIntent(dimensions=["设备名称", "风险"]),
        minimum_fallback_level=2,
    )

    assert selected.trace.catalog_batches_analyzed >= 3
    if selected.context is not None:
        prompt = context_for_prompt(selected.context)
        assert len(selected.context.models) <= 4
        assert len(selected.context.allowed_members) <= 6
        assert "base_label" not in prompt


async def test_missing_concept_returns_structured_semantic_model_gap() -> None:
    selector = SemanticCatalogSelector(
        _catalog(),
        view_top_k=2,
        cube_top_k=3,
        member_top_k=6,
        vector_index_path=None,
        embedding_client=KeywordEmbedding(),
    )

    selected = await selector.select(
        "查询水质酸碱度",
        intent=SemanticIntent(dimensions=["酸碱度"]),
    )

    assert selected.context is None
    assert selected.gap is not None
    assert selected.gap.code == "semantic_model_gap"
    assert selected.gap.missing_concepts == ["酸碱度"]
    assert selected.trace.fallback_level == 3
