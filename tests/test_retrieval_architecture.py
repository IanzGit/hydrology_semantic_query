from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import pytest

from app.agents.scenarios.hydrology_semantic_query.query_child.tools.run_semantic_query import (
    SemanticQueryValidationError,
    validate_semantic_query,
)
from app.agents.scenarios.hydrology_semantic_query.query_child.tools.search_semantic_catalog import (
    SemanticCatalogRetriever,
)

from ..contracts import (
    FilterOperator,
    QueryMode,
    SemanticCatalogMode,
    SemanticFilter,
    SemanticQuery,
)
from ..query_child.client import SemanticCatalogError, catalog_from_meta
from ..query_child.models import (
    CatalogMember,
    CatalogModel,
    SemanticCatalog,
)
from ..query_child.prompts import SEMANTIC_QUERY_RULES


class KeywordEmbedding:
    model_path = "keyword-test"
    keywords = ("设备", "传感器", "当前值", "状态", "更新时间", "报警", "水质")

    def __init__(self, *, fail_query: bool = False) -> None:
        self.fail_query = fail_query
        self.query_calls = 0

    async def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
        return [self._vector(text) for text in texts]

    async def embed_query(self, text: str) -> list[float]:
        self.query_calls += 1
        if self.fail_query:
            raise RuntimeError("query embedding unavailable")
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
    folder: str | None = None,
) -> CatalogMember:
    return CatalogMember(
        name=f"{model}.{name}",
        title=title,
        member_type=member_type,
        data_type=data_type,
        folder=folder,
    )


def _model(
    name: str,
    model_type: str,
    title: str,
    members: Sequence[CatalogMember],
    *,
    component: int | None = 1,
    description: str = "",
    folders: tuple[str, ...] = (),
    domain: str = "hydrology",
) -> CatalogModel:
    return CatalogModel(
        name=name,
        model_type=model_type,
        title=title,
        description=description,
        members={member.name: member for member in members},
        connected_component=component,
        folders=folders,
        business_domain=domain,
    )


def _catalog() -> SemanticCatalog:
    device = "base_device_info"
    value = "base_device_x_value"
    view = "hydrology_water_quality_view"
    other = "base_disconnected"
    return SemanticCatalog(models={
        device: _model(
            device,
            "cube",
            "设备信息",
            [
                _member(device, "name", "设备名称"),
                _member(device, "code", "设备编码"),
                _member(device, "is_enabled", "设备是否启用", data_type="boolean"),
            ],
        ),
        value: _model(
            value,
            "cube",
            "传感器实时值",
            [
                _member(value, "sensor_current_value", "传感器当前值", data_type="number"),
                _member(value, "sensor_status", "传感器状态"),
                _member(value, "updated_at", "更新时间", data_type="time"),
                _member(value, "sample_count", "样本数", member_type="measure", data_type="number"),
            ],
        ),
        view: _model(
            view,
            "view",
            "水质快捷查询",
            [
                _member(view, "device_name", "设备名称", folder="水质汇总"),
                _member(view, "sample_count", "样本数", member_type="measure", data_type="number", folder="水质汇总"),
            ],
            component=None,
            folders=("水质汇总",),
        ),
        other: _model(
            other,
            "cube",
            "断连实体",
            [_member(other, "name", "断连名称")],
            component=2,
            domain="other",
        ),
    })


def _retriever(
    catalog: SemanticCatalog,
    *,
    embedding: KeywordEmbedding | None = None,
    threshold: int = 30000,
    top_k: int = 20,
    model_top_k: int = 2,
    cache_path: Path | None = None,
) -> SemanticCatalogRetriever:
    return SemanticCatalogRetriever(
        catalog,
        model_top_k=model_top_k,
        context_top_k=top_k,
        vector_index_path=str(cache_path) if cache_path else None,
        embedding_client=embedding,
        auto_full_context_max_chars=threshold,
    )


def test_production_code_has_no_removed_selection_chain() -> None:
    scenario = Path(__file__).resolve().parents[1]
    source = "\n".join(
        path.read_text(encoding="utf-8")
        for path in scenario.glob("*.py")
    )
    removed = (
        "SemanticNeed",
        "RetrievalIntent",
        "QueryUnderstanding",
        "NeedCandidate",
        "NeedResolution",
        "allowed_members",
        "binding_candidates",
        "binding_scores",
        "best_member",
        "understand_query",
        "projection_mode",
        "projection_policy",
        "semantic_model_gap",
        "clarification",
        "make_retrieval_node",
        "make_generation_node",
        "make_recovery_node",
        "semantic_generation",
        "failure_recovery",
    )
    assert not Path(scenario / "semantic_catalog_selector.py").exists()
    assert not Path(scenario / "semantic_query_planner.py").exists()
    assert not Path(scenario / "semantic_context.py").exists()
    assert all(term not in source for term in removed)


def test_context_contract_and_prompt_are_retrieval_only() -> None:
    retriever = _retriever(_catalog(), threshold=100000)
    assert retriever.context_top_k == 20
    assert "不是成员绑定、候选白名单或最终路由结果" in SEMANTIC_QUERY_RULES
    assert "同名或相似的传感器属性替代" in SEMANTIC_QUERY_RULES


async def test_auto_small_catalog_provides_complete_context() -> None:
    embedding = KeywordEmbedding()
    retrieved = await _retriever(
        _catalog(), embedding=embedding, threshold=100000
    ).retrieve("中央水仓水质")

    assert retrieved.mode == SemanticCatalogMode.FULL
    assert retrieved.context.strategy == SemanticCatalogMode.FULL
    assert embedding.query_calls == 0
    model_items = {
        item.name: item for item in retrieved.context.items if item.item_type == "model"
    }
    assert set(model_items) == set(_catalog().models)
    assert len(model_items["base_device_x_value"].payload["members"]) == 4
    assert retrieved.trace.index_source == "full_catalog"


async def test_large_catalog_uses_model_then_member_retrieval(tmp_path: Path) -> None:
    retriever = _retriever(
        _catalog(),
        embedding=KeywordEmbedding(),
        threshold=1,
        top_k=3,
        model_top_k=2,
        cache_path=tmp_path / "vectors.sqlite3",
    )
    await retriever.build_index()
    retrieved = await retriever.retrieve("设备传感器状态")

    assert retrieved.mode == SemanticCatalogMode.VECTOR
    assert retrieved.trace.queries == ["设备传感器状态"]
    model_hits = [hit for hit in retrieved.trace.hits if hit.item_type == "model"]
    member_hits = [hit for hit in retrieved.trace.hits if hit.item_type != "model"]
    assert len(model_hits) == 2
    assert len(member_hits) == 3
    assert {hit.item_type for hit in member_hits} <= {"member", "view_folder"}
    assert {hit.model_name for hit in member_hits} <= {
        hit.model_name for hit in model_hits
    }
    for item in retrieved.context.items:
        if item.item_type != "model":
            continue
        assert item.payload["members"]
        assert all(
            set(member) == {"name", "kind", "type"}
            for member in item.payload["members"]
        )


async def test_member_retrieval_reserves_coverage_for_each_candidate_model(
    tmp_path: Path,
) -> None:
    device = "base_device_info"
    sensor = "base_device_x_value"
    catalog = SemanticCatalog(models={
        device: _model(
            device,
            "cube",
            "设备信息",
            [
                _member(device, "name", "设备名称"),
                _member(device, "place", "设备位置"),
                _member(device, "state", "设备状态"),
            ],
        ),
        sensor: _model(
            sensor,
            "cube",
            "传感器信息",
            [
                _member(sensor, f"state_{index}", f"设备传感器状态 {index}")
                for index in range(6)
            ],
        ),
    })

    retriever = _retriever(
        catalog,
        embedding=KeywordEmbedding(),
        threshold=1,
        top_k=2,
        model_top_k=2,
        cache_path=tmp_path / "vectors.sqlite3",
    )
    await retriever.build_index()
    retrieved = await retriever.retrieve("查询火赖沟下游设备的传感器信息")

    model_hits = {
        hit.model_name for hit in retrieved.trace.hits if hit.item_type == "model"
    }
    member_hits = {
        hit.model_name for hit in retrieved.trace.hits if hit.item_type == "member"
    }
    assert member_hits == model_hits


async def test_member_hit_adds_parent_model_and_cube_component() -> None:
    retrieved = await _retriever(_catalog(), threshold=1, top_k=1).retrieve(
        "base_device_x_value.sensor_current_value",
        mode=SemanticCatalogMode.VECTOR,
    )

    member_hits = [hit for hit in retrieved.trace.hits if hit.item_type == "member"]
    assert member_hits[0].name == "base_device_x_value.sensor_current_value"
    keys = {(item.item_type, item.name) for item in retrieved.context.items}
    assert ("model", "base_device_x_value") in keys
    assert ("join_component", "1") in keys


async def test_view_folder_participates_in_candidate_model_retrieval() -> None:
    retrieved = await _retriever(_catalog(), threshold=1, top_k=1).retrieve(
        "hydrology_water_quality_view:水质汇总",
        mode=SemanticCatalogMode.VECTOR,
    )

    assert any(hit.item_type == "view_folder" for hit in retrieved.trace.hits)


async def test_metadata_filter_builds_full_accessible_catalog_before_context() -> None:
    retrieved = await _retriever(_catalog(), threshold=1, top_k=1).retrieve(
        "状态",
        mode=SemanticCatalogMode.VECTOR,
        metadata_filters={"business_domain": "hydrology", "model_type": "cube"},
    )

    assert set(retrieved.catalog.models) == {"base_device_info", "base_device_x_value"}
    assert all(
        item.model_name in {None, "base_device_info", "base_device_x_value"}
        for item in retrieved.context.items
    )
    component_models = {
        model["name"]
        for item in retrieved.context.items
        if item.item_type == "join_component"
        for model in item.payload["models"]
    }
    assert component_models <= {"base_device_info", "base_device_x_value"}


async def test_metadata_filter_can_shrink_auto_mode_to_full() -> None:
    retrieved = await _retriever(_catalog(), threshold=2000).retrieve(
        "水质",
        metadata_filters={"model_name": "hydrology_water_quality_view"},
    )

    assert retrieved.mode == SemanticCatalogMode.FULL
    assert set(retrieved.catalog.models) == {"hydrology_water_quality_view"}


async def test_lexical_fallback_uses_same_context_interface(tmp_path: Path) -> None:
    cache_path = tmp_path / "vectors.sqlite3"
    await _retriever(
        _catalog(),
        embedding=KeywordEmbedding(),
        threshold=1,
        top_k=2,
        cache_path=cache_path,
    ).build_index()
    retrieved = await _retriever(
        _catalog(),
        embedding=KeywordEmbedding(fail_query=True),
        threshold=1,
        top_k=2,
        cache_path=cache_path,
    ).retrieve("设备状态")

    assert retrieved.mode == SemanticCatalogMode.VECTOR
    assert len([hit for hit in retrieved.trace.hits if hit.item_type == "model"]) == 2
    assert len([hit for hit in retrieved.trace.hits if hit.item_type != "model"]) == 2
    assert retrieved.trace.index_source.startswith("lexical_fallback:")
    assert any("词法目录检索" in warning for warning in retrieved.warnings)


async def test_unretrieved_member_is_not_rejected_as_a_whitelist_violation() -> None:
    retrieved = await _retriever(_catalog(), threshold=1, top_k=1).retrieve(
        "base_device_info.name",
        mode=SemanticCatalogMode.VECTOR,
    )
    assert all(
        hit.name != "base_device_x_value.updated_at"
        for hit in retrieved.trace.hits
    )
    query = SemanticQuery(
        query_mode=QueryMode.CUBE,
        models=["base_device_x_value"],
        dimensions=["base_device_x_value.updated_at"],
        ungrouped=True,
    )

    validated = validate_semantic_query(
        query,
        retrieved.catalog,
        requested_max_rows=50,
        hard_max_rows=1000,
    )
    assert validated.query.dimensions == ["base_device_x_value.updated_at"]


def _contains_key(value: object, key: str) -> bool:
    if isinstance(value, dict):
        return key in value or any(_contains_key(child, key) for child in value.values())
    if isinstance(value, list):
        return any(_contains_key(child, key) for child in value)
    return False


def test_semantic_query_tool_schema_has_no_catalog_enums() -> None:
    schema = SemanticQuery.model_json_schema(by_alias=True)
    models = schema["properties"]["models"]

    assert models["items"] == {"type": "string"}
    assert not _contains_key(schema["properties"]["dimensions"], "enum")
    assert not _contains_key(schema["properties"]["measures"], "enum")


@pytest.mark.parametrize(
    ("query", "expected_code"),
    [
        (
            SemanticQuery(
                query_mode=QueryMode.CUBE,
                models=["base_device_x_value"],
                measures=["base_device_x_value.sample_count"],
            ),
            None,
        ),
        (
            SemanticQuery(
                query_mode=QueryMode.CUBE,
                models=["base_device_info"],
                dimensions=["base_device_info.name"],
                ungrouped=True,
            ),
            None,
        ),
        (
            SemanticQuery(
                query_mode=QueryMode.CUBE,
                models=["base_device_info"],
                dimensions=["base_device_info.is_enabled"],
                ungrouped=True,
            ),
            None,
        ),
        (
            SemanticQuery(
                query_mode=QueryMode.VIEW,
                models=["hydrology_water_quality_view"],
                measures=["hydrology_water_quality_view.sample_count"],
                dimensions=["hydrology_water_quality_view.device_name"],
            ),
            None,
        ),
        (
            SemanticQuery(
                query_mode=QueryMode.CUBE,
                models=["base_device_info", "base_device_x_value"],
                dimensions=["base_device_info.name", "base_device_x_value.sensor_status"],
                ungrouped=True,
            ),
            None,
        ),
        (
            SemanticQuery(
                query_mode=QueryMode.CUBE,
                models=["base_device_info", "base_disconnected"],
                dimensions=["base_device_info.name", "base_disconnected.name"],
                ungrouped=True,
            ),
            "join_unreachable",
        ),
    ],
)
def test_validator_supports_planner_selected_view_and_cube_shapes(
    query: SemanticQuery,
    expected_code: str | None,
) -> None:
    if expected_code is None:
        validated = validate_semantic_query(
            query, _catalog(), requested_max_rows=50, hard_max_rows=1000
        )
        assert validated.query.models == query.models
        return
    with pytest.raises(SemanticQueryValidationError) as captured:
        validate_semantic_query(
            query, _catalog(), requested_max_rows=50, hard_max_rows=1000
        )
    assert captured.value.code == expected_code


@pytest.mark.parametrize(
    "query",
    [
        SemanticQuery(
            query_mode=QueryMode.CUBE,
            models=["base_device_x_value"],
            measures=["base_device_x_value.sample_count"],
            dimensions=["base_device_x_value.sensor_status"],
            ungrouped=True,
        ),
        SemanticQuery(
            query_mode=QueryMode.CUBE,
            models=["base_device_x_value"],
            dimensions=["base_device_x_value.sensor_status"],
        ),
    ],
)
def test_validator_rejects_inconsistent_query_shape(query: SemanticQuery) -> None:
    with pytest.raises(SemanticQueryValidationError) as captured:
        validate_semantic_query(
            query, _catalog(), requested_max_rows=50, hard_max_rows=1000
        )
    assert captured.value.code == "query_shape"


def test_validator_uses_stable_unknown_scope_type_and_filter_codes() -> None:
    cases = [
        (
            SemanticQuery(
                query_mode=QueryMode.CUBE,
                models=["missing"],
                dimensions=["missing.name"],
                ungrouped=True,
            ),
            "unknown_model",
        ),
        (
            SemanticQuery(
                query_mode=QueryMode.CUBE,
                models=["base_device_info"],
                dimensions=["base_device_info.missing"],
                ungrouped=True,
            ),
            "unknown_member",
        ),
        (
            SemanticQuery(
                query_mode=QueryMode.CUBE,
                models=["base_device_info"],
                dimensions=["base_device_x_value.sensor_status"],
                ungrouped=True,
            ),
            "member_scope_mismatch",
        ),
        (
            SemanticQuery(
                query_mode=QueryMode.CUBE,
                models=["base_device_x_value"],
                dimensions=["base_device_x_value.sample_count"],
                ungrouped=True,
            ),
            "member_type_mismatch",
        ),
        (
            SemanticQuery(
                query_mode=QueryMode.CUBE,
                models=["base_device_x_value"],
                dimensions=["base_device_x_value.sensor_status"],
                filters=[SemanticFilter(
                    member="base_device_x_value.sensor_status",
                    operator=FilterOperator.GT,
                    values=[1],
                )],
                ungrouped=True,
            ),
            "invalid_filter",
        ),
    ]
    for query, code in cases:
        with pytest.raises(SemanticQueryValidationError) as captured:
            validate_semantic_query(
                query, _catalog(), requested_max_rows=50, hard_max_rows=1000
            )
        assert captured.value.code == code


def test_validator_reports_all_unknown_members_together() -> None:
    query = SemanticQuery(
        query_mode=QueryMode.CUBE,
        models=["base_device_info", "base_device_x_value"],
        dimensions=[
            "base_device_info.location",
            "base_device_info.status",
        ],
        ungrouped=True,
    )

    with pytest.raises(SemanticQueryValidationError) as captured:
        validate_semantic_query(
            query,
            _catalog(),
            requested_max_rows=50,
            hard_max_rows=1000,
        )

    assert captured.value.code == "unknown_member"
    assert captured.value.details == {
        "invalid_members": [
            {
                "code": "unknown_member",
                "member": "base_device_info.location",
                "message": "Cube 语义目录中不存在成员：base_device_info.location",
            },
            {
                "code": "unknown_member",
                "member": "base_device_info.status",
                "message": "Cube 语义目录中不存在成员：base_device_info.status",
            },
        ]
    }


def test_catalog_parses_public_models_and_rejects_foreign_member() -> None:
    payload = {
        "cubes": [{
            "name": "base_device_info",
            "type": "cube",
            "public": True,
            "title": "设备信息",
            "connectedComponent": 1,
            "meta": {"business_domain": "hydrology", "default_projection": ["name"]},
            "measures": [],
            "dimensions": [{
                "name": "base_device_info.name",
                "title": "设备名称",
                "type": "string",
                "public": True,
            }],
            "segments": [],
            "folders": [],
            "hierarchies": [],
        }]
    }
    catalog = catalog_from_meta(payload)
    assert catalog.models["base_device_info"].default_projection == (
        "base_device_info.name",
    )
    payload["cubes"][0]["dimensions"][0]["name"] = "foreign.name"
    with pytest.raises(SemanticCatalogError):
        catalog_from_meta(payload)
