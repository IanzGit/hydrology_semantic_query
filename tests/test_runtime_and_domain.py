from __future__ import annotations

import asyncio

import httpx
import pytest
from pydantic import ValidationError

from app.agents.scenarios.hydrology_semantic_query.tools.run_semantic_query import (
    SemanticQueryValidationError,
    normalize_cube_response,
    validate_semantic_query,
)

from ..client import CubeClient, CubeClientError, catalog_from_meta
from ..config import (
    load_hydrology_semantic_query_settings,
    normalize_cube_url,
)
from ..models import SemanticQuery
from .catalog_expectations import (
    PRIVATE_CUBES,
    PUBLIC_CUBES,
    PUBLIC_VIEWS,
    load_declared_model,
)


def _meta() -> dict:
    views = []
    for name in PUBLIC_VIEWS:
        views.append({
            "name": name,
            "title": name,
            "type": "view",
            "public": True,
            "measures": [
                {"name": f"{name}.count", "title": "数量", "type": "number"},
            ],
            "dimensions": [
                {"name": f"{name}.id", "title": "ID", "type": "number"},
                {"name": f"{name}.name", "title": "名称", "type": "string"},
                {"name": f"{name}.time", "title": "时间", "type": "time"},
            ],
            "segments": [
                {"name": f"{name}.enabled", "title": "已启用"},
            ],
        })
    return {"cubes": views}


def _model() -> dict:
    return load_declared_model()


def test_cube_yaml_exposes_governed_views_and_base_cubes() -> None:
    model = _model()
    assert {item["name"] for item in model["views"]} == PUBLIC_VIEWS
    assert all(item["public"] is True for item in model["views"])
    assert {
        item["name"]
        for item in model["cubes"]
        if item.get("public") is not False and item["name"].startswith("base_")
    } == PUBLIC_CUBES
    assert {
        item["name"]
        for item in model["cubes"]
        if item.get("public") is False or not item["name"].startswith("base_")
    } == PRIVATE_CUBES
    assert all(item.get("title") and item.get("description") for item in model["views"])
    assert all(item.get("meta", {}).get("ai_context") for item in model["views"])
    for cube in model["cubes"]:
        for group in ("measures", "dimensions", "segments"):
            for member in cube.get(group, []):
                assert member.get("title") and member.get("description")


def test_all_public_views_accept_basic_semantic_queries() -> None:
    catalog = catalog_from_meta(_meta())
    for view in PUBLIC_VIEWS:
        query = SemanticQuery.model_validate({
            "query_mode": "view",
            "models": [view],
            "dimensions": [f"{view}.name"],
            "ungrouped": True,
        })
        validate_semantic_query(
            query,
            catalog,
            requested_max_rows=50,
            hard_max_rows=1000,
        )


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        ("http://cube.internal", "http://cube.internal/cubejs-api/v1"),
        ("http://cube.internal/cubejs-api/", "http://cube.internal/cubejs-api/v1"),
        ("https://cube.internal/cubejs-api/v1/", "https://cube.internal/cubejs-api/v1"),
    ],
)
def test_cube_url_normalization(value: str, expected: str) -> None:
    assert normalize_cube_url(value) == expected


@pytest.mark.parametrize(
    "value",
    ["", "cube.internal", "ftp://cube.internal", "http:///v1", "http://cube internal"],
)
def test_cube_url_rejects_invalid_values(value: str) -> None:
    with pytest.raises(ValueError):
        normalize_cube_url(value)


def test_settings_validate_timezone_and_row_limits(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("HYDROLOGY_SEMANTIC_QUERY_TIMEZONE", "Invalid/Timezone")
    with pytest.raises(ValueError, match="IANA"):
        load_hydrology_semantic_query_settings()
    monkeypatch.setenv("HYDROLOGY_SEMANTIC_QUERY_TIMEZONE", "Asia/Shanghai")
    monkeypatch.setenv("HYDROLOGY_SEMANTIC_QUERY_MAX_ROWS", "101")
    monkeypatch.setenv("HYDROLOGY_SEMANTIC_QUERY_HARD_MAX_ROWS", "100")
    with pytest.raises(ValueError, match="MAX_ROWS"):
        load_hydrology_semantic_query_settings()


@pytest.mark.parametrize(
    ("name", "value"),
    [
        ("TIMEOUT_SECONDS", "0"),
        ("TIMEOUT_SECONDS", "nan"),
        ("CONTINUE_WAIT_RETRIES", "-1"),
        ("META_CACHE_TTL_SECONDS", "-1"),
        ("META_CACHE_TTL_SECONDS", "inf"),
        ("MAX_RETRIES", "2"),
    ],
)
def test_settings_reject_invalid_runtime_boundaries(
    monkeypatch: pytest.MonkeyPatch,
    name: str,
    value: str,
) -> None:
    monkeypatch.setenv(f"HYDROLOGY_SEMANTIC_QUERY_{name}", value)
    with pytest.raises(ValueError):
        load_hydrology_semantic_query_settings()


def test_catalog_preserves_view_and_member_ai_context() -> None:
    payload = _meta()
    payload["cubes"][0]["meta"] = {"ai_context": "View 上下文"}
    payload["cubes"][0]["dimensions"][0]["meta"] = {"ai_context": "成员上下文"}
    catalog = catalog_from_meta(payload)
    model = catalog.models[payload["cubes"][0]["name"]]
    assert model.ai_context == "View 上下文"
    assert model.members[payload["cubes"][0]["dimensions"][0]["name"]].ai_context == "成员上下文"


def test_catalog_accepts_public_models_and_uses_meta_member_visibility() -> None:
    payload = _meta()
    payload["cubes"].extend([
        {
            "name": "base_device_info",
            "type": "cube",
            "public": True,
            "title": "设备原始实体",
            "measures": [],
            "dimensions": [
                {
                    "name": "base_device_info.name",
                    "title": "名称",
                    "type": "string",
                },
                {
                    "name": "base_device_info.secret",
                    "title": "私有字段",
                    "type": "string",
                    "public": False,
                },
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
        {
            "name": "hydrology_unmanaged",
            "type": "cube",
            "public": True,
            "measures": [],
            "dimensions": [],
            "segments": [],
        },
    ])

    catalog = catalog_from_meta(payload)

    assert set(catalog.models) == {
        *PUBLIC_VIEWS,
        "base_device_info",
        "hydrology_unmanaged",
    }
    assert catalog.models["base_device_info"].model_type == "cube"
    assert catalog.models["hydrology_unmanaged"].model_type == "cube"
    assert list(catalog.models["base_device_info"].members) == ["base_device_info.name"]
    query = SemanticQuery(
        query_mode="cube",
        models=["base_device_info"],
        dimensions=["base_device_info.name"],
        ungrouped=True,
    )
    validate_semantic_query(
        query,
        catalog,
        requested_max_rows=50,
        hard_max_rows=1000,
    )


@pytest.mark.parametrize("field", ["view", "model"])
def test_semantic_query_rejects_legacy_singular_model_fields(field: str) -> None:
    with pytest.raises(ValidationError):
        SemanticQuery.model_validate({
            field: "view_label_sensor_devices",
            "dimensions": ["view_label_sensor_devices.name"],
            "ungrouped": True,
        })


def test_catalog_public_model_filter_and_recursive_validation() -> None:
    catalog = catalog_from_meta(_meta())
    assert set(catalog.models) == PUBLIC_VIEWS
    query = SemanticQuery.model_validate({
        "query_mode": "view",
        "models": ["view_label_sensor_devices"],
        "dimensions": ["view_label_sensor_devices.name"],
        "filters": [{
            "or": [
                {"member": "view_label_sensor_devices.name", "operator": "equals", "values": ["A"]},
                {"and": [
                    {"member": "view_label_sensor_devices.name", "operator": "equals", "values": ["B"]},
                ]},
            ],
        }],
        "timeDimensions": [{
            "dimension": "view_label_sensor_devices.time",
            "granularity": "day",
            "dateRange": ["2026-05-07", "2026-05-13"],
        }],
        "order": [
            {"member": "view_label_sensor_devices.time", "direction": "asc"},
            {"member": "view_label_sensor_devices.name", "direction": "desc"},
        ],
        "limit": 5000,
        "ungrouped": True,
    })
    validated = validate_semantic_query(
        query,
        catalog,
        requested_max_rows=2000,
        hard_max_rows=1000,
    )
    assert validated.query.limit == 1000
    assert list(validated.query.to_cube_query()["order"]) == [
        "view_label_sensor_devices.time",
        "view_label_sensor_devices.name",
    ]
    assert validated.warnings


def test_top_level_measure_and_dimension_filters_can_coexist() -> None:
    query = SemanticQuery.model_validate({
        "query_mode": "view",
        "models": ["view_label_sensor_devices"],
        "measures": ["view_label_sensor_devices.count"],
        "dimensions": ["view_label_sensor_devices.name"],
        "filters": [
            {"member": "view_label_sensor_devices.name", "operator": "equals", "values": ["A"]},
            {"member": "view_label_sensor_devices.count", "operator": "gt", "values": [1]},
        ],
    })
    validated = validate_semantic_query(
        query,
        catalog_from_meta(_meta()),
        requested_max_rows=50,
        hard_max_rows=1000,
    )
    assert validated.query.to_cube_query()["filters"][1]["values"] == ["1"]


@pytest.mark.parametrize(
    "filter_payload",
    [
        {"member": "view_label_sensor_devices.name", "operator": "contains"},
        {"member": "view_label_sensor_devices.name", "operator": "gt", "values": [1]},
        {"member": "view_label_sensor_devices.count", "operator": "gt", "values": [1, 2]},
        {"member": "view_label_sensor_devices.count", "operator": "gt", "values": ["x"]},
        {"member": "view_label_sensor_devices.name", "operator": "set", "values": ["x"]},
        {"member": "view_label_sensor_devices.name", "operator": "beforeDate", "values": ["today"]},
        {"member": "view_label_sensor_devices.time", "operator": "inDateRange", "values": []},
        {"member": "view_label_sensor_devices.time", "operator": "beforeDate", "values": ["a", "b"]},
    ],
)
def test_filter_operator_compatibility_and_value_counts(filter_payload: dict) -> None:
    query = SemanticQuery.model_validate({
        "query_mode": "view",
        "models": ["view_label_sensor_devices"],
        "dimensions": ["view_label_sensor_devices.name"],
        "filters": [filter_payload],
        "ungrouped": True,
    })
    with pytest.raises(SemanticQueryValidationError):
        validate_semantic_query(
            query,
            catalog_from_meta(_meta()),
            requested_max_rows=50,
            hard_max_rows=1000,
        )


def test_filter_accepts_supported_shapes_and_serializes_scalar_values() -> None:
    query = SemanticQuery.model_validate({
        "query_mode": "view",
        "models": ["view_label_sensor_devices"],
        "dimensions": ["view_label_sensor_devices.name"],
        "filters": [
            {"member": "view_label_sensor_devices.name", "operator": "startsWith", "values": [123]},
            {"member": "view_label_sensor_devices.name", "operator": "notSet"},
            {"member": "view_label_sensor_devices.time", "operator": "inDateRange", "values": ["last 7 days"]},
            {"member": "view_label_sensor_devices.time", "operator": "afterOrOnDate", "values": ["2026-05-07"]},
        ],
        "ungrouped": True,
    })
    validate_semantic_query(
        query,
        catalog_from_meta(_meta()),
        requested_max_rows=50,
        hard_max_rows=1000,
    )
    filters = query.to_cube_query()["filters"]
    assert filters[0]["values"] == ["123"]
    assert "values" not in filters[1]


def test_filter_rejects_unknown_operators_and_nested_values() -> None:
    with pytest.raises(ValidationError):
        SemanticQuery.model_validate({
            "query_mode": "view",
            "models": ["view_label_sensor_devices"],
            "dimensions": ["view_label_sensor_devices.name"],
            "filters": [{
                "member": "view_label_sensor_devices.name",
                "operator": "unknown",
                "values": [["nested"]],
            }],
            "ungrouped": True,
        })


@pytest.mark.parametrize(
    "payload",
    [
        {
            "query_mode": "view",
            "models": ["view_label_sensor_devices"],
            "measures": ["view_label_sensor_devices.name"],
        },
        {
            "query_mode": "view",
            "models": ["view_label_sensor_devices"],
            "dimensions": ["view_label_sensor_devices.name"],
            "timeDimensions": [{"dimension": "view_label_sensor_devices.name", "dateRange": "today"}],
        },
        {
            "query_mode": "view",
            "models": ["view_label_sensor_devices"],
            "dimensions": ["hydrology_boreholes.name"],
        },
        {
            "query_mode": "view",
            "models": ["view_label_sensor_devices"],
            "dimensions": ["view_single_factor_alarms.name"],
        },
        {
            "query_mode": "view",
            "models": ["base_private"],
            "dimensions": ["base_private.name"],
        },
        {
            "query_mode": "view",
            "models": ["view_label_sensor_devices"],
            "dimensions": ["view_label_sensor_devices.unknown"],
        },
        {
            "query_mode": "view",
            "models": ["view_label_sensor_devices"],
            "measures": ["view_label_sensor_devices.count"],
            "dimensions": ["view_label_sensor_devices.name"],
            "ungrouped": True,
        },
        {
            "query_mode": "view",
            "models": ["view_label_sensor_devices"],
            "dimensions": ["view_label_sensor_devices.name"],
            "filters": [{
                "and": [
                    {"member": "view_label_sensor_devices.name", "operator": "equals", "values": ["A"]},
                    {"member": "view_label_sensor_devices.count", "operator": "gt", "values": [1]},
                ],
            }],
        },
    ],
)
def test_validation_rejects_invalid_members_and_types(payload: dict) -> None:
    query = SemanticQuery.model_validate(payload)
    with pytest.raises(SemanticQueryValidationError):
        validate_semantic_query(
            query,
            catalog_from_meta(_meta()),
            requested_max_rows=50,
            hard_max_rows=1000,
        )


async def test_cube_client_caches_meta_posts_load_and_sends_token() -> None:
    requests: list[httpx.Request] = []
    load_calls = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal load_calls
        requests.append(request)
        if request.url.path == "/cubejs-api/v1/meta":
            return httpx.Response(200, json=_meta())
        load_calls += 1
        if load_calls == 1:
            return httpx.Response(200, json={"error": "Continue wait"})
        return httpx.Response(200, json={
            "data": [{"view_label_sensor_devices.count": "2"}],
            "annotation": {
                "measures": {
                    "view_label_sensor_devices.count": {"title": "采样数", "type": "number"},
                },
            },
        })

    transport = httpx.MockTransport(handler)
    async with httpx.AsyncClient(transport=transport) as http_client:
        client = CubeClient(
            base_url="http://cube.internal/cubejs-api/v1",
            token="secret-token",
            timeout_seconds=1,
            continue_wait_retries=2,
            meta_cache_ttl_seconds=60,
            client=http_client,
            continue_wait_delay_seconds=0,
        )
        assert await client.get_meta() == await client.get_meta()
        response = await client.load({"measures": ["view_label_sensor_devices.count"]})
    assert sum(request.url.path == "/cubejs-api/v1/meta" for request in requests) == 1
    assert load_calls == 2
    assert all(request.headers["authorization"] == "secret-token" for request in requests)
    assert requests[-1].method == "POST"
    columns, rows = normalize_cube_response(response)
    assert columns[0].title == "采样数"
    assert rows == [{"view_label_sensor_devices.count": 2}]


async def test_cube_client_redacts_http_errors_and_exhausts_continue_wait() -> None:
    def forbidden(request: httpx.Request) -> httpx.Response:
        return httpx.Response(400, text="secret-token at http://cube.internal/v1/load")

    async with httpx.AsyncClient(transport=httpx.MockTransport(forbidden)) as http_client:
        client = CubeClient(
            base_url="http://cube.internal/cubejs-api/v1",
            token="secret-token",
            timeout_seconds=1,
            continue_wait_retries=0,
            meta_cache_ttl_seconds=60,
            client=http_client,
        )
        with pytest.raises(CubeClientError) as captured:
            await client.load({"measures": ["x"]})
    assert captured.value.retryable_by_model is True
    assert "secret-token" not in str(captured.value)
    assert "cube.internal" not in str(captured.value)

    def waiting(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"error": "Continue wait"})

    async with httpx.AsyncClient(transport=httpx.MockTransport(waiting)) as http_client:
        client = CubeClient(
            base_url="http://cube.internal/cubejs-api/v1",
            token=None,
            timeout_seconds=1,
            continue_wait_retries=1,
            meta_cache_ttl_seconds=60,
            client=http_client,
            continue_wait_delay_seconds=0,
        )
        with pytest.raises(CubeClientError, match="Continue wait"):
            await client.load({"measures": ["x"]})


async def test_cube_client_closes_method_scoped_clients(monkeypatch: pytest.MonkeyPatch) -> None:
    instances = []

    class ScopedClient:
        def __init__(self, *, timeout: float) -> None:
            self.timeout = timeout
            self.closed = False
            self.calls = 0
            instances.append(self)

        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc, traceback) -> None:
            self.closed = True

        async def request(self, method: str, url: str, **kwargs) -> httpx.Response:
            self.calls += 1
            if url.endswith("/meta"):
                return httpx.Response(200, json=_meta())
            if self.calls == 1:
                return httpx.Response(200, json={"error": "Continue wait"})
            return httpx.Response(200, json={"data": []})

    monkeypatch.setattr(
        "app.agents.scenarios.hydrology_semantic_query.client.httpx.AsyncClient",
        ScopedClient,
    )
    client = CubeClient(
        base_url="http://cube.internal/cubejs-api/v1",
        token=None,
        timeout_seconds=1,
        continue_wait_retries=1,
        meta_cache_ttl_seconds=60,
        continue_wait_delay_seconds=0,
    )
    assert await client.get_meta() == await client.get_meta()
    await client.load({"measures": ["x"]})
    assert len(instances) == 2
    assert instances[1].calls == 2
    assert all(item.closed for item in instances)

    async def failed_request(self, method: str, url: str, **kwargs) -> httpx.Response:
        return httpx.Response(500, text="failed")

    monkeypatch.setattr(ScopedClient, "request", failed_request)
    with pytest.raises(CubeClientError):
        await client.load({"measures": ["x"]})
    assert instances[-1].closed is True

    async def cancelled_request(self, method: str, url: str, **kwargs) -> httpx.Response:
        raise asyncio.CancelledError

    monkeypatch.setattr(ScopedClient, "request", cancelled_request)
    with pytest.raises(asyncio.CancelledError):
        await client.load({"measures": ["x"]})
    assert instances[-1].closed is True
