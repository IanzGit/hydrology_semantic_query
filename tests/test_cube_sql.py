from __future__ import annotations

import json

import httpx
import pytest

from ..semantic_cube_client import CubeClient, CubeClientError


async def test_cube_client_posts_sql_query_and_parses_statement_and_params() -> None:
    requests: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        if len(requests) == 1:
            return httpx.Response(200, json={"error": "Continue wait"})
        return httpx.Response(200, json={
            "sql": {
                "status": "ok",
                "sql": ["SELECT count(*) FROM device_x_value WHERE sensor_type = ?", ["water"]],
                "query_type": "regular",
            }
        })

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as http_client:
        client = CubeClient(
            base_url="http://cube.internal/cubejs-api/v1",
            token="secret-token",
            timeout_seconds=1,
            continue_wait_retries=1,
            meta_cache_ttl_seconds=60,
            client=http_client,
            continue_wait_delay_seconds=0,
        )
        query = {"measures": ["hydrology_monitoring_devices.monitoring_sensor_count"]}
        statement, params = await client.get_sql(query)

    assert statement == "SELECT count(*) FROM device_x_value WHERE sensor_type = ?"
    assert params == ["water"]
    assert len(requests) == 2
    assert all(request.method == "POST" for request in requests)
    assert all(request.url.path == "/cubejs-api/v1/sql" for request in requests)
    assert all(request.headers["authorization"] == "secret-token" for request in requests)
    assert json.loads(requests[-1].content) == {"query": query}


@pytest.mark.parametrize("payload", [
    {"sql": {"sql": []}},
    {"sql": {"sql": ["SELECT ?", "invalid"]}},
])
async def test_cube_client_rejects_invalid_sql_response(payload: dict) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=payload)

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as http_client:
        client = CubeClient(
            base_url="http://cube.internal/cubejs-api/v1",
            token=None,
            timeout_seconds=1,
            continue_wait_retries=0,
            meta_cache_ttl_seconds=60,
            client=http_client,
        )
        with pytest.raises(CubeClientError) as captured:
            await client.get_sql({"measures": ["x"]})

    assert captured.value.code == "cube_invalid_response"
