from __future__ import annotations

import os

import pytest

from app.agents.scenarios.hydrology_semantic_query.query_child.tools.run_semantic_query import (
    validate_semantic_query,
)

from ..contracts import SemanticQuery
from ..query_child.client import CubeClient, catalog_from_meta
from ..query_child.config import load_hydrology_semantic_query_settings
from .catalog_expectations import (
    PUBLIC_IDENTIFIER_MEMBERS,
    PUBLIC_MODELS,
)

pytestmark = pytest.mark.skipif(
    os.getenv("HYDROLOGY_SEMANTIC_QUERY_RUN_INTEGRATION") != "1",
    reason="需要真实 Cube 和 MySQL 集成环境",
)


async def test_real_cube_meta_and_business_view_canary_queries() -> None:
    settings = load_hydrology_semantic_query_settings()
    client = CubeClient(
        base_url=settings.cube_url,
        token=settings.cube_token,
        timeout_seconds=settings.timeout_seconds,
        continue_wait_retries=settings.continue_wait_retries,
        meta_cache_ttl_seconds=settings.meta_cache_ttl_seconds,
    )
    meta = await client.get_meta()
    public_names = {
        item["name"]
        for item in meta["cubes"]
        if item.get("public") is not False
    }
    assert public_names == PUBLIC_MODELS
    catalog = catalog_from_meta(meta)
    assert set(catalog.models) == PUBLIC_MODELS
    raw_members = {
        member["name"]: member
        for model in meta["cubes"]
        for group in ("measures", "dimensions", "segments")
        for member in model[group]
    }
    for name in PUBLIC_IDENTIFIER_MEMBERS:
        assert raw_members[name]["type"] == "string"
    assert "base_label_sensor.relation_key" not in catalog.models["base_label_sensor"].members
    canaries = {
        "view_label_sensor_devices": "matched_sensor_count",
        "view_single_factor_alarms": "alarm_event_count",
        "view_multifactor_warnings": "warning_event_count",
    }
    for model, measure in canaries.items():
        query = SemanticQuery.model_validate({
            "query_mode": "view",
            "models": [model],
            "measures": [f"{model}.{measure}"],
        })
        validated = validate_semantic_query(
            query,
            catalog,
            requested_max_rows=settings.max_rows,
            hard_max_rows=settings.hard_max_rows,
        )
        wire_query = validated.query.to_cube_query()
        wire_query["timezone"] = settings.timezone
        response = await client.load(wire_query)
        assert isinstance(response.get("data", []), list)
