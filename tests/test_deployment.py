from __future__ import annotations

import io
import json
import re
from pathlib import Path
from urllib.error import HTTPError, URLError

import pytest
import yaml

from ..query_child import config as query_config
from ..query_child.client import catalog_from_meta
from ..semantic.scripts import validate_cube_meta
from ..semantic.scripts.validate_cube_meta import (
    MetaValidationError,
    validate_meta,
)
from ..semantic.start import rebuild_vector_index
from ..semantic.start.cube_config import (
    CubeConfigError,
    load_cube_config,
    redact_text,
)
from .catalog_expectations import (
    PRIVATE_MEMBERS,
    PUBLIC_CUBES,
    PUBLIC_JOIN_EDGES,
    PUBLIC_VIEWS,
)

SCENARIO_ROOT = Path(
    "app/agents/scenarios/cqccri_smart_query/subgraph/hydrology_semantic_query"
)
ENV_ROOT = SCENARIO_ROOT / "env"
START_ROOT = SCENARIO_ROOT / "semantic/start"
MONITOR_MODEL_ROOT = Path(
    "app/agents/scenarios/cqccri_smart_query/subgraph/hydrology_semantic_query/semantic/model/hydrology_monitor_model"
)
META_FIXTURE = Path(
    "app/agents/scenarios/cqccri_smart_query/subgraph/hydrology_semantic_query/tests/fixtures/cube_meta_1_6_70.json"
)
FIXTURE_PUBLIC_CUBES = frozenset(
    {
        "base_device_info",
        "base_device_x_value",
        "base_label",
        "base_label_sensor",
        "base_multifactor_sensor",
        "base_warn_state_info",
        "base_water_warn_sensor_set",
    }
)
FIXTURE_PUBLIC_MODELS = PUBLIC_VIEWS | FIXTURE_PUBLIC_CUBES


def _meta() -> dict:
    return json.loads(META_FIXTURE.read_text(encoding="utf-8"))


def _fixture_primary_key_members(payload: dict) -> set[str]:
    return {
        member["name"]
        for cube in payload["cubes"]
        for member in cube["dimensions"]
        if member.get("public") is not False
        and member.get("primaryKey") is True
    }


def test_compose_is_local_pinned_and_mounts_scenario_model_read_only() -> None:
    payload = yaml.safe_load(
        (START_ROOT / "docker-compose.yml").read_text(encoding="utf-8")
    )
    cube = payload["services"]["cube"]
    assert cube["image"] == "cubejs/cube:v1.6.70"
    assert cube["network_mode"] == "host"
    assert cube["env_file"] == ["${CUBE_ENV_FILE}"]
    assert "ports" not in cube
    assert "extra_hosts" not in cube
    assert "restart" not in cube
    assert cube["volumes"] == ["${CUBE_MODEL_DIR}:/cube/conf/model:ro"]
    assert cube["environment"] == {
        "PORT": "${CUBE_PORT}",
        "CUBEJS_DEV_MODE": "${CUBEJS_DEV_MODE}",
    }
    assert "/readyz" in cube["healthcheck"]["test"][-1]
    assert all("latest" not in str(value) for value in cube.values())


def test_deployment_environment_template_contains_required_settings() -> None:
    content = (ENV_ROOT / "cube.env.example").read_text(encoding="utf-8")
    keys = {
        line.partition("=")[0]
        for line in content.splitlines()
        if line and not line.startswith("#")
    }
    assert keys == {
        "CUBEJS_DB_HOST",
        "CUBEJS_DB_NAME",
        "CUBEJS_DB_PASS",
        "CUBEJS_DB_PORT",
        "CUBEJS_DB_TYPE",
        "CUBEJS_DB_USER",
        "CUBEJS_DEV_MODE",
        "CUBE_HOST",
        "CUBE_PORT",
        "MODEL_NAME",
    }
    assert "# MODEL_NAME=hydrology_monitor_model" in content
    assert "# CUBEJS_DB_TYPE=mssql" in content
    assert "# CUBEJS_DB_NAME=MainData" in content


def test_agent_environment_template_matches_runtime_settings() -> None:
    keys = {
        line.partition("=")[0]
        for line in (ENV_ROOT / "agent.env.example").read_text(encoding="utf-8").splitlines()
        if line and not line.startswith("#")
    }

    assert query_config._ENV_FILE == (ENV_ROOT / "agent.env").resolve()
    assert keys == {
        "HYDROLOGY_SEMANTIC_QUERY_AUTO_FULL_CONTEXT_MAX_CHARS",
        "HYDROLOGY_SEMANTIC_QUERY_CATALOG_STRATEGY",
        "HYDROLOGY_SEMANTIC_QUERY_CONTEXT_TOP_K",
        "HYDROLOGY_SEMANTIC_QUERY_CONTINUE_WAIT_RETRIES",
        "HYDROLOGY_SEMANTIC_QUERY_CUBE_TOKEN",
        "HYDROLOGY_SEMANTIC_QUERY_CUBE_URL",
        "HYDROLOGY_SEMANTIC_QUERY_EMBEDDING_BATCH_SIZE",
        "HYDROLOGY_SEMANTIC_QUERY_EMBEDDING_CONCURRENCY",
        "HYDROLOGY_SEMANTIC_QUERY_EMBEDDING_MODEL",
        "HYDROLOGY_SEMANTIC_QUERY_MAX_AGENT_ITERATIONS",
        "HYDROLOGY_SEMANTIC_QUERY_MAX_QUERY_ROUNDS",
        "HYDROLOGY_SEMANTIC_QUERY_MAX_RETRIES",
        "HYDROLOGY_SEMANTIC_QUERY_META_CACHE_TTL_SECONDS",
        "HYDROLOGY_SEMANTIC_QUERY_MODEL_TOP_K",
        "HYDROLOGY_SEMANTIC_QUERY_TIMEOUT_SECONDS",
        "HYDROLOGY_SEMANTIC_QUERY_TIMEZONE",
        "HYDROLOGY_SEMANTIC_QUERY_VECTOR_INDEX_PATH",
    }


def test_monitor_model_uses_device_cube_and_exposes_point_address_name() -> None:
    assert not (MONITOR_MODEL_ROOT / "views/device_info.yml").exists()
    device_cube = yaml.safe_load(
        (MONITOR_MODEL_ROOT / "cubes/device_info.yml").read_text(encoding="utf-8")
    )["cubes"][0]
    assert device_cube["meta"]["aliases"] == ["水文设备", "水文监测设备定义"]
    assert device_cube["meta"]["default_projection"] == [
        "code",
        "name",
        "model",
        "factory_name",
        "installation_address",
        "enabled_at",
    ]

    point_view = yaml.safe_load(
        (MONITOR_MODEL_ROOT / "views/point_deployment_analysis.yml").read_text(
            encoding="utf-8"
        )
    )["views"][0]
    assert "address_name" in point_view["meta"]["default_projection"]
    address_cube = next(
        cube
        for cube in point_view["cubes"]
        if cube["join_path"] == "base_point_info.base_mine_address"
    )
    assert address_cube["includes"] == [{"name": "name", "alias": "address_name"}]


def test_monitor_views_expose_curated_business_members() -> None:
    member_counts = {}
    for path in sorted((MONITOR_MODEL_ROOT / "views").glob("*.yml")):
        view = yaml.safe_load(path.read_text(encoding="utf-8"))["views"][0]
        members = {
            include if isinstance(include, str) else include.get("alias", include["name"])
            for cube in view["cubes"]
            for include in cube["includes"]
        }
        assert set(view["meta"]["default_projection"]) <= members
        member_counts[view["name"]] = len(members)

    assert member_counts == {
        "view_exception_event_analysis": 35,
        "view_historical_statistics_analysis": 34,
        "view_historical_trend_analysis": 32,
        "view_point_deployment_analysis": 49,
        "view_realtime_monitoring_analysis": 45,
    }


def test_start_script_prepares_vector_index_after_cube_is_ready() -> None:
    script = (START_ROOT / "start.sh").read_text(encoding="utf-8")
    compose_start = "up -d --force-recreate"

    assert script.index("config --quiet") < script.index(compose_start)
    assert script.index(compose_start) < script.index("validate_cube_meta")
    assert script.index("validate_cube_meta") < script.index("rebuild_vector_index.py")
    assert "CUBE_READY_URL" in script
    assert "time.monotonic() + 120" in script
    assert "elapsed=" in script
    assert 'cd "$project_dir"' in script
    assert 'vector_index_path="$start_dir/../cache/semantic-catalog-vectors.sqlite3"' in script
    assert script.count("print_cube_logs") == 5
    assert "请重启Agent" in script


def _write_cube_config_fixture(tmp_path: Path) -> tuple[Path, Path]:
    models_root = tmp_path / "model"
    env_file = tmp_path / "cube.env"
    env_file.write_text(
        "CUBE_HOST=127.0.0.1\n"
        "CUBE_PORT=4000\n"
        "CUBEJS_DEV_MODE=true\n"
        "MODEL_NAME=hydrology_model\n"
        "CUBEJS_DB_TYPE=mysql\n"
        "CUBEJS_DB_HOST=mysql.internal\n"
        "CUBEJS_DB_PORT=3306\n"
        "CUBEJS_DB_NAME=hydrology\n"
        "CUBEJS_DB_USER=readonly\n"
        "CUBEJS_DB_PASS=top-secret\n",
        encoding="utf-8",
    )
    for directory_name in ("cubes", "views"):
        directory = models_root / "hydrology_model" / directory_name
        directory.mkdir(parents=True)
        (directory / "model.yml").write_text("cubes: []\n", encoding="utf-8")
    return env_file, models_root


def test_cube_config_resolves_model_and_database(tmp_path: Path) -> None:
    env_file, models_root = _write_cube_config_fixture(tmp_path)

    config = load_cube_config(env_file, models_root=models_root)

    assert config.model_name == "hydrology_model"
    assert config.database_type == "mysql"
    assert config.database_host == "mysql.internal"
    assert config.database_name == "hydrology"
    assert config.sensitive_values == ("top-secret",)


def test_cube_config_rejects_duplicate_active_settings(tmp_path: Path) -> None:
    env_file, models_root = _write_cube_config_fixture(tmp_path)
    env_file.write_text(
        f"{env_file.read_text(encoding='utf-8')}CUBEJS_DB_TYPE=mssql\n",
        encoding="utf-8",
    )

    with pytest.raises(CubeConfigError, match="CUBEJS_DB_TYPE.*重复"):
        load_cube_config(env_file, models_root=models_root)


@pytest.mark.parametrize(
    ("key", "value", "message"),
    [
        ("CUBE_PORT", "70000", "CUBE_PORT"),
        ("CUBEJS_DB_PORT", "70000", "CUBEJS_DB_PORT"),
        ("CUBEJS_DEV_MODE", "enabled", "CUBEJS_DEV_MODE"),
    ],
)
def test_cube_config_rejects_invalid_value(
    tmp_path: Path,
    key: str,
    value: str,
    message: str,
) -> None:
    env_file, models_root = _write_cube_config_fixture(tmp_path)
    content = env_file.read_text(encoding="utf-8")
    env_file.write_text(
        re.sub(rf"^{key}=.*$", f"{key}={value}", content, flags=re.MULTILINE),
        encoding="utf-8",
    )

    with pytest.raises(CubeConfigError, match=message):
        load_cube_config(env_file, models_root=models_root)


def test_cube_config_rejects_missing_file(tmp_path: Path) -> None:
    missing_file = tmp_path / "missing.env"

    with pytest.raises(CubeConfigError, match="缺少Cube配置"):
        load_cube_config(missing_file, models_root=tmp_path / "model")


@pytest.mark.parametrize("missing_key", ["MODEL_NAME", "CUBEJS_DB_PASS"])
def test_cube_config_rejects_missing_value(
    tmp_path: Path,
    missing_key: str,
) -> None:
    env_file, models_root = _write_cube_config_fixture(tmp_path)
    env_file.write_text(
        "\n".join(
            line
            for line in env_file.read_text(encoding="utf-8").splitlines()
            if not line.startswith(f"{missing_key}=")
        ),
        encoding="utf-8",
    )

    with pytest.raises(CubeConfigError, match=missing_key):
        load_cube_config(env_file, models_root=models_root)


def test_cube_config_rejects_missing_model_directory(tmp_path: Path) -> None:
    env_file, models_root = _write_cube_config_fixture(tmp_path)
    (models_root / "hydrology_model" / "views" / "model.yml").unlink()

    with pytest.raises(CubeConfigError, match="views"):
        load_cube_config(env_file, models_root=models_root)


def test_cube_config_redacts_secrets() -> None:
    value = "connection failed password=top-secret CUBEJS_DB_PASS=top-secret"

    redacted = redact_text(value, ("top-secret",))

    assert "top-secret" not in redacted
    assert redacted == "connection failed password=*** CUBEJS_DB_PASS=***"


def test_vector_cache_is_scenario_local() -> None:
    project_root = START_ROOT.parents[7]
    scenario_cache = START_ROOT.parent / "cache"

    assert (scenario_cache / "semantic-catalog-vectors.sqlite3").is_file()
    assert (scenario_cache / "semantic-catalog-vectors.sqlite3.lock").is_file()
    assert not (project_root / "cache").exists()


def test_rebuild_script_resolves_relative_index_path_from_project_root() -> None:
    assert rebuild_vector_index.resolve_index_path("cache/index.sqlite3") == (
        rebuild_vector_index.PROJECT_ROOT / "cache" / "index.sqlite3"
    )


def test_rebuild_script_propagates_failure(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    async def failed(settings, *, force: bool):
        raise RuntimeError("embedding failed")

    monkeypatch.setattr(
        rebuild_vector_index,
        "load_hydrology_semantic_query_settings",
        lambda: object(),
    )
    monkeypatch.setattr(rebuild_vector_index, "prepare_vector_index", failed)

    assert rebuild_vector_index.main([]) == 1
    assert "embedding failed" in capsys.readouterr().err


def test_rebuild_script_forwards_force_and_reports_success(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    calls: list[bool] = []

    async def prepared(settings, *, force: bool):
        calls.append(force)
        return "built_disk", 12, tmp_path / "index.sqlite3"

    monkeypatch.setattr(
        rebuild_vector_index,
        "load_hydrology_semantic_query_settings",
        lambda: object(),
    )
    monkeypatch.setattr(rebuild_vector_index, "prepare_vector_index", prepared)

    assert rebuild_vector_index.main(["--force"]) == 0
    assert calls == [True]
    assert "documents=12" in capsys.readouterr().out


def test_meta_validation_accepts_expected_catalog() -> None:
    payload = _meta()
    validate_meta(payload)
    assert {
        item["name"] for item in payload["cubes"] if item.get("public") is not False
    } == FIXTURE_PUBLIC_MODELS


def test_meta_validation_ignores_private_cubes() -> None:
    payload = _meta()
    payload["cubes"].append(
        {"name": "base_hydrology_monitoring", "public": False, "dimensions": []}
    )
    validate_meta(payload)


def test_meta_validation_accepts_new_well_formed_public_cube() -> None:
    payload = _meta()
    payload["cubes"].append(
        {
            "name": "base_hydrology_monitoring",
            "type": "cube",
            "public": True,
            "connectedComponent": 2,
            "measures": [],
            "dimensions": [],
            "segments": [],
            "folders": [],
            "hierarchies": [],
        }
    )
    assert validate_meta(payload) == (len(PUBLIC_VIEWS), len(FIXTURE_PUBLIC_CUBES) + 1)


def test_meta_validation_accepts_partial_well_formed_catalog() -> None:
    payload = _meta()
    removed = payload["cubes"].pop()

    view_count, cube_count = validate_meta(payload)

    assert view_count + cube_count == len(FIXTURE_PUBLIC_MODELS) - 1
    assert removed["name"] not in catalog_from_meta(payload).models


@pytest.mark.parametrize("mutation", ["illegal_model", "duplicate_model"])
def test_meta_validation_rejects_malformed_public_models(mutation: str) -> None:
    payload = _meta()
    if mutation == "illegal_model":
        payload["cubes"].append({"name": "unexpected", "dimensions": []})
    else:
        payload["cubes"].append(payload["cubes"][0])
    with pytest.raises(MetaValidationError):
        validate_meta(payload)


def test_meta_validation_rejects_wrong_identifier_type_and_member_owner() -> None:
    payload = _meta()
    fixture_primary_key_members = _fixture_primary_key_members(payload)
    original_name = sorted(fixture_primary_key_members)[0]
    member = next(
        member
        for cube in payload["cubes"]
        for member in cube["dimensions"]
        if member["name"] == original_name
    )
    member["name"] = "wrong_model.id"
    with pytest.raises(MetaValidationError, match="成员不属于"):
        validate_meta(payload)
    payload = _meta()
    member = next(
        member
        for cube in payload["cubes"]
        for member in cube["dimensions"]
        if member["name"] in fixture_primary_key_members
    )
    member["type"] = "number"
    with pytest.raises(MetaValidationError, match="string"):
        validate_meta(payload)


def test_meta_validation_accepts_isolated_cube_without_connected_component() -> None:
    payload = _meta()
    view = next(item for item in payload["cubes"] if item["name"] in PUBLIC_VIEWS)
    view["type"] = "cube"
    view.pop("connectedComponent", None)
    assert validate_meta(payload) == (
        len(PUBLIC_VIEWS) - 1,
        len(FIXTURE_PUBLIC_CUBES) + 1,
    )


def test_meta_fixture_matches_declared_join_graph_and_private_visibility() -> None:
    payload = _meta()
    actual_join_edges = {
        cube["name"]: frozenset(
            edge["target"] for edge in cube.get("meta", {}).get("join_edges", [])
        )
        for cube in payload["cubes"]
        if cube["name"] in PUBLIC_CUBES
    }
    actual_members = {
        member["name"]: member
        for cube in payload["cubes"]
        for group in ("measures", "dimensions", "segments")
        for member in cube[group]
    }
    visible_members = {
        member_name
        for model in catalog_from_meta(payload).models.values()
        for member_name in model.members
    }

    assert actual_join_edges == {
        name: PUBLIC_JOIN_EDGES[name] for name in actual_join_edges
    }
    assert all(
        actual_members[name].get("public") is False
        for name in PRIVATE_MEMBERS & actual_members.keys()
    )
    assert PRIVATE_MEMBERS.isdisjoint(visible_members)


@pytest.mark.parametrize("payload", [{}, {"cubes": None}, {"cubes": "invalid"}])
def test_meta_validation_rejects_malformed_payload(payload: dict) -> None:
    with pytest.raises(MetaValidationError):
        validate_meta(payload)


def test_fetch_meta_rejects_non_json_response(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(validate_cube_meta, "urlopen", lambda url, timeout: io.BytesIO(b"invalid"))
    with pytest.raises(MetaValidationError, match="JSON"):
        validate_cube_meta.fetch_meta("http://cube/meta", 1)


@pytest.mark.parametrize(
    "error",
    [
        HTTPError("http://cube/meta", 500, "compile failed", {}, None),
        URLError(TimeoutError("timed out")),
    ],
)
def test_fetch_meta_rejects_http_and_network_errors(
    monkeypatch: pytest.MonkeyPatch,
    error: Exception,
) -> None:
    def failed_request(url: str, timeout: float):
        raise error

    monkeypatch.setattr(validate_cube_meta, "urlopen", failed_request)
    with pytest.raises(MetaValidationError):
        validate_cube_meta.fetch_meta("http://cube/meta", 1)
