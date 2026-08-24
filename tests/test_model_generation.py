from __future__ import annotations

import json
from pathlib import Path

import pytest

from ..cube.scripts.generate_cube_models import (
    CubeModelGenerationError,
    base_cube_name,
    build_parser,
    database_url_from_environment,
)
from ..cube.scripts.validate_cube_meta import validate_meta


def test_generator_requires_explicit_tables() -> None:
    parser = build_parser()
    with pytest.raises(SystemExit):
        parser.parse_args([])
    arguments = parser.parse_args([
        "--table",
        "device_info",
        "--table",
        "device_x_value",
    ])
    assert arguments.tables == ["device_info", "device_x_value"]


def test_generated_cube_name_uses_canonical_base_prefix() -> None:
    assert base_cube_name("device_x_value") == "base_device_x_value"


def test_generator_rejects_invalid_database_port(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("CUBEJS_DB_HOST", "127.0.0.1")
    monkeypatch.setenv("CUBEJS_DB_NAME", "hydrology")
    monkeypatch.setenv("CUBEJS_DB_USER", "readonly")
    monkeypatch.setenv("CUBEJS_DB_PORT", "70000")

    with pytest.raises(CubeModelGenerationError, match="端口无效"):
        database_url_from_environment()


def test_meta_validation_uses_interface_catalog_without_local_model_scope() -> None:
    fixture = Path(__file__).with_name("fixtures") / "cube_meta_1_6_70.json"
    payload = json.loads(fixture.read_text(encoding="utf-8"))

    assert validate_meta(payload) == (3, 7)

    payload["cubes"].append({
        "name": "interface_only_cube",
        "type": "cube",
        "public": True,
        "connectedComponent": 2,
        "measures": [],
        "dimensions": [],
        "segments": [],
        "folders": [],
        "hierarchies": [],
    })
    assert validate_meta(payload) == (3, 8)
