from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

MODEL_ROOT = Path(
    "app/agents/scenarios/hydrology_semantic_query/semantic/model/hydrology_model"
)


def load_declared_model() -> dict[str, list[dict[str, Any]]]:
    cubes: list[dict[str, Any]] = []
    views: list[dict[str, Any]] = []
    for path in sorted((MODEL_ROOT / "cubes").glob("*.yml")):
        cubes.extend(yaml.safe_load(path.read_text(encoding="utf-8"))["cubes"])
    for path in sorted((MODEL_ROOT / "views").glob("*.yml")):
        views.extend(yaml.safe_load(path.read_text(encoding="utf-8"))["views"])
    return {"cubes": cubes, "views": views}


def _qualified(model_name: str, member_name: str) -> str:
    return member_name if "." in member_name else f"{model_name}.{member_name}"


DECLARED_MODEL = load_declared_model()
PUBLIC_VIEWS = frozenset(
    view["name"] for view in DECLARED_MODEL["views"] if view.get("public") is not False
)
PUBLIC_CUBES = frozenset(
    cube["name"]
    for cube in DECLARED_MODEL["cubes"]
    if cube.get("public") is not False and cube["name"].startswith("base_")
)
PUBLIC_MODELS = PUBLIC_VIEWS | PUBLIC_CUBES
PRIVATE_CUBES = frozenset(
    cube["name"]
    for cube in DECLARED_MODEL["cubes"]
    if cube.get("public") is False or not cube["name"].startswith("base_")
)
PRIVATE_MEMBERS = frozenset(
    _qualified(cube["name"], member["name"])
    for cube in DECLARED_MODEL["cubes"]
    for group in ("measures", "dimensions", "segments")
    for member in cube.get(group, [])
    if member.get("public") is False
)
PUBLIC_IDENTIFIER_MEMBERS = frozenset(
    _qualified(cube["name"], member["name"])
    for cube in DECLARED_MODEL["cubes"]
    if cube.get("public") is not False and cube["name"].startswith("base_")
    for member in cube.get("dimensions", [])
    if member.get("public") is not False
    and (
        member.get("primary_key") is True
        or member["name"] == "id"
        or member["name"] == "relation_key"
        or member["name"].endswith("_id")
        or member["name"].endswith("_ids")
    )
)
PUBLIC_JOIN_EDGES = {
    cube["name"]: frozenset(
        join["name"]
        for join in cube.get("joins", [])
        if join["name"] in PUBLIC_CUBES
    )
    for cube in DECLARED_MODEL["cubes"]
    if cube.get("public") is not False and cube["name"].startswith("base_")
}
