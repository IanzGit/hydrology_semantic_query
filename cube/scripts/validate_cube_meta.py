from __future__ import annotations

import argparse
import json
import runpy
import sys
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import urlopen

_contract = runpy.run_path(
    str(Path(__file__).resolve().parents[1] / "contract.py")
)
PUBLIC_VIEWS = _contract["PUBLIC_VIEWS"]
PUBLIC_CUBES = _contract["PUBLIC_CUBES"]
PUBLIC_MODELS = _contract["PUBLIC_MODELS"]
CUBE_JOIN_EDGES = _contract["CUBE_JOIN_EDGES"]
PRIVATE_MEMBERS = _contract["PRIVATE_MEMBERS"]
STRING_MEMBERS = _contract["STRING_MEMBERS"]


class MetaValidationError(ValueError):
    pass


def fetch_meta(url: str, timeout_seconds: float) -> dict[str, Any]:
    try:
        with urlopen(url, timeout=timeout_seconds) as response:
            payload = json.load(response)
    except HTTPError as exc:
        raise MetaValidationError(f"Cube /meta 返回 HTTP {exc.code}") from exc
    except URLError as exc:
        raise MetaValidationError(f"Cube /meta 请求失败：{exc.reason}") from exc
    except (json.JSONDecodeError, UnicodeDecodeError) as exc:
        raise MetaValidationError("Cube /meta 返回了非 JSON 响应") from exc
    if not isinstance(payload, dict):
        raise MetaValidationError("Cube /meta JSON 响应必须是对象")
    return payload


def validate_meta(payload: dict[str, Any]) -> None:
    cubes = payload.get("cubes")
    if not isinstance(cubes, list):
        raise MetaValidationError("Cube /meta 响应缺少 cubes 列表")
    public_models = [
        cube
        for cube in cubes
        if isinstance(cube, dict) and cube.get("public") is not False
    ]
    named_models = {
        str(cube.get("name")): cube for cube in public_models if cube.get("name")
    }
    actual_models = set(named_models)
    if (
        actual_models != PUBLIC_MODELS
        or len(public_models) != len(named_models)
    ):
        missing = sorted(PUBLIC_MODELS - actual_models)
        unexpected = sorted(actual_models - PUBLIC_MODELS)
        details = []
        if missing:
            details.append(f"缺失公开 model：{', '.join(missing)}")
        if unexpected:
            details.append(f"非法公开 model：{', '.join(unexpected)}")
        if len(public_models) != len(named_models):
            details.append("cubes 中存在重复或无名项")
        raise MetaValidationError("；".join(details) or "Cube 公开 model 数量不正确")
    members: dict[str, dict[str, Any]] = {}
    for model_name in PUBLIC_MODELS:
        model = named_models[model_name]
        expected_type = "view" if model_name in PUBLIC_VIEWS else "cube"
        if model.get("type") != expected_type:
            raise MetaValidationError(
                f"model {model_name} 类型应为 {expected_type}，实际为 {model.get('type')}"
            )
        for key in ("measures", "dimensions", "segments", "folders", "hierarchies"):
            if not isinstance(model.get(key), list):
                raise MetaValidationError(f"model {model_name} 缺少 {key} 列表")
        for key in ("measures", "dimensions", "segments"):
            members.update(
                {
                    str(member["name"]): member
                    for member in model[key]
                    if isinstance(member, dict) and member.get("name")
                }
            )
    for cube_name in PUBLIC_CUBES:
        meta = named_models[cube_name].get("meta")
        if not isinstance(meta, dict):
            raise MetaValidationError(f"Cube {cube_name} 缺少 meta")
        edges = {
            str(item.get("target"))
            for item in meta.get("join_edges") or []
            if isinstance(item, dict) and item.get("target")
        }
        if edges != CUBE_JOIN_EDGES[cube_name]:
            raise MetaValidationError(
                f"Cube {cube_name} join_edges 不正确：{sorted(edges)}"
            )
    exposed_private = sorted(
        name
        for name in PRIVATE_MEMBERS
        if name in members and members[name].get("public") is not False
    )
    if exposed_private:
        raise MetaValidationError(
            f"技术成员不应公开：{', '.join(exposed_private)}"
        )
    for name in sorted(STRING_MEMBERS):
        member = members.get(name)
        if member is None:
            raise MetaValidationError(f"Cube /meta 缺少成员：{name}")
        if member.get("type") != "string":
            raise MetaValidationError(
                f"Cube 成员 {name} 类型应为 string，实际为 {member.get('type')}"
            )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="验证水文语义 Cube /meta 目录")
    parser.add_argument("--url", required=True)
    parser.add_argument("--timeout-seconds", type=float, default=30)
    return parser


def main() -> int:
    arguments = build_parser().parse_args()
    try:
        validate_meta(fetch_meta(arguments.url, arguments.timeout_seconds))
    except MetaValidationError as exc:
        print(f"验证失败：{exc}", file=sys.stderr)
        return 1
    print(
        f"验证通过：Cube 已暴露 {len(PUBLIC_VIEWS)} 个治理 View 和 {len(PUBLIC_CUBES)} 个受治理基础 Cube，"
        f"{len(STRING_MEMBERS)} 个 ID 成员类型均为 string。"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
