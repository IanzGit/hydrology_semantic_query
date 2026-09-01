from __future__ import annotations

import argparse
import json
import sys
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import urlopen

from ...client import catalog_from_meta


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


def validate_meta(payload: dict[str, Any]) -> tuple[int, int]:
    cubes = payload.get("cubes")
    if not isinstance(cubes, list):
        raise MetaValidationError("Cube /meta 响应缺少 cubes 列表")
    if any(not isinstance(cube, dict) for cube in cubes):
        raise MetaValidationError("Cube /meta 的 cubes 只能包含对象")
    public_models = [
        cube
        for cube in cubes
        if cube.get("public") is not False
    ]
    if not public_models:
        raise MetaValidationError("Cube /meta 不存在公开 model")
    named_models = {
        str(cube.get("name")): cube for cube in public_models if cube.get("name")
    }
    if len(public_models) != len(named_models):
        raise MetaValidationError("公开 models 中存在重复或无名项")
    members: dict[str, dict[str, Any]] = {}
    for model_name, model in named_models.items():
        model_type = model.get("type")
        if model_type not in {"view", "cube"}:
            raise MetaValidationError(
                f"model {model_name} 类型必须是 view 或 cube，实际为 {model_type}"
            )
        for key in ("measures", "dimensions", "segments", "folders", "hierarchies"):
            if not isinstance(model.get(key), list):
                raise MetaValidationError(f"model {model_name} 缺少 {key} 列表")
        for key in ("measures", "dimensions", "segments"):
            for member in model[key]:
                if not isinstance(member, dict) or not member.get("name"):
                    raise MetaValidationError(
                        f"model {model_name} 的 {key} 存在无效成员"
                    )
                member_name = str(member["name"])
                if not member_name.startswith(f"{model_name}."):
                    raise MetaValidationError(
                        f"成员不属于 model {model_name}：{member_name}"
                    )
                if member_name in members:
                    raise MetaValidationError(f"成员名称重复：{member_name}")
                members[member_name] = member
    for name, member in sorted(members.items()):
        if (
            member.get("public") is not False
            and member.get("primaryKey") is True
            and member.get("type") != "string"
        ):
            raise MetaValidationError(
                f"Cube 主键成员 {name} 类型应为 string，实际为 {member.get('type')}"
            )
    try:
        catalog = catalog_from_meta(payload)
    except ValueError as exc:
        raise MetaValidationError(str(exc)) from exc
    view_count = sum(model.model_type == "view" for model in catalog.models.values())
    cube_count = sum(model.model_type == "cube" for model in catalog.models.values())
    return view_count, cube_count


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="验证水文语义 Cube /meta 目录")
    parser.add_argument("--url", required=True)
    parser.add_argument("--timeout-seconds", type=float, default=30)
    return parser


def main() -> int:
    arguments = build_parser().parse_args()
    try:
        view_count, cube_count = validate_meta(
            fetch_meta(arguments.url, arguments.timeout_seconds)
        )
    except MetaValidationError as exc:
        print(f"验证失败：{exc}", file=sys.stderr)
        return 1
    print(
        f"验证通过：Cube 接口暴露 {view_count} 个 View 和 {cube_count} 个 Cube，"
        "全部主键成员类型均为 string。"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
