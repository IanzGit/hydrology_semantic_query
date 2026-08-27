from __future__ import annotations

from typing import Any

from .models import (
    CatalogMember,
    CatalogModel,
    SemanticCatalog,
)


class SemanticCatalogError(ValueError):
    pass


def _meta(payload: dict[str, Any]) -> dict[str, Any]:
    value = payload.get("meta")
    return value if isinstance(value, dict) else {}


def _strings(value: Any) -> tuple[str, ...]:
    if not isinstance(value, list):
        return ()
    return tuple(str(item) for item in value if item is not None and str(item))


def _granularities(raw_member: dict[str, Any]) -> tuple[str, ...]:
    values: list[str] = []
    for item in raw_member.get("granularities") or []:
        value = item.get("name") if isinstance(item, dict) else item
        if value:
            values.append(str(value))
    return tuple(values)


def _folder_members(raw_model: dict[str, Any]) -> tuple[tuple[str, ...], dict[str, str]]:
    names: list[str] = []
    member_folders: dict[str, str] = {}
    for item in raw_model.get("folders") or []:
        if not isinstance(item, dict) or not item.get("name"):
            continue
        name = str(item["name"])
        names.append(name)
        for member in item.get("members") or []:
            member_folders[str(member)] = name
    return tuple(names), member_folders


def _hierarchy_members(raw_model: dict[str, Any]) -> tuple[tuple[str, ...], dict[str, str]]:
    names: list[str] = []
    member_hierarchies: dict[str, str] = {}
    for item in raw_model.get("hierarchies") or []:
        if not isinstance(item, dict) or not item.get("name"):
            continue
        name = str(item["name"])
        names.append(name)
        for member in item.get("levels") or []:
            value = member.get("name") if isinstance(member, dict) else member
            if value:
                member_hierarchies[str(value)] = name
    return tuple(names), member_hierarchies


def _default_projection(model_name: str, meta: dict[str, Any]) -> tuple[str, ...]:
    return tuple(
        value if "." in value else f"{model_name}.{value}"
        for value in _strings(meta.get("default_projection"))
    )


def _is_governed_model(raw_model: dict[str, Any]) -> bool:
    model_type = raw_model.get("type")
    return model_type in {"view", "cube"} and raw_model.get("public") is not False


def catalog_from_meta(payload: dict[str, Any]) -> SemanticCatalog:
    raw_models = payload.get("cubes")
    if not isinstance(raw_models, list):
        raise SemanticCatalogError("Cube /meta 响应缺少 cubes 列表")
    models: dict[str, CatalogModel] = {}
    for raw_model in raw_models:
        if not isinstance(raw_model, dict) or not _is_governed_model(raw_model):
            continue
        if not raw_model.get("name"):
            raise SemanticCatalogError("公开 model 缺少名称")
        name = str(raw_model["name"])
        if name in models:
            raise SemanticCatalogError(f"公开 model 名称重复：{name}")
        model_type = str(raw_model["type"])
        meta = _meta(raw_model)
        folders, member_folders = _folder_members(raw_model)
        hierarchies, member_hierarchies = _hierarchy_members(raw_model)
        members: dict[str, CatalogMember] = {}
        for group_name, member_type in (
            ("measures", "measure"),
            ("dimensions", "dimension"),
            ("segments", "segment"),
        ):
            for raw_member in raw_model.get(group_name) or []:
                if (
                    not isinstance(raw_member, dict)
                    or not raw_member.get("name")
                    or raw_member.get("public") is False
                ):
                    continue
                member_name = str(raw_member["name"])
                if not member_name.startswith(f"{name}."):
                    raise SemanticCatalogError(
                        f"成员不属于 model {name}：{member_name}"
                    )
                if member_name in members:
                    raise SemanticCatalogError(f"成员名称重复：{member_name}")
                member_meta = _meta(raw_member)
                short_name = member_name.partition(".")[2]
                members[member_name] = CatalogMember(
                    name=member_name,
                    title=str(
                        raw_member.get("shortTitle")
                        or raw_member.get("title")
                        or member_name
                    ),
                    description=(
                        str(raw_member["description"])
                        if raw_member.get("description") is not None
                        else None
                    ),
                    member_type=member_type,
                    data_type=str(
                        raw_member.get("type")
                        or ("boolean" if member_type == "segment" else "string")
                    ),
                    ai_context=(
                        str(member_meta["ai_context"])
                        if member_meta.get("ai_context") is not None
                        else None
                    ),
                    granularities=_granularities(raw_member),
                    aliases=_strings(member_meta.get("aliases")),
                    folder=member_folders.get(member_name) or member_folders.get(short_name),
                    hierarchy=(
                        member_hierarchies.get(member_name)
                        or member_hierarchies.get(short_name)
                    ),
                    primary_key=bool(raw_member.get("primaryKey")),
                )
        default_projection = _default_projection(name, meta)
        invalid_projection = [
            member_name
            for member_name in default_projection
            if member_name not in members
            or members[member_name].member_type != "dimension"
        ]
        if invalid_projection:
            raise SemanticCatalogError(
                f"model {name} 的 default_projection 包含不存在或非 dimension 成员："
                + ", ".join(invalid_projection)
            )
        priority = meta.get("priority", meta.get("business_priority", 0.5))
        models[name] = CatalogModel(
            name=name,
            model_type=model_type,
            title=str(raw_model.get("title") or name),
            description=(
                str(raw_model["description"])
                if raw_model.get("description") is not None
                else None
            ),
            ai_context=(
                str(meta["ai_context"])
                if meta.get("ai_context") is not None
                else None
            ),
            members=members,
            folders=folders,
            hierarchies=hierarchies,
            connected_component=raw_model.get("connectedComponent"),
            aliases=_strings(meta.get("aliases")),
            use_cases=_strings(meta.get("use_cases")),
            business_priority=float(priority),
            business_domain=(
                str(meta["business_domain"])
                if meta.get("business_domain") is not None
                else None
            ),
            default_projection=default_projection,
        )
    if not models:
        raise SemanticCatalogError("Cube 目录中不存在受治理的公开水文 model")
    return SemanticCatalog(models=models)
