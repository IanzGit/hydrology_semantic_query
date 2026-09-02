from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from ..contracts import SemanticCatalogMode


class CatalogMember(BaseModel):
    name: str
    title: str
    member_type: Literal["measure", "dimension", "segment"]
    data_type: str
    description: str | None = None
    ai_context: str | None = None
    granularities: tuple[str, ...] = ()
    aliases: tuple[str, ...] = ()
    folder: str | None = None
    hierarchy: str | None = None
    primary_key: bool = False


class CatalogModel(BaseModel):
    name: str
    model_type: Literal["view", "cube"]
    title: str
    description: str | None = None
    ai_context: str | None = None
    members: dict[str, CatalogMember] = Field(default_factory=dict)
    folders: tuple[str, ...] = ()
    hierarchies: tuple[str, ...] = ()
    connected_component: str | int | None = None
    aliases: tuple[str, ...] = ()
    use_cases: tuple[str, ...] = ()
    business_priority: float = Field(default=0.5, ge=0, le=1)
    business_domain: str | None = None
    default_projection: tuple[str, ...] = ()


class SemanticCatalog(BaseModel):
    models: dict[str, CatalogModel] = Field(default_factory=dict)


class CatalogContextItem(BaseModel):
    item_type: Literal["model", "member", "view_folder", "join_component"]
    name: str
    model_name: str | None = None
    score: float | None = None
    payload: dict[str, Any] = Field(default_factory=dict)

    model_config = ConfigDict(extra="forbid")


class SemanticContext(BaseModel):
    strategy: SemanticCatalogMode
    items: list[CatalogContextItem] = Field(default_factory=list)
    retrieval_round: int = Field(default=1, ge=1)

    model_config = ConfigDict(extra="forbid")


__all__ = [
    "CatalogContextItem",
    "CatalogMember",
    "CatalogModel",
    "SemanticCatalog",
    "SemanticContext",
]

