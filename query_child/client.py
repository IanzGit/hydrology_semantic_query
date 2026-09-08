from __future__ import annotations

import asyncio
import re
import time
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Any

import httpx

from .models import CatalogMember, CatalogModel, SemanticCatalog


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


class CubeClientError(RuntimeError):
    def __init__(
        self,
        message: str,
        *,
        code: str = "cube_error",
        status_code: int | None = None,
        retryable_by_model: bool = False,
    ) -> None:
        super().__init__(message)
        self.code = code
        self.status_code = status_code
        self.retryable_by_model = retryable_by_model


class CubeClient:
    def __init__(
        self,
        *,
        base_url: str,
        token: str | None,
        timeout_seconds: float,
        continue_wait_retries: int,
        meta_cache_ttl_seconds: float,
        client: httpx.AsyncClient | None = None,
        continue_wait_delay_seconds: float = 0.1,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.token = token
        self.continue_wait_retries = continue_wait_retries
        self.meta_cache_ttl_seconds = meta_cache_ttl_seconds
        self.continue_wait_delay_seconds = continue_wait_delay_seconds
        self.timeout_seconds = timeout_seconds
        self._client = client
        self._meta: dict[str, Any] | None = None
        self._meta_expires_at = 0.0
        self._meta_lock = asyncio.Lock()

    @property
    def headers(self) -> dict[str, str]:
        return {"Authorization": self.token} if self.token else {}

    def _safe_message(self, text: str) -> str:
        safe = text.replace(self.base_url, "<cube>")
        if self.token:
            safe = safe.replace(self.token, "<redacted>")
        safe = re.sub(r'(?i)(authorization["\s:=]+)[^,}\s]+', r"\1<redacted>", safe)
        safe = re.sub(r"https?://[^\s,}]+", "<cube>", safe, flags=re.IGNORECASE)
        return safe[:1000]

    @asynccontextmanager
    async def _client_scope(self) -> AsyncIterator[httpx.AsyncClient]:
        if self._client is not None:
            yield self._client
            return
        async with httpx.AsyncClient(timeout=self.timeout_seconds) as client:
            yield client

    async def _request(
        self,
        client: httpx.AsyncClient,
        method: str,
        path: str,
        **kwargs: Any,
    ) -> dict[str, Any]:
        try:
            response = await client.request(
                method,
                f"{self.base_url}{path}",
                headers=self.headers,
                **kwargs,
            )
        except httpx.TimeoutException as exc:
            raise CubeClientError("连接 Cube 超时", code="cube_timeout") from exc
        except httpx.HTTPError as exc:
            raise CubeClientError(
                self._safe_message(f"Cube 网络请求失败：{exc}"),
                code="cube_network_error",
            ) from exc
        if response.is_error:
            status = response.status_code
            if status in {401, 403}:
                code = "cube_auth_error"
                message = "Cube 认证失败"
            else:
                code = "cube_http_error"
                message = f"Cube HTTP {status}：{self._safe_message(response.text)}"
            raise CubeClientError(
                message,
                code=code,
                status_code=status,
                retryable_by_model=status == 400,
            )
        try:
            payload = response.json()
        except ValueError as exc:
            raise CubeClientError("Cube 返回了非 JSON 响应", code="cube_invalid_response") from exc
        if not isinstance(payload, dict):
            raise CubeClientError("Cube JSON 响应必须是对象", code="cube_invalid_response")
        return payload

    async def get_meta(self, *, force: bool = False) -> dict[str, Any]:
        now = time.monotonic()
        if not force and self._meta is not None and now < self._meta_expires_at:
            return self._meta
        async with self._meta_lock:
            now = time.monotonic()
            if not force and self._meta is not None and now < self._meta_expires_at:
                return self._meta
            async with self._client_scope() as client:
                payload = await self._request(client, "GET", "/meta")
            self._meta = payload
            self._meta_expires_at = now + self.meta_cache_ttl_seconds
            return payload

    async def load(self, query: dict[str, Any]) -> dict[str, Any]:
        async with self._client_scope() as client:
            return await self._query(client, "/load", query)

    async def get_sql(self, query: dict[str, Any]) -> tuple[str, list[Any]]:
        async with self._client_scope() as client:
            payload = await self._query(client, "/sql", query)
        sql_payload = payload.get("sql")
        raw_sql = sql_payload.get("sql") if isinstance(sql_payload, dict) else sql_payload
        if isinstance(raw_sql, str):
            statement = raw_sql
            params: Any = []
        elif isinstance(raw_sql, list) and len(raw_sql) == 2:
            statement, params = raw_sql
        else:
            raise CubeClientError(
                "Cube /sql 响应缺少 SQL",
                code="cube_invalid_response",
            )
        if not isinstance(statement, str) or not statement.strip():
            raise CubeClientError(
                "Cube /sql 响应中的 SQL 必须是非空字符串",
                code="cube_invalid_response",
            )
        if not isinstance(params, list):
            raise CubeClientError(
                "Cube /sql 响应中的参数必须是数组",
                code="cube_invalid_response",
            )
        return statement, params

    async def _query(
        self,
        client: httpx.AsyncClient,
        path: str,
        query: dict[str, Any],
    ) -> dict[str, Any]:
        for attempt in range(self.continue_wait_retries + 1):
            payload = await self._request(client, "POST", path, json={"query": query})
            if payload.get("error") != "Continue wait":
                if payload.get("error"):
                    raise CubeClientError(
                        self._safe_message(f"Cube 查询失败：{payload['error']}"),
                        code="cube_response_error",
                        retryable_by_model=True,
                    )
                return payload
            if attempt < self.continue_wait_retries:
                await asyncio.sleep(self.continue_wait_delay_seconds)
        raise CubeClientError(
            "Cube Continue wait 重试耗尽",
            code="cube_continue_wait_exhausted",
        )
