from __future__ import annotations

import asyncio
import re
import time
from typing import Any

import httpx


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

    async def aclose(self) -> None:
        return None

    def _safe_message(self, text: str) -> str:
        safe = text.replace(self.base_url, "<cube>")
        if self.token:
            safe = safe.replace(self.token, "<redacted>")
        safe = re.sub(r'(?i)(authorization["\s:=]+)[^,}\s]+', r"\1<redacted>", safe)
        safe = re.sub(r"https?://[^\s,}]+", "<cube>", safe, flags=re.IGNORECASE)
        return safe[:1000]

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
            if self._client is not None:
                payload = await self._request(self._client, "GET", "/meta")
            else:
                async with httpx.AsyncClient(timeout=self.timeout_seconds) as client:
                    payload = await self._request(client, "GET", "/meta")
            self._meta = payload
            self._meta_expires_at = now + self.meta_cache_ttl_seconds
            return payload

    async def load(self, query: dict[str, Any]) -> dict[str, Any]:
        if self._client is not None:
            return await self._query(self._client, "/load", query)
        async with httpx.AsyncClient(timeout=self.timeout_seconds) as client:
            return await self._query(client, "/load", query)

    async def get_sql(self, query: dict[str, Any]) -> tuple[str, list[Any]]:
        if self._client is not None:
            payload = await self._query(self._client, "/sql", query)
        else:
            async with httpx.AsyncClient(timeout=self.timeout_seconds) as client:
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
