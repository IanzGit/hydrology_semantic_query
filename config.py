from __future__ import annotations

import os
from dataclasses import dataclass
from math import isfinite
from pathlib import Path
from urllib.parse import urlsplit
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from dotenv.main import dotenv_values

from .models import SemanticCatalogMode

_ENV_FILE = Path(__file__).with_name(".env")
_DEFAULT_EMBEDDING_MODEL = "/home/ubuntu/code_ws/D20260724-Agent开发/models/bge-large-zh-v1.5"


def _value(values: dict[str, str | None], name: str, default: str) -> str:
    value = os.getenv(name)
    if value is not None:
        return value
    value = values.get(name)
    return value if value is not None else default


def _boolean(values: dict[str, str | None], name: str, default: bool) -> bool:
    value = _value(values, name, str(default))
    normalized = value.strip().lower()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    raise ValueError(f"环境变量 {name} 必须是布尔值")


@dataclass(frozen=True, slots=True)
class HydrologySemanticQuerySettings:
    cube_url: str
    cube_token: str | None
    timeout_seconds: float
    continue_wait_retries: int
    meta_cache_ttl_seconds: float
    max_retries: int
    max_rows: int
    hard_max_rows: int
    timezone: str
    enable_report: bool
    catalog_mode: SemanticCatalogMode = SemanticCatalogMode.AUTO
    embedding_model: str | None = None
    view_top_k: int = 3
    cube_top_k: int = 5
    member_top_k: int = 15
    vector_index_path: str | None = "cache/semantic-catalog-vectors.sqlite3"
    retry_on_empty_result: bool = True
    embedding_batch_size: int = 32
    embedding_concurrency: int = 3
    retrieval_concurrency: int = 3
    context_member_limit: int = 12
    catalog_batch_size: int = 4
    max_cube_models: int = 4
    member_match_threshold: float = 0.55
    auto_full_context_max_chars: int = 18000


def normalize_cube_url(value: str) -> str:
    cube_url = value.strip().rstrip("/")
    if not cube_url:
        raise ValueError("Cube URL 不能为空")
    parsed = urlsplit(cube_url)
    try:
        hostname = parsed.hostname
        port = parsed.port
    except ValueError as exc:
        raise ValueError("Cube URL 必须是有效的 HTTP/HTTPS 地址") from exc
    if (
        parsed.scheme not in {"http", "https"}
        or not parsed.netloc
        or not hostname
        or port is not None and not 0 < port < 65536
        or any(character.isspace() for character in cube_url)
    ):
        raise ValueError("Cube URL 必须是有效的 HTTP/HTTPS 地址")
    if parsed.query or parsed.fragment:
        raise ValueError("Cube URL 不能包含查询参数或片段")
    if cube_url.endswith("/v1"):
        return cube_url
    if cube_url.endswith("/cubejs-api"):
        return f"{cube_url}/v1"
    return f"{cube_url}/cubejs-api/v1"


def load_hydrology_semantic_query_settings() -> HydrologySemanticQuerySettings:
    prefix = "HYDROLOGY_SEMANTIC_QUERY_"
    values = dict(dotenv_values(_ENV_FILE))
    cube_url = normalize_cube_url(
        _value(values, f"{prefix}CUBE_URL", "http://127.0.0.1:4000")
    )
    mode_value = _value(values, f"{prefix}CATALOG_STRATEGY", "auto").strip().lower()
    try:
        catalog_mode = SemanticCatalogMode(mode_value)
    except ValueError as exc:
        raise ValueError(
            f"环境变量 {prefix}CATALOG_STRATEGY 只能是 full、vector 或 auto"
        ) from exc
    settings = HydrologySemanticQuerySettings(
        cube_url=cube_url,
        cube_token=_value(values, f"{prefix}CUBE_TOKEN", "").strip() or None,
        timeout_seconds=float(_value(values, f"{prefix}TIMEOUT_SECONDS", "30")),
        continue_wait_retries=int(
            _value(values, f"{prefix}CONTINUE_WAIT_RETRIES", "6")
        ),
        meta_cache_ttl_seconds=float(
            _value(values, f"{prefix}META_CACHE_TTL_SECONDS", "300")
        ),
        max_retries=int(_value(values, f"{prefix}MAX_RETRIES", "1")),
        max_rows=int(_value(values, f"{prefix}MAX_ROWS", "50")),
        hard_max_rows=int(_value(values, f"{prefix}HARD_MAX_ROWS", "1000")),
        timezone=_value(values, f"{prefix}TIMEZONE", "Asia/Shanghai").strip(),
        enable_report=_boolean(values, f"{prefix}ENABLE_REPORT", True),
        catalog_mode=catalog_mode,
        embedding_model=(
            _value(values, f"{prefix}EMBEDDING_MODEL", _DEFAULT_EMBEDDING_MODEL).strip()
            or None
        ),
        view_top_k=int(_value(values, f"{prefix}VIEW_TOP_K", "3")),
        cube_top_k=int(_value(values, f"{prefix}CUBE_TOP_K", "5")),
        member_top_k=int(_value(values, f"{prefix}MEMBER_TOP_K", "15")),
        vector_index_path=(
            _value(
                values,
                f"{prefix}VECTOR_INDEX_PATH",
                "cache/semantic-catalog-vectors.sqlite3",
            ).strip()
            or None
        ),
        retry_on_empty_result=_boolean(
            values, f"{prefix}RETRY_ON_EMPTY_RESULT", True
        ),
        embedding_batch_size=int(
            _value(values, f"{prefix}EMBEDDING_BATCH_SIZE", "32")
        ),
        embedding_concurrency=int(
            _value(values, f"{prefix}EMBEDDING_CONCURRENCY", "3")
        ),
        retrieval_concurrency=int(
            _value(values, f"{prefix}RETRIEVAL_CONCURRENCY", "3")
        ),
        context_member_limit=int(
            _value(values, f"{prefix}CONTEXT_MEMBER_LIMIT", "12")
        ),
        catalog_batch_size=int(
            _value(values, f"{prefix}CATALOG_BATCH_SIZE", "4")
        ),
        max_cube_models=int(_value(values, f"{prefix}MAX_CUBE_MODELS", "4")),
        member_match_threshold=float(
            _value(values, f"{prefix}MEMBER_MATCH_THRESHOLD", "0.55")
        ),
        auto_full_context_max_chars=int(
            _value(values, f"{prefix}AUTO_FULL_CONTEXT_MAX_CHARS", "18000")
        ),
    )
    if not isfinite(settings.timeout_seconds) or settings.timeout_seconds <= 0:
        raise ValueError(f"环境变量 {prefix}TIMEOUT_SECONDS 必须大于 0")
    if settings.continue_wait_retries < 0 or settings.max_retries < 0:
        raise ValueError("重试次数不能为负数")
    if settings.max_retries > 1:
        raise ValueError(f"环境变量 {prefix}MAX_RETRIES 不能超过 1")
    if not isfinite(settings.meta_cache_ttl_seconds) or settings.meta_cache_ttl_seconds < 0:
        raise ValueError(f"环境变量 {prefix}META_CACHE_TTL_SECONDS 不能为负数")
    if settings.max_rows < 1 or settings.hard_max_rows < 1:
        raise ValueError("结果行数上限必须大于 0")
    if settings.max_rows > settings.hard_max_rows:
        raise ValueError(f"环境变量 {prefix}MAX_ROWS 不能超过 {prefix}HARD_MAX_ROWS")
    for name, value in (
        ("VIEW_TOP_K", settings.view_top_k),
        ("CUBE_TOP_K", settings.cube_top_k),
        ("MEMBER_TOP_K", settings.member_top_k),
        ("EMBEDDING_BATCH_SIZE", settings.embedding_batch_size),
        ("EMBEDDING_CONCURRENCY", settings.embedding_concurrency),
        ("RETRIEVAL_CONCURRENCY", settings.retrieval_concurrency),
        ("CONTEXT_MEMBER_LIMIT", settings.context_member_limit),
        ("CATALOG_BATCH_SIZE", settings.catalog_batch_size),
        ("MAX_CUBE_MODELS", settings.max_cube_models),
    ):
        if value < 1:
            raise ValueError(f"环境变量 {prefix}{name} 必须大于 0")
    if settings.context_member_limit > 12:
        raise ValueError(f"环境变量 {prefix}CONTEXT_MEMBER_LIMIT 不能超过 12")
    if settings.max_cube_models > 4:
        raise ValueError(f"环境变量 {prefix}MAX_CUBE_MODELS 不能超过 4")
    if (
        not isfinite(settings.member_match_threshold)
        or not 0 <= settings.member_match_threshold <= 1
    ):
        raise ValueError(f"环境变量 {prefix}MEMBER_MATCH_THRESHOLD 必须在 0 到 1 之间")
    if settings.auto_full_context_max_chars < 1:
        raise ValueError(
            f"环境变量 {prefix}AUTO_FULL_CONTEXT_MAX_CHARS 必须大于 0"
        )
    try:
        ZoneInfo(settings.timezone)
    except (ZoneInfoNotFoundError, ValueError) as exc:
        raise ValueError(f"环境变量 {prefix}TIMEZONE 必须是有效的 IANA 时区") from exc
    return settings
