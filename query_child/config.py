from __future__ import annotations

import os
from dataclasses import dataclass
from math import isfinite
from pathlib import Path
from urllib.parse import urlsplit
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from dotenv.main import dotenv_values

from ..contracts import SemanticCatalogMode

HYDROLOGY_SEMANTIC_QUERY_ID = "hydrology_semantic_query"
_SCENARIO_DIR = Path(__file__).resolve().parent.parent
_ENV_FILE = _SCENARIO_DIR / "env" / "agent.env"
_DEFAULT_EMBEDDING_MODEL = "/home/ubuntu/code_ws/model/bge-large-zh-v1.5"
_DEFAULT_VECTOR_INDEX_PATH = str(
    _SCENARIO_DIR
    / "semantic"
    / "cache"
    / "semantic-catalog-vectors.sqlite3"
)


def _value(values: dict[str, str | None], name: str, default: str) -> str:
    value = os.getenv(name)
    if value is not None:
        return value
    value = values.get(name)
    return value if value is not None else default


@dataclass(frozen=True, slots=True)
class HydrologySemanticQuerySettings:
    cube_url: str
    cube_token: str | None
    timeout_seconds: float
    continue_wait_retries: int
    meta_cache_ttl_seconds: float
    timezone: str
    catalog_mode: SemanticCatalogMode = SemanticCatalogMode.AUTO
    embedding_model: str | None = None
    model_top_k: int = 5
    context_top_k: int = 20
    vector_index_path: str | None = _DEFAULT_VECTOR_INDEX_PATH
    embedding_batch_size: int = 32
    embedding_concurrency: int = 3
    auto_full_context_max_chars: int = 30000
    max_agent_iterations: int = 6
    max_query_rounds: int = 5


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
        or (port is not None and not 0 < port < 65536)
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
        timezone=_value(values, f"{prefix}TIMEZONE", "Asia/Shanghai").strip(),
        catalog_mode=catalog_mode,
        embedding_model=(
            _value(values, f"{prefix}EMBEDDING_MODEL", _DEFAULT_EMBEDDING_MODEL).strip()
            or None
        ),
        model_top_k=int(_value(values, f"{prefix}MODEL_TOP_K", "5")),
        context_top_k=int(_value(values, f"{prefix}CONTEXT_TOP_K", "20")),
        vector_index_path=(
            _value(
                values,
                f"{prefix}VECTOR_INDEX_PATH",
                _DEFAULT_VECTOR_INDEX_PATH,
            ).strip()
            or None
        ),
        embedding_batch_size=int(
            _value(values, f"{prefix}EMBEDDING_BATCH_SIZE", "32")
        ),
        embedding_concurrency=int(
            _value(values, f"{prefix}EMBEDDING_CONCURRENCY", "3")
        ),
        auto_full_context_max_chars=int(
            _value(values, f"{prefix}AUTO_FULL_CONTEXT_MAX_CHARS", "30000")
        ),
        max_agent_iterations=int(
            _value(values, f"{prefix}MAX_AGENT_ITERATIONS", "6")
        ),
        max_query_rounds=int(
            _value(values, f"{prefix}MAX_QUERY_ROUNDS", "5")
        ),
    )
    if not isfinite(settings.timeout_seconds) or settings.timeout_seconds <= 0:
        raise ValueError(f"环境变量 {prefix}TIMEOUT_SECONDS 必须大于 0")
    if settings.continue_wait_retries < 0:
        raise ValueError("重试次数不能为负数")
    if not isfinite(settings.meta_cache_ttl_seconds) or settings.meta_cache_ttl_seconds < 0:
        raise ValueError(f"环境变量 {prefix}META_CACHE_TTL_SECONDS 不能为负数")
    for name, value in (
        ("MODEL_TOP_K", settings.model_top_k),
        ("CONTEXT_TOP_K", settings.context_top_k),
        ("EMBEDDING_BATCH_SIZE", settings.embedding_batch_size),
        ("EMBEDDING_CONCURRENCY", settings.embedding_concurrency),
    ):
        if value < 1:
            raise ValueError(f"环境变量 {prefix}{name} 必须大于 0")
    if settings.auto_full_context_max_chars < 1:
        raise ValueError(
            f"环境变量 {prefix}AUTO_FULL_CONTEXT_MAX_CHARS 必须大于 0"
        )
    if not 3 <= settings.max_agent_iterations <= 12:
        raise ValueError(
            f"环境变量 {prefix}MAX_AGENT_ITERATIONS 必须在 3 到 12 之间"
        )
    if not 1 <= settings.max_query_rounds <= 5:
        raise ValueError(
            f"环境变量 {prefix}MAX_QUERY_ROUNDS 必须在 1 到 5 之间"
        )
    try:
        ZoneInfo(settings.timezone)
    except (ZoneInfoNotFoundError, ValueError) as exc:
        raise ValueError(f"环境变量 {prefix}TIMEZONE 必须是有效的 IANA 时区") from exc
    return settings
