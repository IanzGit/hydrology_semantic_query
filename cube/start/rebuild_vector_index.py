from __future__ import annotations

import argparse
import asyncio
import sys
from pathlib import Path
from time import perf_counter

from app.agents.scenarios.hydrology_semantic_query.client import (
    CubeClient,
    catalog_from_meta,
)
from app.agents.scenarios.hydrology_semantic_query.config import (
    HydrologySemanticQuerySettings,
    load_hydrology_semantic_query_settings,
)
from app.agents.scenarios.hydrology_semantic_query.tools.search_semantic_catalog import (
    SemanticCatalogRetriever,
    SentenceTransformerEmbedding,
)

PROJECT_ROOT = Path(__file__).resolve().parents[6]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="准备水文语义目录向量索引")
    parser.add_argument("--force", action="store_true")
    return parser


def resolve_index_path(value: str | None) -> Path:
    if not value:
        raise ValueError("HYDROLOGY_SEMANTIC_QUERY_VECTOR_INDEX_PATH 不能为空")
    path = Path(value).expanduser()
    return path if path.is_absolute() else (PROJECT_ROOT / path).resolve()


async def prepare_vector_index(
    settings: HydrologySemanticQuerySettings,
    *,
    force: bool,
) -> tuple[str, int, Path]:
    if not settings.embedding_model:
        raise ValueError("HYDROLOGY_SEMANTIC_QUERY_EMBEDDING_MODEL 不能为空")
    index_path = resolve_index_path(settings.vector_index_path)
    client = CubeClient(
        base_url=settings.cube_url,
        token=settings.cube_token,
        timeout_seconds=settings.timeout_seconds,
        continue_wait_retries=settings.continue_wait_retries,
        meta_cache_ttl_seconds=settings.meta_cache_ttl_seconds,
    )
    meta_started = perf_counter()
    catalog = catalog_from_meta(await client.get_meta(force=True))
    print(f"语义目录加载完成：elapsed={perf_counter() - meta_started:.2f}s", flush=True)
    embedding = SentenceTransformerEmbedding(settings.embedding_model)
    retriever = SemanticCatalogRetriever(
        catalog,
        model_top_k=settings.model_top_k,
        context_top_k=settings.context_top_k,
        vector_index_path=str(index_path),
        embedding_client=embedding,
        mode=settings.catalog_mode,
        embedding_batch_size=settings.embedding_batch_size,
        embedding_concurrency=settings.embedding_concurrency,
        auto_full_context_max_chars=settings.auto_full_context_max_chars,
    )
    index_started = perf_counter()
    source = await retriever.build_index(force=force)
    print(
        f"向量索引准备完成：source={source} "
        f"elapsed={perf_counter() - index_started:.2f}s",
        flush=True,
    )
    return source, retriever.document_count, index_path


def main(argv: list[str] | None = None) -> int:
    arguments = build_parser().parse_args(argv)
    started = perf_counter()
    try:
        source, document_count, index_path = asyncio.run(
            prepare_vector_index(
                load_hydrology_semantic_query_settings(),
                force=arguments.force,
            )
        )
    except Exception as exc:
        print(f"语义目录向量索引准备失败：{str(exc)[:1000]}", file=sys.stderr)
        return 1
    action = "复用" if source in {"memory", "disk_cache"} else "重建"
    print(
        f"语义目录向量索引已{action}：documents={document_count} "
        f"source={source} path={index_path} elapsed={perf_counter() - started:.2f}s"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
