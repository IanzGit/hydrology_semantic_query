from __future__ import annotations

import sqlite3
from collections.abc import Sequence
from pathlib import Path

import pytest

from ..config import load_hydrology_semantic_query_settings
from ..models import (
    CatalogContextItem,
    CatalogMember,
    CatalogModel,
    RetrievalHit,
    RetrievalTrace,
    SemanticCatalog,
    SemanticCatalogMode,
    SemanticContext,
)
from ..nodes import _safe_response_excerpt
from ..semantic_catalog_retriever import (
    RetrievedSemanticContext,
    SemanticCatalogRetriever,
    SemanticContextRetrievalError,
    merge_retrieved_context,
)


class CountingEmbedding:
    model_path = "counting-test"

    def __init__(
        self,
        *,
        fail_documents: bool = False,
        fail_query: bool = False,
    ) -> None:
        self.fail_documents = fail_documents
        self.fail_query = fail_query
        self.document_calls = 0
        self.document_batch_sizes: list[int] = []

    async def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
        self.document_calls += 1
        self.document_batch_sizes.append(len(texts))
        if self.fail_documents:
            raise RuntimeError("document embedding unavailable")
        return [self._vector(text) for text in texts]

    async def embed_query(self, text: str) -> list[float]:
        if self.fail_query:
            raise RuntimeError("query embedding unavailable")
        return self._vector(text)

    @staticmethod
    def _vector(text: str) -> list[float]:
        return [float(text.count("监测")), float(text.count("设备")), 0.1]


def _catalog(count: int = 4) -> SemanticCatalog:
    models: dict[str, CatalogModel] = {}
    for index in range(count):
        name = f"model_{index}"
        member_name = f"{name}.value"
        models[name] = CatalogModel(
            name=name,
            model_type="cube",
            title=f"监测设备 {index}",
            connected_component=1,
            business_domain="hydrology",
            members={
                member_name: CatalogMember(
                    name=member_name,
                    title=f"监测值 {index}",
                    member_type="dimension",
                    data_type="number",
                )
            },
        )
    return SemanticCatalog(models=models)


def _retriever(
    catalog: SemanticCatalog,
    embedding: CountingEmbedding | None,
    *,
    cache_path: str | None = None,
    batch_size: int = 3,
) -> SemanticCatalogRetriever:
    return SemanticCatalogRetriever(
        catalog,
        context_top_k=2,
        vector_index_path=cache_path,
        embedding_client=embedding,
        mode=SemanticCatalogMode.VECTOR,
        embedding_batch_size=batch_size,
        embedding_concurrency=2,
        auto_full_context_max_chars=1,
    )


async def test_sqlite_vector_cache_is_batched_and_reused(tmp_path: Path) -> None:
    cache_path = tmp_path / "semantic-vectors.sqlite3"
    embedding = CountingEmbedding()
    retriever = _retriever(
        _catalog(), embedding, cache_path=str(cache_path), batch_size=3
    )

    assert await retriever.prepare() == "built_disk"
    assert embedding.document_calls > 1
    assert all(size <= 3 for size in embedding.document_batch_sizes)

    cached_embedding = CountingEmbedding(fail_documents=True)
    cached = _retriever(
        _catalog(), cached_embedding, cache_path=str(cache_path), batch_size=3
    )
    assert await cached.prepare() == "disk_cache"
    assert cached_embedding.document_calls == 0


async def test_old_index_version_is_rebuilt_as_v4(tmp_path: Path) -> None:
    cache_path = tmp_path / "semantic-vectors.sqlite3"
    connection = sqlite3.connect(cache_path)
    connection.execute("CREATE TABLE metadata (key TEXT PRIMARY KEY, value TEXT NOT NULL)")
    connection.execute("INSERT INTO metadata VALUES ('version', '3')")
    connection.commit()
    connection.close()
    retriever = _retriever(
        _catalog(1), CountingEmbedding(), cache_path=str(cache_path)
    )

    assert await retriever.prepare() == "built_disk"
    connection = sqlite3.connect(cache_path)
    version = connection.execute(
        "SELECT value FROM metadata WHERE key = 'version'"
    ).fetchone()[0]
    connection.close()
    assert version == "4"


@pytest.mark.parametrize(
    "embedding",
    [
        CountingEmbedding(fail_documents=True),
        CountingEmbedding(fail_query=True),
        None,
    ],
)
async def test_vector_unavailable_falls_back_to_bounded_lexical_retrieval(
    embedding: CountingEmbedding | None,
) -> None:
    retrieved = await _retriever(_catalog(), embedding).retrieve("监测设备")

    assert len(retrieved.trace.hits) == 2
    assert retrieved.context.strategy == SemanticCatalogMode.VECTOR
    assert any("词法目录检索" in warning for warning in retrieved.warnings)


async def test_invalid_metadata_filter_is_rejected() -> None:
    with pytest.raises(SemanticContextRetrievalError):
        await _retriever(_catalog(), None).retrieve(
            "监测设备",
            metadata_filters={"private": True},
        )


def _retrieved(
    query: str,
    item_name: str,
    *,
    round_number: int,
) -> RetrievedSemanticContext:
    item = CatalogContextItem(
        item_type="member",
        name=item_name,
        model_name=item_name.partition(".")[0],
        score=0.8,
        payload={"name": item_name},
    )
    return RetrievedSemanticContext(
        mode=SemanticCatalogMode.VECTOR,
        catalog=_catalog(),
        context=SemanticContext(
            strategy=SemanticCatalogMode.VECTOR,
            items=[item],
            retrieval_round=round_number,
        ),
        trace=RetrievalTrace(
            strategy=SemanticCatalogMode.VECTOR,
            queries=[query],
            hits=[RetrievalHit(
                item_type="member",
                name=item_name,
                model_name=item.model_name,
                score=0.8,
            )],
            index_source="memory",
        ),
    )


def test_context_refresh_merges_items_queries_and_round() -> None:
    merged = merge_retrieved_context(
        _retrieved("原问题", "model_0.value", round_number=1),
        _retrieved("原问题 + error", "model_1.value", round_number=2),
    )

    assert merged.context.retrieval_round == 2
    assert [item.name for item in merged.context.items] == [
        "model_0.value",
        "model_1.value",
    ]
    assert merged.trace.queries == ["原问题", "原问题 + error"]


@pytest.mark.parametrize(
    "name",
    [
        "CONTEXT_TOP_K",
        "EMBEDDING_BATCH_SIZE",
        "EMBEDDING_CONCURRENCY",
        "AUTO_FULL_CONTEXT_MAX_CHARS",
    ],
)
def test_positive_catalog_settings(name: str, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(f"HYDROLOGY_SEMANTIC_QUERY_{name}", "0")
    with pytest.raises(ValueError, match=name):
        load_hydrology_semantic_query_settings()


def test_new_catalog_defaults(monkeypatch: pytest.MonkeyPatch) -> None:
    names = (
        "CATALOG_STRATEGY",
        "CONTEXT_TOP_K",
        "AUTO_FULL_CONTEXT_MAX_CHARS",
    )
    for name in names:
        monkeypatch.delenv(f"HYDROLOGY_SEMANTIC_QUERY_{name}", raising=False)

    settings = load_hydrology_semantic_query_settings()
    assert settings.catalog_mode == SemanticCatalogMode.AUTO
    assert settings.context_top_k == 20
    assert settings.auto_full_context_max_chars == 30000


def test_removed_catalog_settings_are_not_public_fields() -> None:
    fields = set(load_hydrology_semantic_query_settings().__dataclass_fields__)
    assert fields.isdisjoint({
        "view_top_k",
        "cube_top_k",
        "member_top_k",
        "retry_on_empty_result",
        "retrieval_concurrency",
        "context_member_limit",
        "catalog_batch_size",
        "max_cube_models",
        "member_match_threshold",
    })


def test_generation_response_excerpt_redacts_credentials() -> None:
    excerpt = _safe_response_excerpt(
        'authorization: Bearer secret token="private" api_key=abc123 cookie=session'
    )
    assert "secret" not in excerpt
    assert "private" not in excerpt
    assert "abc123" not in excerpt
    assert "session" not in excerpt
