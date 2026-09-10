from __future__ import annotations

import asyncio
import sqlite3
import sys
from collections.abc import Sequence
from pathlib import Path
from types import SimpleNamespace

import pytest

from app.agents.scenarios.cqccri_smart_query.subgraph.hydrology_semantic_query.query_child.tools.search_semantic_catalog import (
    RetrievedSemanticContext,
    SemanticCatalogRetriever,
    SemanticContextRetrievalError,
    SentenceTransformerEmbedding,
    VectorIndexNotReadyError,
    merge_retrieved_context,
)

from ..contracts import RetrievalHit, RetrievalTrace, SemanticCatalogMode
from ..query_child import config as config_module
from ..query_child.config import load_hydrology_semantic_query_settings
from ..query_child.models import (
    CatalogContextItem,
    CatalogMember,
    CatalogModel,
    SemanticCatalog,
    SemanticContext,
)
from ..query_child.runtime import safe_response_excerpt


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
        model_top_k=2,
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

    assert await retriever.build_index() == "built_disk"
    assert embedding.document_calls > 1
    assert all(size <= 3 for size in embedding.document_batch_sizes)

    cached_embedding = CountingEmbedding(fail_documents=True)
    cached = _retriever(
        _catalog(), cached_embedding, cache_path=str(cache_path), batch_size=3
    )
    assert await cached.build_index() == "disk_cache"
    assert cached_embedding.document_calls == 0
    assert cached.cache_miss_reason is None


async def test_sentence_transformer_model_is_loaded_once_on_first_embedding(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model_path = tmp_path / "embedding-model"
    model_path.mkdir()
    loaded_paths: list[str] = []

    class FakeSentenceTransformer:
        def __init__(self, path: str) -> None:
            loaded_paths.append(path)

        def encode(self, texts, **kwargs):
            return [[float(len(text))] for text in texts]

    monkeypatch.setitem(
        sys.modules,
        "sentence_transformers",
        SimpleNamespace(SentenceTransformer=FakeSentenceTransformer),
    )
    embedding = SentenceTransformerEmbedding(str(model_path))

    assert loaded_paths == []
    vectors = await asyncio.gather(
        embedding.embed_documents(["监测"]),
        embedding.embed_documents(["设备"]),
        embedding.embed_documents(["告警"]),
    )

    assert loaded_paths == [str(model_path)]
    assert vectors == [[[2.0]], [[2.0]], [[2.0]]]


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

    assert await retriever.build_index() == "built_disk"
    assert retriever.cache_miss_reason == "索引版本不匹配：expected=4 actual=3"
    connection = sqlite3.connect(cache_path)
    version = connection.execute(
        "SELECT value FROM metadata WHERE key = 'version'"
    ).fetchone()[0]
    connection.close()
    assert version == "4"


async def test_force_rebuild_does_not_reuse_valid_cache(tmp_path: Path) -> None:
    cache_path = tmp_path / "semantic-vectors.sqlite3"
    original = CountingEmbedding()
    await _retriever(
        _catalog(), original, cache_path=str(cache_path)
    ).build_index()
    forced = CountingEmbedding()

    assert await _retriever(
        _catalog(), forced, cache_path=str(cache_path)
    ).build_index(force=True) == "built_disk"
    assert forced.document_calls > 0


async def test_query_rejects_missing_index_without_embedding_documents(
    tmp_path: Path,
) -> None:
    embedding = CountingEmbedding(fail_documents=True)
    retriever = _retriever(
        _catalog(),
        embedding,
        cache_path=str(tmp_path / "missing.sqlite3"),
    )

    with pytest.raises(VectorIndexNotReadyError, match="重建脚本"):
        await retriever.retrieve("监测设备")
    assert embedding.document_calls == 0


async def test_query_rejects_stale_index_without_rebuilding(tmp_path: Path) -> None:
    cache_path = tmp_path / "semantic-vectors.sqlite3"
    await _retriever(
        _catalog(1), CountingEmbedding(), cache_path=str(cache_path)
    ).build_index()
    embedding = CountingEmbedding(fail_documents=True)

    with pytest.raises(VectorIndexNotReadyError, match="不一致"):
        await _retriever(
            _catalog(2), embedding, cache_path=str(cache_path)
        ).retrieve("监测设备")
    assert embedding.document_calls == 0


async def test_query_rejects_corrupted_index_without_rebuilding(tmp_path: Path) -> None:
    cache_path = tmp_path / "semantic-vectors.sqlite3"
    cache_path.write_text("invalid", encoding="utf-8")
    embedding = CountingEmbedding(fail_documents=True)

    with pytest.raises(VectorIndexNotReadyError, match="损坏"):
        await _retriever(
            _catalog(), embedding, cache_path=str(cache_path)
        ).retrieve("监测设备")
    assert embedding.document_calls == 0


async def test_query_embedding_failure_falls_back_to_lexical_retrieval(
    tmp_path: Path,
) -> None:
    cache_path = tmp_path / "semantic-vectors.sqlite3"
    await _retriever(
        _catalog(), CountingEmbedding(), cache_path=str(cache_path)
    ).build_index()
    embedding = CountingEmbedding(fail_query=True)
    retrieved = await _retriever(
        _catalog(), embedding, cache_path=str(cache_path)
    ).retrieve("监测设备")

    assert len([hit for hit in retrieved.trace.hits if hit.item_type == "model"]) == 2
    assert len([hit for hit in retrieved.trace.hits if hit.item_type != "model"]) == 2
    assert retrieved.context.strategy == SemanticCatalogMode.VECTOR
    assert any("词法目录检索" in warning for warning in retrieved.warnings)


async def test_missing_embedding_falls_back_to_lexical_retrieval() -> None:
    retrieved = await _retriever(_catalog(), None).retrieve("监测设备")

    assert len([hit for hit in retrieved.trace.hits if hit.item_type == "model"]) == 2
    assert len([hit for hit in retrieved.trace.hits if hit.item_type != "model"]) == 2
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
    current = _retrieved("原问题", "model_0.value", round_number=1)
    refreshed = _retrieved("原问题 + error", "model_1.value", round_number=2)
    current.context.items.append(CatalogContextItem(
        item_type="join_component",
        name="1",
        payload={"models": [{"name": "model_0"}]},
    ))
    refreshed.context.items.append(CatalogContextItem(
        item_type="join_component",
        name="1",
        payload={"models": [{"name": "model_0"}, {"name": "model_1"}]},
    ))
    merged = merge_retrieved_context(current, refreshed)

    assert merged.context.retrieval_round == 2
    assert [item.name for item in merged.context.items] == [
        "model_0.value",
        "1",
        "model_1.value",
    ]
    component = next(
        item for item in merged.context.items if item.item_type == "join_component"
    )
    assert [model["name"] for model in component.payload["models"]] == [
        "model_0",
        "model_1",
    ]
    assert merged.trace.queries == ["原问题", "原问题 + error"]


@pytest.mark.parametrize(
    "name",
    [
        "CONTEXT_TOP_K",
        "MODEL_TOP_K",
        "EMBEDDING_BATCH_SIZE",
        "EMBEDDING_CONCURRENCY",
        "AUTO_FULL_CONTEXT_MAX_CHARS",
    ],
)
def test_positive_catalog_settings(name: str, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(f"HYDROLOGY_SEMANTIC_QUERY_{name}", "0")
    with pytest.raises(ValueError, match=name):
        load_hydrology_semantic_query_settings()


def test_new_catalog_defaults(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setattr(config_module, "_ENV_FILE", tmp_path / "missing.env")
    names = (
        "CATALOG_STRATEGY",
        "MODEL_TOP_K",
        "CONTEXT_TOP_K",
        "AUTO_FULL_CONTEXT_MAX_CHARS",
        "MAX_AGENT_ITERATIONS",
        "MAX_QUERY_ROUNDS",
    )
    for name in names:
        monkeypatch.delenv(f"HYDROLOGY_SEMANTIC_QUERY_{name}", raising=False)

    settings = load_hydrology_semantic_query_settings()
    assert settings.catalog_mode == SemanticCatalogMode.AUTO
    assert settings.model_top_k == 5
    assert settings.context_top_k == 20
    assert settings.auto_full_context_max_chars == 30000
    assert settings.max_agent_iterations == 6
    assert settings.max_query_rounds == 5
    assert settings.vector_index_path == str(
        Path(config_module.__file__).resolve().parent.parent
        / "semantic"
        / "cache"
        / "semantic-catalog-vectors.sqlite3"
    )


@pytest.mark.parametrize("value", ["2", "13"])
def test_agent_iteration_setting_is_bounded(
    value: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("HYDROLOGY_SEMANTIC_QUERY_MAX_AGENT_ITERATIONS", value)
    with pytest.raises(ValueError, match="MAX_AGENT_ITERATIONS"):
        load_hydrology_semantic_query_settings()


@pytest.mark.parametrize("value", ["0", "6"])
def test_query_round_setting_is_bounded(
    value: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("HYDROLOGY_SEMANTIC_QUERY_MAX_QUERY_ROUNDS", value)
    with pytest.raises(ValueError, match="MAX_QUERY_ROUNDS"):
        load_hydrology_semantic_query_settings()


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
    excerpt = safe_response_excerpt(
        'authorization: Bearer secret token="private" api_key=abc123 cookie=session'
    )
    assert "secret" not in excerpt
    assert "private" not in excerpt
    assert "abc123" not in excerpt
    assert "session" not in excerpt
