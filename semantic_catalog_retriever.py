from __future__ import annotations

import asyncio
import hashlib
import json
import math
import os
import re
import sqlite3
import threading
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

from .models import (
    CatalogContextItem,
    CatalogMember,
    CatalogModel,
    RetrievalHit,
    RetrievalTrace,
    SemanticCatalog,
    SemanticCatalogMode,
    SemanticContext,
)

try:
    import fcntl
except ImportError:
    fcntl = None


@runtime_checkable
class EmbeddingClient(Protocol):
    async def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
        ...

    async def embed_query(self, text: str) -> list[float]:
        ...


class SentenceTransformerEmbedding:
    def __init__(self, model_path: str) -> None:
        path = Path(model_path).expanduser()
        if not path.is_dir():
            raise ValueError(f"本地嵌入模型目录不存在：{path}")
        from sentence_transformers import SentenceTransformer

        self.model_path = str(path)
        self._model = SentenceTransformer(self.model_path)

    def _embed_documents_blocking(self, texts: Sequence[str]) -> list[list[float]]:
        if not texts:
            return []
        vectors = self._model.encode(
            list(texts),
            convert_to_numpy=True,
            show_progress_bar=False,
        )
        return [[float(value) for value in vector] for vector in vectors]

    async def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
        return await asyncio.to_thread(self._embed_documents_blocking, texts)

    async def embed_query(self, text: str) -> list[float]:
        return (await self.embed_documents([text]))[0]


class SemanticContextRetrievalError(ValueError):
    pass


@dataclass
class _CatalogDocument:
    doc_id: str
    item_type: str
    name: str
    model_name: str | None
    content: str
    payload: dict[str, Any]
    related_models: tuple[str, ...] = ()
    vector: list[float] | None = None


@dataclass(frozen=True)
class _ScoredDocument:
    score: float
    document: _CatalogDocument


@dataclass
class RetrievedSemanticContext:
    mode: SemanticCatalogMode
    catalog: SemanticCatalog
    context: SemanticContext
    trace: RetrievalTrace
    warnings: list[str] = field(default_factory=list)


def _member_payload(member: CatalogMember) -> dict[str, Any]:
    return {
        "name": member.name,
        "title": member.title,
        "kind": member.member_type,
        "type": member.data_type,
        "description": member.description,
        "ai_context": member.ai_context,
        "aliases": member.aliases,
        "folder": member.folder,
        "hierarchy": member.hierarchy,
        "granularities": member.granularities,
        "projection_role": member.projection_role,
    }


def _model_payload(model: CatalogModel) -> dict[str, Any]:
    members = list(model.members.values())
    return {
        "name": model.name,
        "type": model.model_type,
        "title": model.title,
        "description": model.description,
        "ai_context": model.ai_context,
        "aliases": model.aliases,
        "use_cases": model.use_cases,
        "business_domain": model.business_domain,
        "connected_component": model.connected_component,
        "default_projection": model.default_projection,
        "members": [_member_payload(member) for member in members],
    }


def _component_payload(
    component: str | int,
    models: Sequence[CatalogModel],
) -> dict[str, Any]:
    return {
        "component": component,
        "models": [
            {
                "name": model.name,
                "title": model.title,
                "description": model.description,
            }
            for model in models
        ],
    }


def _content(payload: dict[str, Any]) -> str:
    return json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def _documents(catalog: SemanticCatalog) -> list[_CatalogDocument]:
    documents: list[_CatalogDocument] = []
    components: dict[str | int, list[CatalogModel]] = {}
    for model in catalog.models.values():
        model_payload = _model_payload(model)
        documents.append(_CatalogDocument(
            doc_id=f"model:{model.name}",
            item_type="model",
            name=model.name,
            model_name=model.name,
            content=_content(model_payload),
            payload=model_payload,
            related_models=(model.name,),
        ))
        for member in model.members.values():
            member_payload = {
                "model": {
                    "name": model.name,
                    "type": model.model_type,
                    "title": model.title,
                    "description": model.description,
                },
                "member": _member_payload(member),
            }
            documents.append(_CatalogDocument(
                doc_id=f"member:{member.name}",
                item_type="member",
                name=member.name,
                model_name=model.name,
                content=_content(member_payload),
                payload=member_payload,
                related_models=(model.name,),
            ))
        if model.model_type == "view":
            for folder in model.folders:
                folder_members = [
                    member
                    for member in model.members.values()
                    if member.folder == folder
                ]
                if not folder_members:
                    continue
                folder_payload = {
                    "model": model.name,
                    "model_title": model.title,
                    "folder": folder,
                    "members": [
                        _member_payload(member) for member in folder_members
                    ],
                }
                documents.append(_CatalogDocument(
                    doc_id=f"view_folder:{model.name}:{folder}",
                    item_type="view_folder",
                    name=f"{model.name}:{folder}",
                    model_name=model.name,
                    content=_content(folder_payload),
                    payload=folder_payload,
                    related_models=(model.name,),
                ))
        if model.model_type == "cube" and model.connected_component is not None:
            components.setdefault(model.connected_component, []).append(model)
    for component, models in components.items():
        ordered = sorted(models, key=lambda model: model.name)
        component_payload = _component_payload(component, ordered)
        documents.append(_CatalogDocument(
            doc_id=f"join_component:{component}",
            item_type="join_component",
            name=str(component),
            model_name=None,
            content=_content(component_payload),
            payload=component_payload,
            related_models=tuple(model.name for model in ordered),
        ))
    return documents


def _catalog_signature(documents: Sequence[_CatalogDocument]) -> str:
    payload = [
        {
            "id": document.doc_id,
            "content": document.content,
            "item_type": document.item_type,
            "name": document.name,
            "model_name": document.model_name,
            "related_models": document.related_models,
        }
        for document in documents
    ]
    return hashlib.sha256(_content({"documents": payload}).encode()).hexdigest()


def _cosine(left: Sequence[float], right: Sequence[float]) -> float:
    if len(left) != len(right):
        raise ValueError("嵌入向量维度不一致")
    dot = sum(a * b for a, b in zip(left, right, strict=True))
    left_norm = math.sqrt(sum(value * value for value in left))
    right_norm = math.sqrt(sum(value * value for value in right))
    if left_norm == 0 or right_norm == 0:
        return 0.0
    return dot / (left_norm * right_norm)


def _normalized(value: str) -> str:
    return re.sub(r"[^0-9a-zA-Z\u4e00-\u9fff]+", "", value).lower()


def _lexical_score(left: str, right: str) -> float:
    left_value = _normalized(left)
    right_value = _normalized(right)
    if not left_value or not right_value:
        return 0.0
    if left_value == right_value:
        return 1.0
    if left_value in right_value or right_value in left_value:
        return min(len(left_value), len(right_value)) / max(
            len(left_value), len(right_value)
        )
    left_pairs = {
        left_value[index : index + 2]
        for index in range(max(1, len(left_value) - 1))
    }
    right_pairs = {
        right_value[index : index + 2]
        for index in range(max(1, len(right_value) - 1))
    }
    overlap = len(left_pairs & right_pairs)
    return 2 * overlap / (len(left_pairs) + len(right_pairs))


_PATH_LOCKS: dict[str, threading.Lock] = {}
_PATH_LOCKS_GUARD = threading.Lock()
_ASYNC_PATH_LOCKS: dict[tuple[int, str], asyncio.Lock] = {}


def _async_index_lock(path: Path) -> asyncio.Lock:
    key = (id(asyncio.get_running_loop()), str(path.resolve()))
    with _PATH_LOCKS_GUARD:
        return _ASYNC_PATH_LOCKS.setdefault(key, asyncio.Lock())


class _IndexLock:
    def __init__(self, path: Path) -> None:
        key = str(path.resolve())
        with _PATH_LOCKS_GUARD:
            self._thread_lock = _PATH_LOCKS.setdefault(key, threading.Lock())
        self._lock_path = path.with_name(f"{path.name}.lock")
        self._handle: Any = None

    def acquire(self) -> None:
        self._thread_lock.acquire()
        try:
            self._lock_path.parent.mkdir(parents=True, exist_ok=True)
            self._handle = self._lock_path.open("a+b")
            if fcntl is not None:
                fcntl.flock(self._handle.fileno(), fcntl.LOCK_EX)
        except Exception:
            if self._handle is not None:
                self._handle.close()
                self._handle = None
            self._thread_lock.release()
            raise

    def release(self) -> None:
        try:
            if self._handle is not None and fcntl is not None:
                fcntl.flock(self._handle.fileno(), fcntl.LOCK_UN)
            if self._handle is not None:
                self._handle.close()
        finally:
            self._handle = None
            self._thread_lock.release()


class SemanticCatalogRetriever:
    _INDEX_VERSION = "4"

    def __init__(
        self,
        catalog: SemanticCatalog,
        *,
        context_top_k: int,
        vector_index_path: str | None,
        embedding_client: EmbeddingClient | None,
        mode: SemanticCatalogMode = SemanticCatalogMode.AUTO,
        embedding_batch_size: int = 32,
        embedding_concurrency: int = 3,
        auto_full_context_max_chars: int = 30000,
    ) -> None:
        self.catalog = catalog
        self.context_top_k = context_top_k
        self.vector_index_path = vector_index_path
        self.embedding_client = embedding_client
        self.mode = mode
        self.embedding_batch_size = embedding_batch_size
        self.embedding_concurrency = embedding_concurrency
        self.auto_full_context_max_chars = auto_full_context_max_chars
        self._prepare_lock = asyncio.Lock()
        self._documents = _documents(catalog)
        self._documents_by_id = {
            document.doc_id: document for document in self._documents
        }
        embedding_name = type(embedding_client).__qualname__
        embedding_source = getattr(
            embedding_client,
            "model_path",
            getattr(embedding_client, "model", ""),
        )
        signature = _catalog_signature(self._documents)
        self._signature = hashlib.sha256(
            f"{signature}:{embedding_name}:{embedding_source}".encode()
        ).hexdigest()

    @property
    def vector_ready(self) -> bool:
        return bool(self._documents) and all(
            document.vector is not None for document in self._documents
        )

    def _cache_path(self) -> Path | None:
        return Path(self.vector_index_path) if self.vector_index_path else None

    def _load_cache(self) -> bool:
        path = self._cache_path()
        if not path or not path.exists():
            return False
        try:
            with path.open("rb") as source:
                if source.read(16) != b"SQLite format 3\x00":
                    return False
            connection = sqlite3.connect(path)
            try:
                metadata = dict(connection.execute("SELECT key, value FROM metadata"))
                if metadata.get("version") != self._INDEX_VERSION:
                    return False
                if metadata.get("signature") != self._signature:
                    return False
                rows = connection.execute(
                    "SELECT id, vector FROM vectors ORDER BY position"
                ).fetchall()
            finally:
                connection.close()
            by_id = {doc_id: json.loads(vector) for doc_id, vector in rows}
            if set(by_id) != set(self._documents_by_id):
                return False
            for document in self._documents:
                vector = by_id[document.doc_id]
                if not isinstance(vector, list):
                    return False
                document.vector = [float(value) for value in vector]
            return True
        except (OSError, sqlite3.Error, ValueError, TypeError):
            return False

    def _save_cache(self) -> None:
        path = self._cache_path()
        if not path or not self.vector_ready:
            return
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_name(
            f".{path.name}.{os.getpid()}.{threading.get_ident()}.tmp"
        )
        try:
            connection = sqlite3.connect(temporary)
            try:
                connection.execute(
                    "CREATE TABLE metadata (key TEXT PRIMARY KEY, value TEXT NOT NULL)"
                )
                connection.execute(
                    "CREATE TABLE vectors ("
                    "id TEXT PRIMARY KEY, position INTEGER NOT NULL, vector TEXT NOT NULL)"
                )
                connection.executemany(
                    "INSERT INTO metadata(key, value) VALUES (?, ?)",
                    [
                        ("version", self._INDEX_VERSION),
                        ("signature", self._signature),
                    ],
                )
                connection.executemany(
                    "INSERT INTO vectors VALUES (?, ?, ?)",
                    [
                        (
                            document.doc_id,
                            position,
                            json.dumps(document.vector),
                        )
                        for position, document in enumerate(self._documents)
                    ],
                )
                connection.commit()
            finally:
                connection.close()
            os.replace(temporary, path)
        finally:
            if temporary.exists():
                temporary.unlink()

    async def _embed_documents(self) -> None:
        assert self.embedding_client is not None
        semaphore = asyncio.Semaphore(self.embedding_concurrency)
        batches = [
            self._documents[index : index + self.embedding_batch_size]
            for index in range(0, len(self._documents), self.embedding_batch_size)
        ]

        async def embed(
            batch: Sequence[_CatalogDocument],
        ) -> list[list[float]]:
            async with semaphore:
                return await self.embedding_client.embed_documents(
                    [document.content for document in batch]
                )

        results = await asyncio.gather(*(embed(batch) for batch in batches))
        for batch, vectors in zip(batches, results, strict=True):
            if len(vectors) != len(batch):
                raise ValueError("嵌入模型返回的向量数量与文档数量不一致")
            for document, vector in zip(batch, vectors, strict=True):
                document.vector = [float(value) for value in vector]

    async def prepare(self) -> str:
        if self.embedding_client is None:
            return "disabled"
        if self.vector_ready:
            return "memory"
        async with self._prepare_lock:
            if self.vector_ready:
                return "memory"
            path = self._cache_path()
            if path is None:
                await self._embed_documents()
                return "built_memory"
            async with _async_index_lock(path):
                lock = _IndexLock(path)
                acquire_task = asyncio.create_task(asyncio.to_thread(lock.acquire))
                try:
                    await asyncio.shield(acquire_task)
                except asyncio.CancelledError:
                    await acquire_task
                    lock.release()
                    raise
                try:
                    if await asyncio.to_thread(self._load_cache):
                        return "disk_cache"
                    await self._embed_documents()
                    await asyncio.to_thread(self._save_cache)
                    return "built_disk"
                finally:
                    lock.release()

    def _eligible_models(self, filters: dict[str, Any]) -> set[str]:
        allowed_keys = {"model_name", "model_type", "title", "business_domain"}
        invalid = set(filters) - allowed_keys
        if invalid:
            raise SemanticContextRetrievalError(
                f"不支持的 catalog metadata filter：{sorted(invalid)}"
            )
        names: set[str] = set()
        for name, model in self.catalog.models.items():
            values = {
                "model_name": name,
                "model_type": model.model_type,
                "title": model.title,
                "business_domain": model.business_domain,
            }
            if all(
                actual in expected
                if isinstance(expected, (list, tuple, set))
                else actual == expected
                for key, expected in filters.items()
                for actual in [values[key]]
            ):
                names.add(name)
        if not names:
            raise SemanticContextRetrievalError(
                "catalog metadata filter 未匹配到任何受治理的公开模型"
            )
        return names

    def _catalog_subset(self, model_names: set[str]) -> SemanticCatalog:
        return SemanticCatalog(
            models={
                name: self.catalog.models[name]
                for name in sorted(model_names)
            }
        )

    def _catalog_size(self, model_names: set[str]) -> int:
        payload = {
            name: _model_payload(self.catalog.models[name])
            for name in sorted(model_names)
        }
        return len(_content(payload))

    def _resolve_mode(
        self,
        requested_mode: SemanticCatalogMode,
        model_names: set[str],
    ) -> SemanticCatalogMode:
        if requested_mode != SemanticCatalogMode.AUTO:
            return requested_mode
        if self._catalog_size(model_names) <= self.auto_full_context_max_chars:
            return SemanticCatalogMode.FULL
        return SemanticCatalogMode.VECTOR

    @staticmethod
    def _eligible_document(
        document: _CatalogDocument,
        model_names: set[str],
    ) -> bool:
        return bool(set(document.related_models) & model_names)

    def _scoped_document(
        self,
        document: _CatalogDocument,
        model_names: set[str],
    ) -> _CatalogDocument | None:
        if not self._eligible_document(document, model_names):
            return None
        if document.item_type != "join_component":
            return document
        related_models = tuple(
            name for name in document.related_models if name in model_names
        )
        if related_models == document.related_models:
            return document
        component = document.payload["component"]
        payload = _component_payload(
            component,
            [self.catalog.models[name] for name in related_models],
        )
        return _CatalogDocument(
            doc_id=document.doc_id,
            item_type=document.item_type,
            name=document.name,
            model_name=None,
            content=_content(payload),
            payload=payload,
            related_models=related_models,
        )

    def _rank(
        self,
        question: str,
        vector: Sequence[float] | None,
        model_names: set[str],
        limit: int,
    ) -> list[_ScoredDocument]:
        ranked: list[_ScoredDocument] = []
        for document in self._documents:
            scoped_document = self._scoped_document(document, model_names)
            if scoped_document is None:
                continue
            vector_score = (
                _cosine(vector, scoped_document.vector)
                if vector is not None and scoped_document.vector is not None
                else 0.0
            )
            lexical_score = max(
                _lexical_score(question, scoped_document.name),
                _lexical_score(question, scoped_document.content),
            )
            ranked.append(_ScoredDocument(
                score=max(vector_score, lexical_score),
                document=scoped_document,
            ))
        ranked.sort(key=lambda item: (-item.score, item.document.doc_id))
        return ranked[:limit]

    @staticmethod
    def _context_item(
        document: _CatalogDocument,
        score: float | None,
    ) -> CatalogContextItem:
        return CatalogContextItem(
            item_type=document.item_type,
            name=document.name,
            model_name=document.model_name,
            score=round(score, 6) if score is not None else None,
            payload=document.payload,
        )

    def _full_items(self, model_names: set[str]) -> list[CatalogContextItem]:
        items = [
            CatalogContextItem(
                item_type="model",
                name=name,
                model_name=name,
                payload=_model_payload(self.catalog.models[name]),
            )
            for name in sorted(model_names)
        ]
        component_documents = [
            scoped
            for document in self._documents
            if document.item_type == "join_component"
            and (scoped := self._scoped_document(document, model_names)) is not None
        ]
        items.extend(
            self._context_item(document, None)
            for document in component_documents
        )
        return items

    def _search_items(
        self,
        ranked: list[_ScoredDocument],
        model_names: set[str],
    ) -> list[CatalogContextItem]:
        items: list[CatalogContextItem] = []
        seen: set[tuple[str, str]] = set()

        def add(document: _CatalogDocument, score: float | None) -> None:
            key = (document.item_type, document.name)
            if key in seen:
                return
            seen.add(key)
            items.append(self._context_item(document, score))

        for item in ranked:
            add(item.document, item.score)
        parent_models = {
            item.document.model_name
            for item in ranked
            if item.document.model_name in model_names
        }
        for model_name in sorted(parent_models):
            assert model_name is not None
            add(self._documents_by_id[f"model:{model_name}"], None)
            model = self.catalog.models[model_name]
            if model.model_type != "cube" or model.connected_component is None:
                continue
            component_id = f"join_component:{model.connected_component}"
            component_document = self._documents_by_id.get(component_id)
            if component_document is not None:
                scoped_component = self._scoped_document(
                    component_document,
                    model_names,
                )
                if scoped_component is not None:
                    add(scoped_component, None)
        return items

    async def retrieve(
        self,
        question: str,
        *,
        mode: SemanticCatalogMode | None = None,
        metadata_filters: dict[str, Any] | None = None,
        limit: int | None = None,
        retrieval_round: int = 1,
    ) -> RetrievedSemanticContext:
        model_names = self._eligible_models(dict(metadata_filters or {}))
        effective_mode = self._resolve_mode(mode or self.mode, model_names)
        catalog = self._catalog_subset(model_names)
        if effective_mode == SemanticCatalogMode.FULL:
            items = self._full_items(model_names)
            hits = [
                RetrievalHit(
                    item_type=item.item_type,
                    name=item.name,
                    model_name=item.model_name,
                )
                for item in items
            ]
            return RetrievedSemanticContext(
                mode=effective_mode,
                catalog=catalog,
                context=SemanticContext(
                    strategy=effective_mode,
                    items=items,
                    retrieval_round=retrieval_round,
                ),
                trace=RetrievalTrace(
                    strategy=effective_mode,
                    queries=[question],
                    hits=hits,
                    index_source="full_catalog",
                ),
            )
        warnings: list[str] = []
        index_source = "disabled"
        question_vector: Sequence[float] | None = None
        if self.embedding_client is not None:
            try:
                index_source = await self.prepare()
                question_vector = await self.embedding_client.embed_query(question)
            except Exception as exc:
                warnings.append(
                    "向量检索不可用，已使用统一词法目录检索。"
                    f"原因：{str(exc)[:200]}"
                )
                index_source = f"lexical_fallback:{index_source}"
        else:
            warnings.append("嵌入模型不可用，已使用统一词法目录检索。")
            index_source = "lexical"
        ranked = self._rank(
            question,
            question_vector,
            model_names,
            limit or self.context_top_k,
        )
        items = self._search_items(ranked, model_names)
        hits = [
            RetrievalHit(
                item_type=item.document.item_type,
                name=item.document.name,
                model_name=item.document.model_name,
                score=round(item.score, 6),
            )
            for item in ranked
        ]
        return RetrievedSemanticContext(
            mode=effective_mode,
            catalog=catalog,
            context=SemanticContext(
                strategy=effective_mode,
                items=items,
                retrieval_round=retrieval_round,
            ),
            trace=RetrievalTrace(
                strategy=effective_mode,
                queries=[question],
                hits=hits,
                index_source=index_source,
            ),
            warnings=warnings,
        )


def merge_retrieved_context(
    current: RetrievedSemanticContext,
    refreshed: RetrievedSemanticContext,
) -> RetrievedSemanticContext:
    items: list[CatalogContextItem] = []
    seen_items: set[tuple[str, str]] = set()
    for item in [*current.context.items, *refreshed.context.items]:
        key = (item.item_type, item.name)
        if key in seen_items:
            continue
        seen_items.add(key)
        items.append(item)
    hits: list[RetrievalHit] = []
    seen_hits: set[tuple[str, str, str | None]] = set()
    for hit in [*current.trace.hits, *refreshed.trace.hits]:
        key = (hit.item_type, hit.name, hit.model_name)
        if key in seen_hits:
            continue
        seen_hits.add(key)
        hits.append(hit)
    warnings = list(current.warnings)
    warnings.extend(
        warning for warning in refreshed.warnings if warning not in warnings
    )
    return RetrievedSemanticContext(
        mode=refreshed.mode,
        catalog=refreshed.catalog,
        context=SemanticContext(
            strategy=refreshed.context.strategy,
            items=items,
            retrieval_round=refreshed.context.retrieval_round,
        ),
        trace=RetrievalTrace(
            strategy=refreshed.trace.strategy,
            queries=list(dict.fromkeys([
                *current.trace.queries,
                *refreshed.trace.queries,
            ])),
            hits=hits,
            index_source=refreshed.trace.index_source,
        ),
        warnings=warnings,
    )
