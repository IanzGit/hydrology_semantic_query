from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import math
import os
import re
import sqlite3
import threading
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

from langchain.tools import ToolRuntime
from langgraph.types import Command

from app.agents.scenarios.hydrology_semantic_query.models import (
    CatalogContextItem,
    CatalogMember,
    CatalogModel,
    FailureKind,
    RetrievalHit,
    RetrievalTrace,
    SemanticCatalog,
    SemanticCatalogMode,
    SemanticContext,
    StepStatus,
)
from app.agents.scenarios.hydrology_semantic_query.runtime import (
    HydrologySemanticQueryServices,
    HydrologySemanticQueryState,
    build_error,
    build_step,
    outcome_for_error,
    request_data,
    safe_response_excerpt,
    thought_output,
)

from .common import error_payload, tool_message

logger = logging.getLogger("uvicorn.error")

try:
    import fcntl
except ImportError:
    fcntl = None

_PATH_LOCKS: dict[str, threading.Lock] = {}
_PATH_LOCKS_GUARD = threading.Lock()
_ASYNC_PATH_LOCKS: dict[tuple[int, str], asyncio.Lock] = {}


def async_index_lock(path: Path) -> asyncio.Lock:
    key = (id(asyncio.get_running_loop()), str(path.resolve()))
    with _PATH_LOCKS_GUARD:
        return _ASYNC_PATH_LOCKS.setdefault(key, asyncio.Lock())


class IndexLock:
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


def load_cache(
    path: Path,
    *,
    version: str,
    signature: str,
    documents: list[CatalogDocument],
    documents_by_id: dict[str, CatalogDocument],
    on_rejected: Callable[[str], None] | None = None,
) -> bool:
    def rejected(reason: str) -> bool:
        if on_rejected is not None:
            on_rejected(reason)
        return False

    if not path.exists():
        return rejected("索引文件不存在")
    try:
        with path.open("rb") as source:
            if source.read(16) != b"SQLite format 3\x00":
                return rejected("索引文件不是有效的 SQLite 数据库")
        connection = sqlite3.connect(path)
        try:
            metadata = dict(connection.execute("SELECT key, value FROM metadata"))
            if metadata.get("version") != version:
                return rejected(
                    f"索引版本不匹配：expected={version} "
                    f"actual={metadata.get('version', '<missing>')}"
                )
            if metadata.get("signature") != signature:
                return rejected("目录或嵌入模型签名已变化")
            rows = connection.execute(
                "SELECT id, vector FROM vectors ORDER BY position"
            ).fetchall()
        finally:
            connection.close()
        by_id = {doc_id: json.loads(vector) for doc_id, vector in rows}
        if set(by_id) != set(documents_by_id):
            return rejected(
                "索引文档集合不一致："
                f"expected={len(documents_by_id)} actual={len(by_id)}"
            )
        for document in documents:
            vector = by_id[document.doc_id]
            if not isinstance(vector, list):
                return rejected(f"索引向量格式无效：document={document.doc_id}")
            document.vector = [float(value) for value in vector]
        return True
    except (OSError, sqlite3.Error, ValueError, TypeError) as exc:
        return rejected(f"索引读取失败：{type(exc).__name__}: {str(exc)[:200]}")


def save_cache(
    path: Path,
    *,
    version: str,
    signature: str,
    documents: list[CatalogDocument],
) -> None:
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
                [("version", version), ("signature", signature)],
            )
            connection.executemany(
                "INSERT INTO vectors VALUES (?, ?, ?)",
                [
                    (document.doc_id, position, json.dumps(document.vector))
                    for position, document in enumerate(documents)
                ],
            )
            connection.commit()
        finally:
            connection.close()
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()

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
        self.model_path = str(path)
        self._model: Any = None
        self._model_lock = threading.Lock()

    def _load_model(self) -> Any:
        if self._model is not None:
            return self._model
        with self._model_lock:
            if self._model is None:
                from sentence_transformers import SentenceTransformer

                self._model = SentenceTransformer(self.model_path)
        return self._model

    def _embed_documents_blocking(self, texts: Sequence[str]) -> list[list[float]]:
        if not texts:
            return []
        vectors = self._load_model().encode(
            list(texts),
            convert_to_numpy=True,
            show_progress_bar=False,
        )
        return [[float(value) for value in vector] for vector in vectors]

    async def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
        return await asyncio.to_thread(self._embed_documents_blocking, texts)

    async def embed_query(self, text: str) -> list[float]:
        return (await self.embed_documents([text]))[0]

@dataclass
class CatalogDocument:
    doc_id: str
    item_type: str
    name: str
    model_name: str | None
    content: str
    payload: dict[str, Any]
    related_models: tuple[str, ...] = ()
    vector: list[float] | None = None


def member_payload(member: CatalogMember) -> dict[str, Any]:
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
    }


def model_summary_payload(model: CatalogModel) -> dict[str, Any]:
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
        "members": [
            {
                "name": member.name,
                "kind": member.member_type,
                "type": member.data_type,
            }
            for member in model.members.values()
        ],
    }


def model_payload(model: CatalogModel) -> dict[str, Any]:
    return {
        **model_summary_payload(model),
        "members": [member_payload(member) for member in model.members.values()],
    }


def component_payload(
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


def content(payload: dict[str, Any]) -> str:
    return json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def build_documents(catalog: SemanticCatalog) -> list[CatalogDocument]:
    documents: list[CatalogDocument] = []
    components: dict[str | int, list[CatalogModel]] = {}
    for model in catalog.models.values():
        model_summary = model_summary_payload(model)
        documents.append(CatalogDocument(
            doc_id=f"model:{model.name}",
            item_type="model",
            name=model.name,
            model_name=model.name,
            content=content(model_payload(model)),
            payload=model_summary,
            related_models=(model.name,),
        ))
        for member in model.members.values():
            payload = {
                "model": {
                    "name": model.name,
                    "type": model.model_type,
                    "title": model.title,
                    "description": model.description,
                },
                "member": member_payload(member),
            }
            documents.append(CatalogDocument(
                doc_id=f"member:{member.name}",
                item_type="member",
                name=member.name,
                model_name=model.name,
                content=content(payload),
                payload=payload,
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
                payload = {
                    "model": model.name,
                    "model_title": model.title,
                    "folder": folder,
                    "members": [member_payload(member) for member in folder_members],
                }
                documents.append(CatalogDocument(
                    doc_id=f"view_folder:{model.name}:{folder}",
                    item_type="view_folder",
                    name=f"{model.name}:{folder}",
                    model_name=model.name,
                    content=content(payload),
                    payload=payload,
                    related_models=(model.name,),
                ))
        if model.model_type == "cube" and model.connected_component is not None:
            components.setdefault(model.connected_component, []).append(model)
    for component, models in components.items():
        ordered = sorted(models, key=lambda model: model.name)
        payload = component_payload(component, ordered)
        documents.append(CatalogDocument(
            doc_id=f"join_component:{component}",
            item_type="join_component",
            name=str(component),
            model_name=None,
            content=content(payload),
            payload=payload,
            related_models=tuple(model.name for model in ordered),
        ))
    return documents


def catalog_signature(documents: Sequence[CatalogDocument]) -> str:
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
    return hashlib.sha256(content({"documents": payload}).encode()).hexdigest()

@dataclass(frozen=True)
class ScoredDocument:
    score: float
    document: CatalogDocument


def cosine(left: Sequence[float], right: Sequence[float]) -> float:
    if len(left) != len(right):
        raise ValueError("嵌入向量维度不一致")
    dot = sum(a * b for a, b in zip(left, right, strict=True))
    left_norm = math.sqrt(sum(value * value for value in left))
    right_norm = math.sqrt(sum(value * value for value in right))
    if left_norm == 0 or right_norm == 0:
        return 0.0
    return dot / (left_norm * right_norm)


def normalized(value: str) -> str:
    return re.sub(r"[^0-9a-zA-Z\u4e00-\u9fff]+", "", value).lower()


def lexical_similarity(left: str, right: str) -> float:
    left_value = normalized(left)
    right_value = normalized(right)
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

class SemanticContextRetrievalError(ValueError):
    pass


class VectorIndexNotReadyError(RuntimeError):
    pass


@dataclass
class RetrievedSemanticContext:
    mode: SemanticCatalogMode
    catalog: SemanticCatalog
    context: SemanticContext
    trace: RetrievalTrace
    warnings: list[str] = field(default_factory=list)


class SemanticCatalogRetriever:
    _INDEX_VERSION = "4"

    def __init__(
        self,
        catalog: SemanticCatalog,
        *,
        model_top_k: int = 5,
        context_top_k: int,
        vector_index_path: str | None,
        embedding_client: EmbeddingClient | None,
        mode: SemanticCatalogMode = SemanticCatalogMode.AUTO,
        embedding_batch_size: int = 32,
        embedding_concurrency: int = 3,
        auto_full_context_max_chars: int = 30000,
    ) -> None:
        self.catalog = catalog
        self.model_top_k = model_top_k
        self.context_top_k = context_top_k
        self.vector_index_path = vector_index_path
        self.embedding_client = embedding_client
        self.mode = mode
        self.embedding_batch_size = embedding_batch_size
        self.embedding_concurrency = embedding_concurrency
        self.auto_full_context_max_chars = auto_full_context_max_chars
        self._prepare_lock = asyncio.Lock()
        self._documents = build_documents(catalog)
        self._documents_by_id = {
            document.doc_id: document for document in self._documents
        }
        self.cache_miss_reason: str | None = None
        embedding_name = type(embedding_client).__qualname__
        embedding_source = getattr(
            embedding_client,
            "model_path",
            getattr(embedding_client, "model", ""),
        )
        signature = catalog_signature(self._documents)
        self._signature = hashlib.sha256(
            f"{signature}:{embedding_name}:{embedding_source}".encode()
        ).hexdigest()

    @property
    def vector_ready(self) -> bool:
        return bool(self._documents) and all(
            document.vector is not None for document in self._documents
        )

    @property
    def document_count(self) -> int:
        return len(self._documents)

    def _cache_path(self) -> Path | None:
        return Path(self.vector_index_path) if self.vector_index_path else None

    def _load_cache(self) -> bool:
        path = self._cache_path()
        if not path:
            self.cache_miss_reason = "未配置索引路径"
            return False
        self.cache_miss_reason = None

        def record_rejection(reason: str) -> None:
            self.cache_miss_reason = reason

        return load_cache(
            path,
            version=self._INDEX_VERSION,
            signature=self._signature,
            documents=self._documents,
            documents_by_id=self._documents_by_id,
            on_rejected=record_rejection,
        )

    def _save_cache(self) -> None:
        path = self._cache_path()
        if not path or not self.vector_ready:
            return
        save_cache(
            path,
            version=self._INDEX_VERSION,
            signature=self._signature,
            documents=self._documents,
        )

    async def _embed_documents(self) -> None:
        assert self.embedding_client is not None
        semaphore = asyncio.Semaphore(self.embedding_concurrency)
        batches = [
            self._documents[index : index + self.embedding_batch_size]
            for index in range(0, len(self._documents), self.embedding_batch_size)
        ]

        async def embed(
            batch: Sequence[CatalogDocument],
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

    async def _acquire_file_lock(self, path: Path) -> IndexLock:
        lock = IndexLock(path)
        acquire_task = asyncio.create_task(asyncio.to_thread(lock.acquire))
        try:
            await asyncio.shield(acquire_task)
        except asyncio.CancelledError:
            await acquire_task
            lock.release()
            raise
        return lock

    async def build_index(self, *, force: bool = False) -> str:
        if self.embedding_client is None:
            raise ValueError("部署向量索引需要可用的嵌入模型")
        path = self._cache_path()
        if path is None:
            raise ValueError("部署向量索引需要非空的持久化路径")
        if self.vector_ready and not force:
            return "memory"
        async with self._prepare_lock:
            if self.vector_ready and not force:
                return "memory"
            async with async_index_lock(path):
                lock = await self._acquire_file_lock(path)
                try:
                    if not force and await asyncio.to_thread(self._load_cache):
                        return "disk_cache"
                    if force:
                        self.cache_miss_reason = "已请求强制重建"
                    logger.warning(
                        "语义目录向量索引缓存未复用：%s",
                        self.cache_miss_reason,
                    )
                    await self._embed_documents()
                    await asyncio.to_thread(self._save_cache)
                    return "built_disk"
                finally:
                    lock.release()

    async def load_prebuilt_index(self) -> str:
        if self.vector_ready:
            return "memory"
        path = self._cache_path()
        if path is None:
            raise VectorIndexNotReadyError(
                "语义目录向量索引未配置持久化路径，请先执行 Cube start 向量重建脚本"
            )
        async with self._prepare_lock:
            if self.vector_ready:
                return "memory"
            async with async_index_lock(path):
                lock = await self._acquire_file_lock(path)
                try:
                    loaded = await asyncio.to_thread(self._load_cache)
                finally:
                    lock.release()
            if not loaded:
                raise VectorIndexNotReadyError(
                    f"语义目录向量索引缺失、损坏或与最新目录不一致：{path}。"
                    f"原因：{self.cache_miss_reason}。"
                    "请先执行 Cube start 向量重建脚本"
                )
            return "disk_cache"

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
            name: model_payload(self.catalog.models[name])
            for name in sorted(model_names)
        }
        return len(content(payload))

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
        document: CatalogDocument,
        model_names: set[str],
    ) -> bool:
        return bool(set(document.related_models) & model_names)

    def _scoped_document(
        self,
        document: CatalogDocument,
        model_names: set[str],
    ) -> CatalogDocument | None:
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
        payload = component_payload(
            component,
            [self.catalog.models[name] for name in related_models],
        )
        return CatalogDocument(
            doc_id=document.doc_id,
            item_type=document.item_type,
            name=document.name,
            model_name=None,
            content=content(payload),
            payload=payload,
            related_models=related_models,
        )

    def _rank(
        self,
        question: str,
        vector: Sequence[float] | None,
        model_names: set[str],
        limit: int,
        item_types: set[str] | None = None,
    ) -> list[ScoredDocument]:
        ranked: list[ScoredDocument] = []
        for document in self._documents:
            if item_types is not None and document.item_type not in item_types:
                continue
            scoped_document = self._scoped_document(document, model_names)
            if scoped_document is None:
                continue
            vector_score = (
                cosine(vector, scoped_document.vector)
                if vector is not None and scoped_document.vector is not None
                else 0.0
            )
            lexical_score = max(
                lexical_similarity(question, scoped_document.name),
                lexical_similarity(question, scoped_document.content),
            )
            ranked.append(ScoredDocument(
                score=max(vector_score, lexical_score),
                document=scoped_document,
            ))
        ranked.sort(key=lambda item: (-item.score, item.document.doc_id))
        return ranked[:limit]

    @staticmethod
    def _context_item(
        document: CatalogDocument,
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
                payload=model_payload(self.catalog.models[name]),
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
        ranked: list[ScoredDocument],
        model_names: set[str],
    ) -> list[CatalogContextItem]:
        items: list[CatalogContextItem] = []
        seen: set[tuple[str, str]] = set()

        def add(document: CatalogDocument, score: float | None) -> None:
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
                index_source = await self.load_prebuilt_index()
            except VectorIndexNotReadyError:
                raise
            try:
                question_vector = await self.embedding_client.embed_query(question)
            except Exception as exc:
                warnings.append(
                    "向量检索不可用，已使用两阶段词法目录检索。"
                    f"原因：{str(exc)[:200]}"
                )
                index_source = f"lexical_fallback:{index_source}"
        else:
            warnings.append("嵌入模型不可用，已使用两阶段词法目录检索。")
            index_source = "lexical"
        model_limit = min(len(model_names), limit or self.model_top_k)
        model_ranked = self._rank(
            question,
            question_vector,
            model_names,
            model_limit,
            {"model"},
        )
        candidate_model_names = {
            item.document.model_name
            for item in model_ranked
            if item.document.model_name is not None
        }
        all_member_ranked = self._rank(
            question,
            question_vector,
            candidate_model_names,
            len(self._documents),
            {"member", "view_folder"},
        )
        member_limit = limit or self.context_top_k
        member_ranked: list[ScoredDocument] = []
        selected_document_ids: set[str] = set()
        for model_item in model_ranked:
            if len(member_ranked) >= member_limit:
                break
            model_name = model_item.document.model_name
            candidate = next(
                (
                    item
                    for item in all_member_ranked
                    if item.document.model_name == model_name
                    and item.document.doc_id not in selected_document_ids
                ),
                None,
            )
            if candidate is None:
                continue
            member_ranked.append(candidate)
            selected_document_ids.add(candidate.document.doc_id)
        for item in all_member_ranked:
            if len(member_ranked) >= member_limit:
                break
            if item.document.doc_id in selected_document_ids:
                continue
            member_ranked.append(item)
            selected_document_ids.add(item.document.doc_id)
        ranked = [*model_ranked, *member_ranked]
        items = self._search_items(ranked, candidate_model_names)
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
    item_positions: dict[tuple[str, str], int] = {}
    for item in [*current.context.items, *refreshed.context.items]:
        key = (item.item_type, item.name)
        if key in item_positions:
            items[item_positions[key]] = item
            continue
        item_positions[key] = len(items)
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

class CatalogSearchRuntime:
    def __init__(self, services: HydrologySemanticQueryServices) -> None:
        self.services = services
        self._retriever_lock = asyncio.Lock()
        self.retriever: SemanticCatalogRetriever | None = None
        self.retrieval_limits: list[int | None] = []
        self.retrieval_questions: list[str] = []
        if services.embedding is None and services.settings.embedding_model:
            try:
                services.embedding = SentenceTransformerEmbedding(
                    services.settings.embedding_model
                )
            except Exception as exc:
                warning = (
                    "嵌入模型不可用，已改用两阶段词法目录检索。"
                    f"原因：{str(exc)[:200]}"
                )
                services.startup_warnings.append(warning)
                logger.warning("hydrology_semantic_query startup warning: %s", warning)

    async def retrieve_context(
        self,
        question: str,
        catalog: SemanticCatalog,
        *,
        mode: SemanticCatalogMode | None = None,
        metadata_filters: dict[str, Any] | None = None,
        limit: int | None = None,
        retrieval_round: int = 1,
    ) -> RetrievedSemanticContext:
        self.retrieval_questions.append(question)
        self.retrieval_limits.append(limit)
        async with self._retriever_lock:
            if self.retriever is None or self.retriever.catalog != catalog:
                settings = self.services.settings
                self.retriever = SemanticCatalogRetriever(
                    catalog,
                    model_top_k=settings.model_top_k,
                    context_top_k=settings.context_top_k,
                    vector_index_path=settings.vector_index_path,
                    embedding_client=self.services.embedding,
                    mode=settings.catalog_mode,
                    embedding_batch_size=settings.embedding_batch_size,
                    embedding_concurrency=settings.embedding_concurrency,
                    auto_full_context_max_chars=settings.auto_full_context_max_chars,
                )
            retriever = self.retriever
        return await retriever.retrieve(
            question,
            mode=mode,
            metadata_filters=metadata_filters,
            limit=limit,
            retrieval_round=retrieval_round,
        )


def _retrieval_metadata(retrieved: RetrievedSemanticContext) -> dict[str, Any]:
    item_counts: dict[str, int] = {}
    for item in retrieved.context.items:
        item_counts[item.item_type] = item_counts.get(item.item_type, 0) + 1
    return {
        "catalog_mode": retrieved.mode.value,
        "strategy": retrieved.context.strategy.value,
        "retrieval_round": retrieved.context.retrieval_round,
        "context_item_count": len(retrieved.context.items),
        "context_item_counts": item_counts,
        "retrieval_queries": retrieved.trace.queries,
        "retrieval_hits": [
            hit.model_dump(mode="json", exclude_none=True)
            for hit in retrieved.trace.hits
        ],
        "accessible_model_count": len(retrieved.catalog.models),
        "index_source": retrieved.trace.index_source,
    }


def _append_unique(target: list[str], values: list[str]) -> None:
    target.extend(value for value in values if value not in target)

async def search_semantic_catalog_service(
    *,
    query: str,
    runtime: ToolRuntime,
    limit: int | None,
    search_runtime: CatalogSearchRuntime,
    services: HydrologySemanticQueryServices,
) -> Command:
    state: HydrologySemanticQueryState = runtime.state
    started = time.perf_counter()
    steps = list(state.get("steps", []))
    warnings = list(state.get("warnings", []))
    current_context = state.get("semantic_context")
    retrieval_round = current_context.retrieval_round + 1 if current_context else 1
    try:
        request = request_data(state, services.settings)
        full_catalog = state.get("full_catalog")
        if full_catalog is None:
            raise RuntimeError("Cube 语义目录尚未加载")
        retrieved = await search_runtime.retrieve_context(
            query.strip(),
            full_catalog,
            mode=request["catalog_mode"],
            metadata_filters=request["catalog_metadata_filters"],
            limit=limit,
            retrieval_round=retrieval_round,
        )
        if current_context is None:
            merged = retrieved
        else:
            current = RetrievedSemanticContext(
                mode=state["catalog_mode"],
                catalog=state["catalog"],
                context=current_context,
                trace=state["retrieval_trace"],
            )
            merged = merge_retrieved_context(current, retrieved)
        _append_unique(warnings, merged.warnings)
        stage = "semantic_retrieval" if current_context is None else "context_refresh"
        steps.append(build_step(
            stage,
            started,
            attempt=retrieval_round,
            status=StepStatus.SUCCESS,
            metadata=_retrieval_metadata(merged),
        ))
        observation = {
            "ok": True,
            "kind": "semantic_catalog",
            "strategy": retrieved.mode.value,
            "retrieval_round": retrieval_round,
            "context": retrieved.context.model_dump(mode="json", exclude_none=True),
            "trace": retrieved.trace.model_dump(mode="json", exclude_none=True),
            "warnings": retrieved.warnings,
        }
        return Command(update={
            "messages": [tool_message(runtime, "search_semantic_catalog", observation)],
            "catalog": merged.catalog,
            "catalog_mode": merged.mode,
            "semantic_context": merged.context,
            "retrieval_trace": merged.trace,
            "steps": steps,
            "warnings": warnings,
            "stage": stage,
            "last_tool_terminal": False,
            "search_count": state.get("search_count", 0) + 1,
            "stream_outputs": thought_output(
                "检索语义目录",
                f"第 {retrieval_round} 轮检索返回 {len(retrieved.context.items)} 个相关目录项",
            ),
        })
    except Exception as exc:
        validation_error = isinstance(exc, SemanticContextRetrievalError)
        error = build_error(
            stage="semantic_retrieval",
            code=exc.__class__.__name__,
            kind=FailureKind.VALIDATION if validation_error else FailureKind.SYSTEM,
            exc=exc,
            retryable=False,
        )
        steps.append(build_step(
            "semantic_retrieval",
            started,
            attempt=retrieval_round,
            status=StepStatus.FAILED,
            summary=safe_response_excerpt(str(exc)),
        ))
        observation = error_payload(error, terminal=True)
        return Command(update={
            "messages": [
                tool_message(
                    runtime,
                    "search_semantic_catalog",
                    observation,
                    error=True,
                )
            ],
            "steps": steps,
            "stage": "semantic_retrieval",
            "error": error,
            "outcome": outcome_for_error(error),
            "last_tool_terminal": True,
            "stream_outputs": thought_output("检索语义目录", "语义目录检索失败"),
        })
