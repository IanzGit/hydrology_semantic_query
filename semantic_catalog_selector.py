from __future__ import annotations

import asyncio
import hashlib
import json
import math
import os
import re
import sqlite3
import threading
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from itertools import combinations
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

from .models import (
    CatalogMember,
    CatalogModel,
    NeedCandidate,
    ProjectionMode,
    ProjectionPolicy,
    RetrievalIntent,
    RetrievalTrace,
    SemanticCatalog,
    SemanticCatalogMode,
    SemanticContext,
    SemanticModelGap,
    SemanticNeed,
)
from .semantic_context import SemanticJoinGraph

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


@dataclass
class _VectorDocument:
    doc_id: str
    content: str
    metadata: dict[str, Any]
    vector: list[float] | None = None


@dataclass(frozen=True)
class _ScoredDocument:
    score: float
    document: _VectorDocument


@dataclass
class SelectedSemanticCatalog:
    mode: SemanticCatalogMode
    catalog: SemanticCatalog
    selected_models: list[str]
    context: SemanticContext | None
    trace: RetrievalTrace
    gap: SemanticModelGap | None = None
    warnings: list[str] = field(default_factory=list)
    index_source: str = "disabled"


@dataclass(frozen=True)
class NeedBindingCandidate:
    need_index: int
    model_name: str
    member_name: str
    score: float


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
    }


def _model_payload(model: CatalogModel) -> dict[str, Any]:
    return {
        "name": model.name,
        "model_type": model.model_type,
        "title": model.title,
        "description": model.description,
        "ai_context": model.ai_context,
        "aliases": model.aliases,
        "use_cases": model.use_cases,
        "business_domain": model.business_domain,
        "priority": model.business_priority,
        "connected_component": model.connected_component,
        "join_edges": model.join_edges,
        "main_members": [
            {
                "name": member.name,
                "title": member.title,
                "kind": member.member_type,
                "type": member.data_type,
            }
            for member in list(model.members.values())[:12]
        ],
    }


def _model_document(model: CatalogModel) -> _VectorDocument:
    return _VectorDocument(
        doc_id=f"{model.model_type}:{model.name}:overview",
        content=json.dumps(
            _model_payload(model),
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ),
        metadata={
            "part": model.model_type,
            "model_name": model.name,
            "model_type": model.model_type,
            "title": model.title,
            "member_names": [],
        },
    )


def _folder_documents(model: CatalogModel) -> list[_VectorDocument]:
    if model.model_type != "view":
        return []
    documents: list[_VectorDocument] = []
    for folder in model.folders:
        members = [member for member in model.members.values() if member.folder == folder]
        if not members:
            continue
        payload = {
            "model_name": model.name,
            "title": model.title,
            "folder": folder,
            "members": [_member_payload(member) for member in members],
        }
        documents.append(
            _VectorDocument(
                doc_id=f"view:{model.name}:folder:{folder}",
                content=json.dumps(
                    payload,
                    ensure_ascii=False,
                    sort_keys=True,
                    separators=(",", ":"),
                ),
                metadata={
                    "part": "view_folder",
                    "model_name": model.name,
                    "model_type": model.model_type,
                    "title": model.title,
                    "member_names": [member.name for member in members],
                },
            )
        )
    return documents


def _member_document(model: CatalogModel, member: CatalogMember) -> _VectorDocument:
    payload = {
        "model_name": model.name,
        "model_type": model.model_type,
        **_member_payload(member),
    }
    return _VectorDocument(
        doc_id=f"member:{member.name}",
        content=json.dumps(
            payload,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ),
        metadata={
            "part": "member",
            "model_name": model.name,
            "model_type": model.model_type,
            "title": member.title,
            "member_names": [member.name],
            "member_type": member.member_type,
            "data_type": member.data_type,
        },
    )


def _catalog_signature(documents: Sequence[_VectorDocument]) -> str:
    payload = [
        {
            "id": document.doc_id,
            "content": document.content,
            "metadata": document.metadata,
        }
        for document in documents
    ]
    encoded = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode()
    return hashlib.sha256(encoded).hexdigest()


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
        return min(len(left_value), len(right_value)) / max(len(left_value), len(right_value))
    return 0.0


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


class SemanticCatalogSelector:
    _INDEX_VERSION = "3"

    def __init__(
        self,
        catalog: SemanticCatalog,
        *,
        view_top_k: int = 3,
        cube_top_k: int = 5,
        member_top_k: int = 15,
        vector_index_path: str | None,
        embedding_client: EmbeddingClient | None,
        mode: SemanticCatalogMode = SemanticCatalogMode.AUTO,
        embedding_batch_size: int = 32,
        embedding_concurrency: int = 3,
        retrieval_concurrency: int = 3,
        context_member_limit: int = 12,
        catalog_batch_size: int = 4,
        max_cube_models: int = 4,
        member_match_threshold: float = 0.55,
        auto_full_context_max_chars: int = 18000,
    ) -> None:
        self.catalog = catalog
        self.view_top_k = view_top_k
        self.cube_top_k = cube_top_k
        self.member_top_k = member_top_k
        self.vector_index_path = vector_index_path
        self.embedding_client = embedding_client
        self.mode = mode
        self.embedding_batch_size = embedding_batch_size
        self.embedding_concurrency = embedding_concurrency
        self.retrieval_concurrency = retrieval_concurrency
        self.context_member_limit = context_member_limit
        self.catalog_batch_size = catalog_batch_size
        self.max_cube_models = max_cube_models
        self.member_match_threshold = member_match_threshold
        self.auto_full_context_max_chars = auto_full_context_max_chars
        self._prepare_lock = asyncio.Lock()
        self._documents: list[_VectorDocument] = []
        for model in catalog.models.values():
            self._documents.append(_model_document(model))
            self._documents.extend(_folder_documents(model))
            self._documents.extend(
                _member_document(model, member) for member in model.members.values()
            )
        self._member_documents = {
            document.metadata["member_names"][0]: document
            for document in self._documents
            if document.metadata["part"] == "member"
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
            if set(by_id) != {document.doc_id for document in self._documents}:
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
                    "id TEXT PRIMARY KEY, position INTEGER NOT NULL, "
                    "part TEXT NOT NULL, model_name TEXT NOT NULL, "
                    "model_type TEXT NOT NULL, member_names TEXT NOT NULL, "
                    "content TEXT NOT NULL, metadata TEXT NOT NULL, vector TEXT NOT NULL)"
                )
                connection.executemany(
                    "INSERT INTO metadata(key, value) VALUES (?, ?)",
                    [
                        ("version", self._INDEX_VERSION),
                        ("signature", self._signature),
                    ],
                )
                connection.executemany(
                    "INSERT INTO vectors VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                    [
                        (
                            document.doc_id,
                            position,
                            str(document.metadata["part"]),
                            str(document.metadata.get("model_name") or ""),
                            str(document.metadata.get("model_type") or ""),
                            json.dumps(
                                document.metadata.get("member_names") or [],
                                ensure_ascii=False,
                            ),
                            document.content,
                            json.dumps(document.metadata, ensure_ascii=False),
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

    def _batches(self) -> list[list[_VectorDocument]]:
        return [
            self._documents[index : index + self.embedding_batch_size]
            for index in range(0, len(self._documents), self.embedding_batch_size)
        ]

    async def _embed_documents(self) -> None:
        assert self.embedding_client is not None
        semaphore = asyncio.Semaphore(self.embedding_concurrency)

        async def embed(batch: Sequence[_VectorDocument]) -> list[list[float]]:
            async with semaphore:
                return await self.embedding_client.embed_documents(
                    [document.content for document in batch]
                )

        batches = self._batches()
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

    def _rank(
        self,
        question: str,
        vector: Sequence[float] | None,
        *,
        parts: set[str],
        top_k: int,
        eligible_models: set[str] | None = None,
    ) -> list[_ScoredDocument]:
        ranked: list[_ScoredDocument] = []
        for document in self._documents:
            if document.metadata["part"] not in parts:
                continue
            if eligible_models is not None and document.metadata.get("model_name") not in eligible_models:
                continue
            vector_score = (
                _cosine(vector, document.vector)
                if vector is not None and document.vector is not None
                else 0.0
            )
            lexical = max(
                _lexical_score(question, str(document.metadata.get("title") or "")),
                _lexical_score(question, document.content),
            )
            ranked.append(_ScoredDocument(max(vector_score, lexical), document))
        ranked.sort(key=lambda item: (-item.score, item.document.doc_id))
        return ranked[:top_k]

    def _rank_models(
        self,
        question: str,
        vector: Sequence[float] | None,
        *,
        model_type: str,
        top_k: int,
        eligible_models: set[str],
    ) -> list[_ScoredDocument]:
        parts = {model_type, "view_folder"} if model_type == "view" else {"cube"}
        ranked = self._rank(
            question,
            vector,
            parts=parts,
            top_k=max(top_k * 4, top_k),
            eligible_models=eligible_models,
        )
        selected: list[_ScoredDocument] = []
        names: set[str] = set()
        for item in ranked:
            name = str(item.document.metadata["model_name"])
            if name in names:
                continue
            names.add(name)
            selected.append(item)
            if len(selected) == top_k:
                break
        return selected

    @staticmethod
    def _need_key(need: SemanticNeed) -> str:
        parts = [need.usage, need.phrase]
        if need.aggregate:
            parts.append(need.aggregate)
        return ":".join(parts)

    @staticmethod
    def _member_matches_need(member: CatalogMember, need: SemanticNeed) -> bool:
        if need.aggregate is not None:
            return member.member_type == "measure"
        if need.usage in {"select", "group"}:
            return member.member_type == "dimension"
        return member.member_type in {"dimension", "segment"}

    @staticmethod
    def _need_binding_text(need: SemanticNeed, question: str) -> str:
        parts = [f"业务语义：{need.phrase}", f"用途：{need.usage}"]
        if need.aggregate:
            parts.append(f"聚合：{need.aggregate}")
        parts.append(f"原始问题：{question}")
        return "；".join(parts)

    @staticmethod
    def _member_lexical_score(concept: str, member: CatalogMember) -> float:
        names = [member.name.partition(".")[2], member.title, *member.aliases]
        return max((_lexical_score(concept, value) for value in names), default=0.0)

    def _need_member_similarity(
        self,
        need: SemanticNeed,
        vector: Sequence[float] | None,
        member: CatalogMember,
    ) -> float:
        document = self._member_documents[member.name]
        lexical_score = self._member_lexical_score(need.phrase, member)
        if vector is None or document.vector is None:
            return lexical_score
        vector_score = _cosine(vector, document.vector)
        return 0.7 * vector_score + 0.3 * lexical_score

    def _need_member_score(
        self,
        *,
        need: SemanticNeed,
        member: CatalogMember,
        need_vector: Sequence[float] | None,
        full_query_member_scores: dict[str, float],
        scope_score: float,
    ) -> float:
        need_score = self._need_member_similarity(need, need_vector, member)
        query_score = full_query_member_scores.get(member.name, 0.0)
        return round(
            0.45 * need_score + 0.35 * scope_score + 0.20 * query_score,
            6,
        )

    def _need_bindings(
        self,
        needs: list[SemanticNeed],
        need_vectors: dict[str, Sequence[float] | None],
        candidate_models: Iterable[str],
        full_query_member_scores: dict[str, float],
        scope_scores: dict[str, float],
    ) -> dict[str, list[NeedBindingCandidate]]:
        bindings: dict[str, list[NeedBindingCandidate]] = {}
        for index, need in enumerate(needs):
            candidates = bindings.setdefault(self._need_key(need), [])
            for model_name in sorted(set(candidate_models)):
                model = self.catalog.models[model_name]
                for member in model.members.values():
                    if self._member_matches_need(member, need):
                        candidates.append(NeedBindingCandidate(
                            need_index=index,
                            model_name=model_name,
                            member_name=member.name,
                            score=self._need_member_score(
                                need=need,
                                member=member,
                                need_vector=need_vectors.get(self._need_key(need)),
                                full_query_member_scores=full_query_member_scores,
                                scope_score=scope_scores.get(model_name, 0.0),
                            ),
                        ))
        return bindings

    def _binding_coverage(
        self,
        model_names: Iterable[str],
        needs: list[SemanticNeed],
        bindings: dict[str, list[NeedBindingCandidate]],
    ) -> dict[str, float]:
        coverage: dict[str, float] = {}
        for model_name in model_names:
            if not needs:
                coverage[model_name] = 1.0
                continue
            covered = sum(
                any(
                    candidate.model_name == model_name
                    and candidate.score >= self._binding_threshold()
                    for candidate in bindings.get(self._need_key(need), [])
                )
                for need in needs
            )
            coverage[model_name] = covered / len(needs)
        return coverage

    def _binding_threshold(self) -> float:
        return 0.45 * self.member_match_threshold

    def _resolve_need_bindings(
        self,
        needs: list[SemanticNeed],
        bindings: dict[str, list[NeedBindingCandidate]],
        model_names: Iterable[str],
    ) -> tuple[dict[str, str], list[str]]:
        allowed = set(model_names)
        resolved: dict[str, str] = {}
        missing: list[str] = []
        for need in needs:
            candidates = [
                candidate
                for candidate in bindings.get(self._need_key(need), [])
                if candidate.model_name in allowed
                and candidate.score >= self._binding_threshold()
            ]
            if not candidates:
                missing.append(need.phrase)
                continue
            best = max(
                candidates,
                key=lambda candidate: (candidate.score, candidate.member_name),
            )
            resolved[self._need_key(need)] = best.member_name
        return resolved, list(dict.fromkeys(missing))

    @staticmethod
    def _binding_scores(
        bindings: dict[str, list[NeedBindingCandidate]],
    ) -> dict[str, dict[str, float]]:
        return {
            key: {
                candidate.member_name: candidate.score
                for candidate in sorted(
                    candidates,
                    key=lambda candidate: (candidate.member_name, candidate.score),
                )
            }
            for key, candidates in bindings.items()
        }

    def _binding_candidates(
        self,
        bindings: dict[str, list[NeedBindingCandidate]],
        allowed_members: Iterable[str],
    ) -> dict[str, list[NeedCandidate]]:
        allowed = set(allowed_members)
        return {
            key: [
                NeedCandidate(member=candidate.member_name, score=candidate.score)
                for candidate in sorted(
                    candidates,
                    key=lambda candidate: (-candidate.score, candidate.member_name),
                )
                if candidate.member_name in allowed
            ][: self.member_top_k]
            for key, candidates in bindings.items()
        }

    def _exact_model_score(self, question: str, model: CatalogModel) -> float:
        values = [
            model.name,
            model.title,
            model.description or "",
            model.ai_context or "",
            *model.aliases,
            *model.use_cases,
        ]
        return max((_lexical_score(question, value) for value in values), default=0.0)

    def _scope_score(self, question: str, model: CatalogModel) -> float:
        return self._exact_model_score(question, model)

    def _rerank(
        self,
        question: str,
        candidates: list[_ScoredDocument],
        binding_coverage: dict[str, float],
        connectivity: dict[str, float] | None = None,
        scope_scores: dict[str, float] | None = None,
    ) -> dict[str, float]:
        scores: dict[str, float] = {}
        for item in candidates:
            name = str(item.document.metadata["model_name"])
            model = self.catalog.models[name]
            scores[name] = round(
                0.35 * item.score
                + 0.25 * binding_coverage.get(name, 0.0)
                + 0.15 * (scope_scores or {}).get(name, 0.0)
                + 0.10 * self._exact_model_score(question, model)
                + 0.10 * (connectivity or {}).get(name, 1.0)
                + 0.05 * model.business_priority,
                6,
            )
        return scores

    def _eligible_models(self, filters: dict[str, Any]) -> set[str]:
        allowed_keys = {"model_name", "model_type", "title", "business_domain"}
        invalid = set(filters) - allowed_keys
        if invalid:
            raise ValueError(f"不支持的 catalog metadata filter：{sorted(invalid)}")
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
        return names

    def _component_expansion(
        self,
        cube_pool: set[str],
        eligible_models: set[str],
    ) -> list[str]:
        cubes = [
            self.catalog.models[name]
            for name in eligible_models
            if self.catalog.models[name].model_type == "cube"
        ]
        components = {
            self.catalog.models[name].connected_component
            for name in cube_pool
            if self.catalog.models[name].connected_component is not None
        }
        domains = {
            self.catalog.models[name].business_domain
            for name in cube_pool
            if self.catalog.models[name].business_domain is not None
        }
        return sorted(
            model.name
            for model in cubes
            if not cube_pool
            or not components and not domains
            or model.connected_component in components
            or model.business_domain is not None
            and model.business_domain in domains
        )

    def _suggested_members(
        self,
        selected_models: list[str],
        needs: list[SemanticNeed],
        resolved_members: Iterable[str],
        member_hits: list[_ScoredDocument],
        question: str,
        question_vector: Sequence[float] | None,
        projection_mode: ProjectionMode,
        projection_policy: ProjectionPolicy,
    ) -> list[str]:
        suggested: list[str] = []

        def add(name: str) -> None:
            if name not in suggested and len(suggested) < min(8, self.context_member_limit):
                suggested.append(name)

        selected = set(selected_models)
        desired_types: set[str] | None
        if projection_mode == ProjectionMode.DETAIL:
            desired_types = {"dimension"}
        elif projection_mode == ProjectionMode.AGGREGATE:
            desired_types = {"measure"}
        elif projection_policy == ProjectionPolicy.SUMMARY:
            desired_types = {"measure"}
        elif projection_policy == ProjectionPolicy.MODEL_DEFAULT:
            desired_types = {"dimension"}
        else:
            desired_types = None

        def add_member(name: str) -> None:
            model_name = name.partition(".")[0]
            member = self.catalog.models[model_name].members[name]
            if desired_types is None or member.member_type in desired_types:
                add(name)

        for name in resolved_members:
            add_member(name)
        if projection_mode == ProjectionMode.DETAIL or (
            projection_mode == ProjectionMode.DEFAULT
            and projection_policy == ProjectionPolicy.MODEL_DEFAULT
        ):
            for model_name in selected_models:
                for member in self.catalog.models[model_name].members.values():
                    if member.primary_key and member.member_type == "dimension":
                        add(member.name)
        for item in [
            *member_hits,
            *self._rank(
                question,
                question_vector,
                parts={"member"},
                top_k=len(self._member_documents),
                eligible_models=selected,
            ),
        ]:
            if item.document.metadata["model_name"] not in selected:
                continue
            name = str(item.document.metadata["member_names"][0])
            member = self.catalog.models[name.partition(".")[0]].members[name]
            if desired_types is None or member.member_type in desired_types:
                add(name)
        return suggested

    def _allowed_members(
        self,
        candidate_models: list[str],
        required_members: Iterable[str],
        member_hits: list[_ScoredDocument],
        question: str,
        question_vector: Sequence[float] | None,
    ) -> list[str]:
        allowed: list[str] = []

        def add(name: str) -> None:
            if name not in allowed and len(allowed) < self.context_member_limit:
                allowed.append(name)

        for item in member_hits:
            if item.document.metadata["model_name"] in candidate_models:
                add(str(item.document.metadata["member_names"][0]))
        for name in required_members:
            add(name)
        for model_name in candidate_models:
            for member in self.catalog.models[model_name].members.values():
                if member.primary_key:
                    add(member.name)
        ranked = self._rank(
            question,
            question_vector,
            parts={"member"},
            top_k=len(self._member_documents),
            eligible_models=set(candidate_models),
        )
        for item in ranked:
            add(str(item.document.metadata["member_names"][0]))
        return allowed

    def _context(
        self,
        *,
        retrieval_intent: RetrievalIntent,
        candidate_models: list[str],
        allowed_members: list[str],
        binding_candidates: dict[str, list[NeedCandidate]],
        suggested_members: list[str],
        projection_mode: ProjectionMode,
        projection_policy: ProjectionPolicy,
        retrieval_level: int,
    ) -> SemanticContext:
        model_details = {
            name: {
                "name": name,
                "type": self.catalog.models[name].model_type,
                "title": self.catalog.models[name].title,
                "description": self.catalog.models[name].description,
                "aliases": self.catalog.models[name].aliases,
                "use_cases": self.catalog.models[name].use_cases,
            }
            for name in candidate_models
        }
        member_details = {
            name: _member_payload(
                self.catalog.models[name.partition(".")[0]].members[name]
            )
            for name in allowed_members
        }
        fixed_context = {
            name: self.catalog.models[name].ai_context or ""
            for name in candidate_models
            if self.catalog.models[name].model_type == "view"
            and self.catalog.models[name].ai_context
        }
        return SemanticContext(
            retrieval_intent=retrieval_intent,
            candidate_models=candidate_models,
            allowed_members=allowed_members,
            binding_candidates=binding_candidates,
            suggested_members=suggested_members,
            projection_mode=projection_mode,
            projection_policy=projection_policy,
            model_details=model_details,
            member_details=member_details,
            fixed_business_context=fixed_context,
            retrieval_level=retrieval_level,
        )

    def _catalog_subset(
        self,
        selected_models: list[str],
        allowed_members: list[str],
    ) -> SemanticCatalog:
        allowed = set(allowed_members)
        return SemanticCatalog(
            models={
                name: self.catalog.models[name].model_copy(
                    update={
                        "members": {
                            member_name: member
                            for member_name, member in self.catalog.models[name].members.items()
                            if member_name in allowed
                        }
                    }
                )
                for name in selected_models
            },
        )

    def _eligible_catalog_size(self, eligible_models: set[str]) -> int:
        payload = {
            name: {
                "model": _model_payload(self.catalog.models[name]),
                "members": [
                    _member_payload(member)
                    for member in self.catalog.models[name].members.values()
                ],
            }
            for name in sorted(eligible_models)
        }
        return len(json.dumps(payload, ensure_ascii=False, separators=(",", ":")))

    def _resolve_mode(
        self,
        requested_mode: SemanticCatalogMode,
        eligible_models: set[str],
    ) -> SemanticCatalogMode:
        if requested_mode != SemanticCatalogMode.AUTO:
            return requested_mode
        if self._eligible_catalog_size(eligible_models) <= self.auto_full_context_max_chars:
            return SemanticCatalogMode.FULL
        return SemanticCatalogMode.VECTOR

    @staticmethod
    def _binding_member_priority(
        bindings: dict[str, list[NeedBindingCandidate]],
    ) -> list[str]:
        ordered = [
            sorted(candidates, key=lambda candidate: (-candidate.score, candidate.member_name))
            for candidates in bindings.values()
        ]
        members: list[str] = []
        for index in range(max((len(candidates) for candidates in ordered), default=0)):
            for candidates in ordered:
                if index < len(candidates) and candidates[index].member_name not in members:
                    members.append(candidates[index].member_name)
        return members

    def _select_full_context(
        self,
        *,
        question: str,
        retrieval_intent: RetrievalIntent,
        eligible_models: set[str],
        projection_mode: ProjectionMode,
        projection_policy: ProjectionPolicy,
    ) -> SelectedSemanticCatalog:
        candidate_models = sorted(eligible_models)
        allowed_members = [
            member.name
            for model_name in candidate_models
            for member in self.catalog.models[model_name].members.values()
        ]
        context = self._context(
            retrieval_intent=retrieval_intent,
            candidate_models=candidate_models,
            allowed_members=allowed_members,
            binding_candidates={},
            suggested_members=self._suggested_members(
                candidate_models,
                retrieval_intent.needs,
                [],
                [],
                question,
                None,
                projection_mode,
                projection_policy,
            ),
            projection_mode=projection_mode,
            projection_policy=projection_policy,
            retrieval_level=0,
        )
        return SelectedSemanticCatalog(
            mode=SemanticCatalogMode.FULL,
            catalog=self._catalog_subset(candidate_models, allowed_members),
            selected_models=candidate_models,
            context=context,
            trace=RetrievalTrace(
                view_candidates=[
                    name
                    for name in candidate_models
                    if self.catalog.models[name].model_type == "view"
                ],
                cube_candidates=[
                    name
                    for name in candidate_models
                    if self.catalog.models[name].model_type == "cube"
                ],
            ),
            index_source="full_catalog",
        )

    async def _select_vector_context(
        self,
        question: str,
        *,
        retrieval_intent: RetrievalIntent,
        eligible_models: set[str],
        minimum_fallback_level: int,
        projection_mode: ProjectionMode,
        projection_policy: ProjectionPolicy,
    ) -> SelectedSemanticCatalog:
        warnings: list[str] = []
        index_source = "disabled"
        question_vector: Sequence[float] | None = None
        if self.embedding_client is not None:
            try:
                index_source = await self.prepare()
                question_vector = await self.embedding_client.embed_query(question)
            except Exception as exc:
                warnings.append(
                    "向量检索不可用，已使用分批词法目录分析。"
                    f"原因：{str(exc)[:200]}"
                )
        else:
            warnings.append("嵌入模型不可用，已使用分批词法目录分析。")
        view_ranked = self._rank_models(
            question,
            question_vector,
            model_type="view",
            top_k=self.view_top_k,
            eligible_models=eligible_models,
        )
        cube_ranked = self._rank_models(
            question,
            question_vector,
            model_type="cube",
            top_k=self.cube_top_k,
            eligible_models=eligible_models,
        )
        member_hits = self._rank(
            question,
            question_vector,
            parts={"member"},
            top_k=self.member_top_k,
            eligible_models=eligible_models,
        )
        view_candidates = [str(item.document.metadata["model_name"]) for item in view_ranked]
        cube_candidates = [str(item.document.metadata["model_name"]) for item in cube_ranked]
        member_parent_models = list(dict.fromkeys(
            str(item.document.metadata["model_name"]) for item in member_hits
        ))
        eligible_cubes = {
            name for name in eligible_models if self.catalog.models[name].model_type == "cube"
        }
        expanded_cube_models = list(dict.fromkeys([
            *cube_candidates,
            *(name for name in member_parent_models if name in eligible_cubes),
        ]))
        graph = SemanticJoinGraph.from_catalog(self.catalog)
        for left, right in combinations(expanded_cube_models, 2):
            path = graph.shortest_path(left, right)
            if path and set(path).issubset(eligible_models):
                for name in path:
                    if name not in expanded_cube_models:
                        expanded_cube_models.append(name)
        candidate_models = list(dict.fromkeys([
            *view_candidates,
            *expanded_cube_models,
            *member_parent_models,
        ]))
        scope_scores = {
            name: self._scope_score(question, self.catalog.models[name])
            for name in eligible_models
        }
        full_query_member_scores = {
            str(item.document.metadata["member_names"][0]): item.score
            for item in member_hits
        }
        trace = RetrievalTrace(
            view_candidates=view_candidates,
            cube_candidates=cube_candidates,
            member_hits=list(full_query_member_scores),
            scope_scores=scope_scores,
            fallback_level=min(minimum_fallback_level, 3),
        )
        needs = retrieval_intent.needs
        need_vectors: dict[str, Sequence[float] | None] = {
            self._need_key(need): None for need in needs
        }
        if self.embedding_client is not None and question_vector is not None:
            semaphore = asyncio.Semaphore(self.retrieval_concurrency)

            async def embed_need(need: SemanticNeed) -> tuple[str, Sequence[float]]:
                async with semaphore:
                    return (
                        self._need_key(need),
                        await self.embedding_client.embed_query(
                            self._need_binding_text(need, question)
                        ),
                    )

            need_vectors.update(await asyncio.gather(*(embed_need(need) for need in needs)))
        bindings = self._need_bindings(
            needs,
            need_vectors,
            candidate_models,
            full_query_member_scores,
            scope_scores,
        )
        allowed_members = self._allowed_members(
            candidate_models,
            self._binding_member_priority(bindings),
            member_hits,
            question,
            question_vector,
        )
        binding_candidates = self._binding_candidates(bindings, allowed_members)
        suggested_members = self._suggested_members(
            candidate_models,
            needs,
            self._binding_member_priority(bindings),
            member_hits,
            question,
            question_vector,
            projection_mode,
            projection_policy,
        )
        trace.binding_scores = self._binding_scores(bindings)
        trace.binding_candidates = binding_candidates
        trace.suggested_members = suggested_members
        context = self._context(
            retrieval_intent=retrieval_intent,
            candidate_models=candidate_models,
            allowed_members=allowed_members,
            binding_candidates=binding_candidates,
            suggested_members=suggested_members,
            projection_mode=projection_mode,
            projection_policy=projection_policy,
            retrieval_level=trace.fallback_level,
        )
        return SelectedSemanticCatalog(
            mode=SemanticCatalogMode.VECTOR,
            catalog=self._catalog_subset(candidate_models, allowed_members),
            selected_models=candidate_models,
            context=context,
            trace=trace,
            warnings=warnings,
            index_source=index_source,
        )

    async def select(
        self,
        question: str,
        *,
        retrieval_intent: RetrievalIntent,
        mode: SemanticCatalogMode | None = None,
        metadata_filters: dict[str, Any] | None = None,
        minimum_fallback_level: int = 0,
        projection_mode: ProjectionMode = ProjectionMode.DEFAULT,
        projection_policy: ProjectionPolicy = ProjectionPolicy.MODEL_DEFAULT,
    ) -> SelectedSemanticCatalog:
        requested_mode = mode or self.mode
        eligible_models = self._eligible_models(dict(metadata_filters or {}))
        if not eligible_models:
            gap = SemanticModelGap(
                message="catalog metadata filter 未匹配到任何受治理的公开模型",
                missing_concepts=[need.phrase for need in retrieval_intent.needs],
            )
            return SelectedSemanticCatalog(
                mode=requested_mode,
                catalog=SemanticCatalog(),
                selected_models=[],
                context=None,
                trace=RetrievalTrace(fallback_level=3),
                gap=gap,
            )
        effective_mode = self._resolve_mode(requested_mode, eligible_models)
        if effective_mode == SemanticCatalogMode.FULL:
            return self._select_full_context(
                question=question,
                retrieval_intent=retrieval_intent,
                eligible_models=eligible_models,
                projection_mode=projection_mode,
                projection_policy=projection_policy,
            )
        return await self._select_vector_context(
            question,
            retrieval_intent=retrieval_intent,
            eligible_models=eligible_models,
            minimum_fallback_level=minimum_fallback_level,
            projection_mode=projection_mode,
            projection_policy=projection_policy,
        )
