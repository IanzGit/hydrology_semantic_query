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
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

from .models import (
    CatalogMember,
    CatalogModel,
    MemberRequirement,
    QueryMode,
    RetrievalTrace,
    SemanticCatalog,
    SemanticCatalogMode,
    SemanticContext,
    SemanticIntent,
    SemanticModelGap,
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
        mode: SemanticCatalogMode = SemanticCatalogMode.VECTOR,
        embedding_batch_size: int = 32,
        embedding_concurrency: int = 3,
        retrieval_concurrency: int = 3,
        context_member_limit: int = 12,
        catalog_batch_size: int = 4,
        max_cube_models: int = 4,
        member_match_threshold: float = 0.55,
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
    def _requirement_key(requirement: MemberRequirement) -> str:
        return f"{requirement.role}:{requirement.phrase}"

    @staticmethod
    def _member_matches_requirement(
        member: CatalogMember,
        requirement: MemberRequirement,
    ) -> bool:
        if requirement.role == "aggregate":
            return member.member_type == "measure"
        if requirement.role in {"project", "group"}:
            return member.member_type == "dimension"
        if requirement.role == "filter":
            return member.member_type in {"measure", "dimension"}
        if requirement.role == "order":
            return member.member_type in {"measure", "dimension"}
        return False

    def _member_score(
        self,
        concept: str,
        vector: Sequence[float] | None,
        member: CatalogMember,
    ) -> float:
        document = self._member_documents[member.name]
        vector_score = (
            _cosine(vector, document.vector)
            if vector is not None and document.vector is not None
            else 0.0
        )
        lexical = self._member_lexical_score(concept, member)
        return max(vector_score, lexical)

    @staticmethod
    def _member_lexical_score(concept: str, member: CatalogMember) -> float:
        names = [member.name.partition(".")[2], member.title, *member.aliases]
        return max((_lexical_score(concept, value) for value in names), default=0.0)

    def _requirement_matches(
        self,
        requirements: list[MemberRequirement],
        requirement_vectors: dict[str, Sequence[float] | None],
        *,
        model_type: str,
        eligible_models: set[str],
    ) -> dict[str, dict[str, tuple[float, str]]]:
        matches: dict[str, dict[str, tuple[float, str]]] = {}
        for requirement in requirements:
            key = self._requirement_key(requirement)
            by_model: dict[str, tuple[float, str]] = {}
            for model_name in sorted(eligible_models):
                model = self.catalog.models[model_name]
                if model.model_type != model_type:
                    continue
                ranked = [
                    (
                        self._member_score(
                            requirement.phrase,
                            requirement_vectors.get(key),
                            member,
                        ),
                        member.name,
                    )
                    for member in model.members.values()
                    if self._member_matches_requirement(member, requirement)
                ]
                if ranked:
                    by_model[model_name] = max(ranked)
            matches[key] = by_model
        return matches

    def _requirement_coverage(
        self,
        view_candidates: list[str],
        requirements: list[MemberRequirement],
        matches: dict[str, dict[str, tuple[float, str]]],
        global_matches: dict[str, dict[str, tuple[float, str]]] | None = None,
    ) -> dict[str, float]:
        coverage: dict[str, float] = {}
        hard_requirements = [item for item in requirements if item.required]
        if not hard_requirements:
            return {name: 1.0 for name in view_candidates}
        for model_name in view_candidates:
            covered = 0
            for requirement in hard_requirements:
                key = self._requirement_key(requirement)
                by_model = matches.get(key, {})
                best_global = max(
                    (
                        score
                        for score, _ in (global_matches or matches)
                        .get(key, {})
                        .values()
                    ),
                    default=0.0,
                )
                score = by_model.get(model_name, (0.0, ""))[0]
                if score >= max(self.member_match_threshold, best_global * 0.85):
                    covered += 1
            coverage[model_name] = covered / len(hard_requirements)
        return coverage

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

    def _scope_score(
        self,
        question: str,
        model: CatalogModel,
        subjects: list[str],
    ) -> float:
        values = [
            model.name,
            model.title,
            model.description or "",
            model.ai_context or "",
            *model.aliases,
            *model.use_cases,
        ]
        subject_score = max(
            (
                _lexical_score(subject, value)
                for subject in subjects
                for value in values
            ),
            default=0.0,
        )
        return max(subject_score, self._exact_model_score(question, model))

    def _rerank(
        self,
        question: str,
        candidates: list[_ScoredDocument],
        coverage: dict[str, float],
        connectivity: dict[str, float] | None = None,
        scope_scores: dict[str, float] | None = None,
    ) -> dict[str, float]:
        scores: dict[str, float] = {}
        for item in candidates:
            name = str(item.document.metadata["model_name"])
            model = self.catalog.models[name]
            scores[name] = round(
                0.35 * item.score
                + 0.25 * coverage.get(name, 0.0)
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
            matched = True
            for key, expected in filters.items():
                actual = values[key]
                if isinstance(expected, (list, tuple, set)):
                    matched = actual in expected
                else:
                    matched = actual == expected
                if not matched:
                    break
            if matched:
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
            if name in self.catalog.models
            and self.catalog.models[name].connected_component is not None
        }
        domains = {
            self.catalog.models[name].business_domain
            for name in cube_pool
            if name in self.catalog.models
            and self.catalog.models[name].business_domain is not None
        }
        expanded = [
            model.name
            for model in cubes
            if not cube_pool
            or not components and not domains
            or model.connected_component in components
            or model.business_domain is not None and model.business_domain in domains
        ]
        return sorted(expanded)

    def _best_cube_members(
        self,
        requirements: list[MemberRequirement],
        matches: dict[str, dict[str, tuple[float, str]]],
        cube_pool: Iterable[str],
        model_scores: dict[str, float],
    ) -> tuple[dict[str, str], list[str]]:
        allowed = set(cube_pool)
        selected: dict[str, str] = {}
        missing: list[str] = []
        for requirement in requirements:
            key = self._requirement_key(requirement)
            ranked = [
                (score, member_name, model_name)
                for model_name, (score, member_name) in matches.get(key, {}).items()
                if model_name in allowed and score >= self.member_match_threshold
            ]
            if not ranked:
                if requirement.required:
                    missing.append(requirement.phrase)
                continue
            _, member_name, _ = max(
                ranked,
                key=lambda item: (
                    0.7 * item[0]
                    + 0.2
                    * self._member_lexical_score(
                        requirement.phrase,
                        self.catalog.models[item[2]].members[item[1]],
                    )
                    + 0.1 * model_scores.get(item[2], 0.0),
                    item[0],
                    item[1],
                ),
            )
            selected[key] = member_name
        return selected, missing

    @staticmethod
    def _qualifier_terms(value: str) -> tuple[str, ...]:
        terms = {
            "有效": ("有效", "启用", "活动", "未停用"),
            "有效状态": ("有效", "启用", "活动", "未停用"),
            "启用": ("启用", "有效", "活动"),
            "启用状态": ("启用", "有效", "活动"),
            "活动中": ("活动中", "活动", "启用", "有效"),
        }
        return tuple(dict.fromkeys((value, *terms.get(value, ()))))

    def _absorb_legacy_scope_requirements(
        self,
        requirements: list[MemberRequirement],
        matches: dict[str, dict[str, tuple[float, str]]],
        scope_scores: dict[str, float],
    ) -> tuple[list[MemberRequirement], list[str]]:
        retained: list[MemberRequirement] = []
        absorbed: list[str] = []
        for requirement in requirements:
            key = self._requirement_key(requirement)
            best_member = max(
                (score for score, _ in matches.get(key, {}).values()),
                default=0.0,
            )
            best_scope = max(scope_scores.values(), default=0.0)
            if (
                requirement.source == "legacy"
                and requirement.role == "project"
                and best_scope >= 0.7
                and best_member < self.member_match_threshold
            ):
                absorbed.append(requirement.phrase)
                continue
            retained.append(requirement)
        return retained, absorbed

    def _resolve_qualifiers(
        self,
        selected_models: list[str],
        intent: SemanticIntent,
    ) -> tuple[dict[str, str], list[str]]:
        resolved: dict[str, str] = {}
        missing: list[str] = []
        for qualifier in intent.qualifiers:
            fixed_matches: list[tuple[float, str]] = []
            member_matches: list[tuple[float, str]] = []
            for model_name in selected_models:
                model = self.catalog.models[model_name]
                context = " ".join(
                    value
                    for value in (
                        model.ai_context,
                        model.description,
                        *model.use_cases,
                    )
                    if value
                )
                terms = self._qualifier_terms(qualifier)
                fixed_score = max(
                    (
                        1.0 if term in context else _lexical_score(term, context)
                        for term in terms
                    ),
                    default=0.0,
                )
                if fixed_score >= 0.55 and model.model_type == "view":
                    fixed_matches.append((fixed_score, model_name))
                for member in model.members.values():
                    if member.member_type not in {"measure", "dimension"}:
                        continue
                    names = [
                        member.name.partition(".")[2],
                        member.title,
                        *member.aliases,
                    ]
                    score = max(
                        (
                            _lexical_score(term, name)
                            for term in terms
                            for name in names
                        ),
                        default=0.0,
                    )
                    short_name = member.name.partition(".")[2].lower()
                    if any(
                        token in short_name
                        for token in ("enabled", "active", "valid", "is_enabled")
                    ):
                        score = max(score, 0.7)
                    if score >= self.member_match_threshold:
                        member_matches.append((score, member.name))
            if fixed_matches:
                resolved[qualifier] = "fixed_business_context"
            elif member_matches:
                _, member_name = max(member_matches)
                resolved[qualifier] = member_name
            else:
                missing.append(qualifier)
        return resolved, missing

    def _resolve_temporal(
        self,
        selected_models: list[str],
        intent: SemanticIntent,
    ) -> tuple[dict[str, Any], list[str], list[str]]:
        temporal = intent.temporal
        if temporal is None:
            return {}, [], []
        resolution: dict[str, Any] = {"operator": temporal.operator}
        suggested: list[str] = []
        missing: list[str] = []
        time_members = [
            member
            for model_name in selected_models
            for member in self.catalog.models[model_name].members.values()
            if member.member_type == "dimension" and member.data_type == "time"
        ]
        if temporal.operator == "current":
            fixed = any(
                self.catalog.models[name].model_type == "view"
                and any(
                    term in " ".join(
                        value
                        for value in (
                            self.catalog.models[name].ai_context,
                            self.catalog.models[name].description,
                        )
                        if value
                    )
                    for term in ("当前", "启用", "状态")
                )
                for name in selected_models
            )
            state_member = next(
                (
                    member
                    for model_name in selected_models
                    for member in self.catalog.models[model_name].members.values()
                    if member.member_type == "dimension"
                    and any(
                        term in " ".join(
                            [
                                member.name.partition(".")[2],
                                member.title,
                                *member.aliases,
                            ]
                        )
                        for term in ("当前", "状态", "active", "enabled")
                    )
                ),
                None,
            )
            if fixed:
                resolution["resolution"] = "fixed_business_context"
            elif state_member is not None:
                resolution.update({
                    "resolution": "current_state_member",
                    "member": state_member.name,
                })
                suggested.append(state_member.name)
            else:
                resolution["resolution"] = "deferred"
            return resolution, suggested, missing
        if not time_members:
            missing.append(temporal.raw_phrase or temporal.operator)
            resolution["resolution"] = "unavailable"
            return resolution, suggested, missing
        ranked = sorted(
            time_members,
            key=lambda member: (
                self._member_lexical_score(
                    temporal.field_hint or "开始时间",
                    member,
                ),
                member.name,
            ),
            reverse=True,
        )
        selected = ranked[0]
        resolution.update({
            "resolution": "time_dimension",
            "time_member": selected.name,
            "direction": "desc" if temporal.operator == "latest" else None,
            "limit": 1 if temporal.operator == "latest" else None,
        })
        suggested.append(selected.name)
        return resolution, suggested, missing

    def _suggested_members(
        self,
        selected_models: list[str],
        requirements: list[MemberRequirement],
        resolved_members: Iterable[str],
        member_hits: list[_ScoredDocument],
        question: str,
        question_vector: Sequence[float] | None,
        result_shape: str,
        operator_members: Iterable[str],
    ) -> list[str]:
        suggested: list[str] = []

        def add(name: str) -> None:
            if name in suggested or len(suggested) >= min(8, self.context_member_limit):
                return
            suggested.append(name)

        for name in resolved_members:
            add(name)
        for name in operator_members:
            add(name)
        selected = set(selected_models)
        if not requirements:
            member_type = "measure" if result_shape == "aggregate" else "dimension"
            for model_name in selected_models:
                for member in self.catalog.models[model_name].members.values():
                    if member.member_type == member_type:
                        add(member.name)
            return suggested
        if result_shape == "detail":
            for model_name in selected_models:
                for member in self.catalog.models[model_name].members.values():
                    if member.primary_key and member.member_type == "dimension":
                        add(member.name)
        for item in member_hits:
            if item.document.metadata.get("model_name") in selected:
                name = str(item.document.metadata["member_names"][0])
                member = self.catalog.models[name.partition(".")[0]].members[name]
                if result_shape == "detail" and member.member_type != "dimension":
                    continue
                if result_shape == "aggregate" and member.member_type != "measure":
                    continue
                add(name)
        ranked = self._rank(
            question,
            question_vector,
            parts={"member"},
            top_k=len(self._member_documents),
            eligible_models=selected,
        )
        for item in ranked:
            name = str(item.document.metadata["member_names"][0])
            member = self.catalog.models[name.partition(".")[0]].members[name]
            if result_shape == "detail" and member.member_type != "dimension":
                continue
            if result_shape == "aggregate" and member.member_type != "measure":
                continue
            add(name)
        for model_name in selected_models:
            for member in self.catalog.models[model_name].members.values():
                if result_shape == "detail" and member.member_type == "dimension":
                    add(member.name)
                elif result_shape == "aggregate" and member.member_type == "measure":
                    add(member.name)
        return suggested

    def _allowed_members(
        self,
        selected_models: list[str],
        required_members: Iterable[str],
        member_hits: list[_ScoredDocument],
        question: str,
        question_vector: Sequence[float] | None,
    ) -> list[str]:
        allowed: list[str] = []

        def add(name: str) -> None:
            if name not in allowed and len(allowed) < self.context_member_limit:
                allowed.append(name)

        for name in required_members:
            add(name)
        for model_name in selected_models:
            for member in self.catalog.models[model_name].members.values():
                if member.primary_key:
                    add(member.name)
        for item in member_hits:
            if item.document.metadata["model_name"] in selected_models:
                add(str(item.document.metadata["member_names"][0]))
        ranked = self._rank(
            question,
            question_vector,
            parts={"member"},
            top_k=len(self._member_documents),
            eligible_models=set(selected_models),
        )
        for item in ranked:
            add(str(item.document.metadata["member_names"][0]))
        return allowed

    def _context(
        self,
        *,
        intent: SemanticIntent,
        query_mode: QueryMode,
        selected_models: list[str],
        allowed_members: list[str],
        suggested_members: list[str],
        projection_policy: str,
        fallback_anchor: list[str],
        operator_resolution: dict[str, Any],
        resolved_qualifiers: dict[str, str],
        join_paths: list[list[str]],
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
            for name in selected_models
        }
        member_details: dict[str, dict[str, Any]] = {}
        for name in allowed_members:
            model_name = name.partition(".")[0]
            member = self.catalog.models[model_name].members[name]
            member_details[name] = _member_payload(member)
        fixed_context = {
            name: self.catalog.models[name].ai_context or ""
            for name in selected_models
            if self.catalog.models[name].model_type == "view"
            and self.catalog.models[name].ai_context
        }
        return SemanticContext(
            intent=intent,
            query_mode=query_mode,
            models=selected_models,
            allowed_members=allowed_members,
            suggested_members=suggested_members,
            projection_policy=projection_policy,
            fallback_anchor=fallback_anchor,
            operator_resolution=operator_resolution,
            resolved_qualifiers=resolved_qualifiers,
            model_details=model_details,
            member_details=member_details,
            join_paths=join_paths,
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

    async def select(
        self,
        question: str,
        *,
        intent: SemanticIntent,
        mode: SemanticCatalogMode | None = None,
        metadata_filters: dict[str, Any] | None = None,
        minimum_fallback_level: int = 0,
    ) -> SelectedSemanticCatalog:
        warnings: list[str] = []
        index_source = "disabled"
        question_vector: Sequence[float] | None = None
        effective_mode = mode or self.mode
        if self.embedding_client is not None and effective_mode != SemanticCatalogMode.FULL:
            try:
                index_source = await self.prepare()
                question_vector = await self.embedding_client.embed_query(question)
            except Exception as exc:
                warnings.append(
                    "向量检索不可用，已使用分批词法目录分析。"
                    f"原因：{str(exc)[:200]}"
                )
        else:
            warnings.append("嵌入模型不可用或请求完整模式，已使用分批词法目录分析。")
        eligible_models = self._eligible_models(dict(metadata_filters or {}))
        if not eligible_models:
            gap = SemanticModelGap(
                message="catalog metadata filter 未匹配到任何受治理的公开模型",
                missing_concepts=[
                    requirement.phrase for requirement in intent.hard_requirements()
                ],
            )
            return SelectedSemanticCatalog(
                mode=effective_mode,
                catalog=SemanticCatalog(),
                selected_models=[],
                context=None,
                trace=RetrievalTrace(fallback_level=3),
                gap=gap,
                warnings=warnings,
                index_source=index_source,
            )
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
        member_parent_models = list(
            dict.fromkeys(str(item.document.metadata["model_name"]) for item in member_hits)
        )
        scope_scores = {
            name: self._scope_score(
                question,
                self.catalog.models[name],
                intent.scope_terms(),
            )
            for name in eligible_models
        }
        trace = RetrievalTrace(
            view_candidates=view_candidates,
            cube_candidates=cube_candidates,
            member_hits=[str(item.document.metadata["member_names"][0]) for item in member_hits],
            member_parent_models=member_parent_models,
            scope_scores=scope_scores,
        )
        requirements = intent.member_requirements
        requirement_vectors: dict[str, Sequence[float] | None] = {
            self._requirement_key(requirement): None
            for requirement in requirements
        }
        if self.embedding_client is not None and question_vector is not None:
            semaphore = asyncio.Semaphore(self.retrieval_concurrency)

            async def embed_requirement(
                requirement: MemberRequirement,
            ) -> tuple[str, Sequence[float]]:
                async with semaphore:
                    return (
                        self._requirement_key(requirement),
                        await self.embedding_client.embed_query(requirement.phrase),
                    )

            requirement_vectors.update(
                await asyncio.gather(
                    *(embed_requirement(requirement) for requirement in requirements)
                )
            )
        view_matches = self._requirement_matches(
            requirements,
            requirement_vectors,
            model_type="view",
            eligible_models=eligible_models,
        )
        cube_matches = self._requirement_matches(
            requirements,
            requirement_vectors,
            model_type="cube",
            eligible_models=eligible_models,
        )
        requirements, absorbed = self._absorb_legacy_scope_requirements(
            requirements,
            {key: {**view_matches.get(key, {}), **cube_matches.get(key, {})}
             for key in requirement_vectors},
            scope_scores,
        )
        if absorbed:
            intent = intent.model_copy(update={"member_requirements": requirements})
            requirement_vectors = {
                self._requirement_key(requirement): requirement_vectors.get(
                    self._requirement_key(requirement)
                )
                for requirement in requirements
            }
            view_matches = self._requirement_matches(
                requirements,
                requirement_vectors,
                model_type="view",
                eligible_models=eligible_models,
            )
            cube_matches = self._requirement_matches(
                requirements,
                requirement_vectors,
                model_type="cube",
                eligible_models=eligible_models,
            )
        global_matches = {
            key: {
                **view_matches.get(key, {}),
                **cube_matches.get(key, {}),
            }
            for key in {
                self._requirement_key(requirement) for requirement in requirements
            }
        }
        trace.view_coverage = self._requirement_coverage(
            view_candidates,
            requirements,
            view_matches,
            global_matches,
        )
        trace.member_coverage.update(trace.view_coverage)
        trace.rerank_scores = self._rerank(
            question,
            view_ranked,
            trace.view_coverage,
            scope_scores=scope_scores,
        )
        view_resolutions: dict[str, tuple[dict[str, str], dict[str, Any], list[str]]] = {}
        complete_views: list[str] = []
        for name in view_candidates:
            if trace.view_coverage.get(name) != 1.0:
                continue
            qualifier_resolution, qualifier_missing = self._resolve_qualifiers(
                [name],
                intent,
            )
            operator_resolution, _, operator_missing = self._resolve_temporal(
                [name],
                intent,
            )
            if qualifier_missing or operator_missing:
                continue
            view_resolutions[name] = (
                qualifier_resolution,
                operator_resolution,
                [],
            )
            complete_views.append(name)
        if complete_views and minimum_fallback_level == 0:
            selected_model = max(
                complete_views,
                key=lambda name: (trace.rerank_scores.get(name, 0.0), name),
            )
            qualifier_resolution, operator_resolution, _ = view_resolutions[selected_model]
            required_members = [
                view_matches[self._requirement_key(requirement)][selected_model][1]
                for requirement in requirements
                if selected_model
                in view_matches.get(self._requirement_key(requirement), {})
            ]
            _, operator_suggestions, _ = self._resolve_temporal(
                [selected_model],
                intent,
            )
            suggested_members = self._suggested_members(
                [selected_model],
                requirements,
                required_members,
                member_hits,
                question,
                question_vector,
                intent.result_shape,
                operator_suggestions,
            )
            qualifier_members = [
                value
                for value in qualifier_resolution.values()
                if value != "fixed_business_context"
            ]
            allowed_members = self._allowed_members(
                [selected_model],
                [*required_members, *suggested_members, *qualifier_members],
                member_hits,
                question,
                question_vector,
            )
            trace.resolved_requirements = {
                requirement.phrase: view_matches[
                    self._requirement_key(requirement)
                ][selected_model][1]
                for requirement in requirements
                if selected_model
                in view_matches.get(self._requirement_key(requirement), {})
            }
            trace.resolved_qualifiers = qualifier_resolution
            trace.operator_resolution = operator_resolution
            trace.suggested_members = suggested_members
            context = self._context(
                intent=intent,
                query_mode=QueryMode.VIEW,
                selected_models=[selected_model],
                allowed_members=allowed_members,
                suggested_members=suggested_members,
                projection_policy=(
                    "explicit" if requirements else "model_default"
                ),
                fallback_anchor=[],
                operator_resolution=operator_resolution,
                resolved_qualifiers=qualifier_resolution,
                join_paths=[],
                retrieval_level=0,
            )
            return SelectedSemanticCatalog(
                mode=effective_mode,
                catalog=self._catalog_subset([selected_model], allowed_members),
                selected_models=[selected_model],
                context=context,
                trace=trace,
                warnings=warnings,
                index_source=index_source,
            )
        cube_pool = {
            *cube_candidates,
            *(
                name
                for name in member_parent_models
                if name in self.catalog.models
                and self.catalog.models[name].model_type == "cube"
            ),
        }
        eligible_cubes = {
            name
            for name in eligible_models
            if self.catalog.models[name].model_type == "cube"
        }
        trace.cube_coverage = self._requirement_coverage(
            sorted(eligible_cubes),
            requirements,
            cube_matches,
        )
        trace.member_coverage.update(trace.cube_coverage)
        graph = SemanticJoinGraph.from_catalog(self.catalog)
        trace.cube_connectivity = {
            name: float(
                not eligible_cubes - {name}
                or any(
                    graph.shortest_path(name, other) is not None
                    for other in eligible_cubes - {name}
                )
            )
            for name in eligible_cubes
        }
        all_cube_ranked = self._rank_models(
            question,
            question_vector,
            model_type="cube",
            top_k=len(eligible_cubes),
            eligible_models=eligible_models,
        )
        cube_rerank_scores = self._rerank(
            question,
            all_cube_ranked,
            trace.cube_coverage,
            trace.cube_connectivity,
            scope_scores,
        )
        trace.rerank_scores.update(cube_rerank_scores)
        best_members, missing = self._best_cube_members(
            requirements,
            cube_matches,
            cube_pool,
            cube_rerank_scores,
        )
        fallback_level = 1
        if (missing and requirements) or minimum_fallback_level >= 2:
            fallback_level = 2
            expanded = self._component_expansion(cube_pool, eligible_models)
            batched_best: dict[str, str] = {}
            if requirements:
                for index in range(0, len(expanded), self.catalog_batch_size):
                    batch = expanded[index : index + self.catalog_batch_size]
                    batch_best, _ = self._best_cube_members(
                        requirements,
                        cube_matches,
                        batch,
                        cube_rerank_scores,
                    )
                    for key, member_name in batch_best.items():
                        current = batched_best.get(key)
                        model_name = member_name.partition(".")[0]
                        score = cube_matches[key][model_name][0]
                        if current is None:
                            batched_best[key] = member_name
                            continue
                        current_model = current.partition(".")[0]
                        if score > cube_matches[key][current_model][0]:
                            batched_best[key] = member_name
                    trace.catalog_batches_analyzed += 1
            best_members = batched_best
            missing = [
                requirement.phrase
                for requirement in requirements
                if self._requirement_key(requirement) not in batched_best
            ]
            cube_pool.update(expanded)
        if missing:
            trace.missing_requirements = missing
            trace.fallback_level = 3
            gap = SemanticModelGap(
                message="公开 Cube 语义模型中缺少完成查询所需的成员",
                missing_concepts=missing,
            )
            return SelectedSemanticCatalog(
                mode=effective_mode,
                catalog=SemanticCatalog(),
                selected_models=[],
                context=None,
                trace=trace,
                gap=gap,
                warnings=warnings,
                index_source=index_source,
            )
        fallback_anchor: list[str] = []
        if requirements:
            required_models = list(
                dict.fromkeys(name.partition(".")[0] for name in best_members.values())
            )
        else:
            anchor_candidates = [
                *cube_candidates,
                *[
                    str(item.document.metadata["model_name"])
                    for item in all_cube_ranked
                ],
                *sorted(
                    eligible_cubes,
                    key=lambda name: (scope_scores.get(name, 0.0), name),
                    reverse=True,
                ),
            ]
            anchor = next(
                (name for name in anchor_candidates if name in eligible_cubes),
                None,
            )
            if anchor is None:
                trace.fallback_level = 3
                gap = SemanticModelGap(
                    message="scope-only 查询没有可用的 Cube fallback anchor",
                    missing_concepts=intent.scope_terms(),
                )
                return SelectedSemanticCatalog(
                    mode=effective_mode,
                    catalog=SemanticCatalog(),
                    selected_models=[],
                    context=None,
                    trace=trace,
                    gap=gap,
                    warnings=warnings,
                    index_source=index_source,
                )
            fallback_anchor = [anchor]
            required_models = [anchor]
        ambiguous_pairs = graph.ambiguous_pairs(required_models)
        if ambiguous_pairs:
            trace.fallback_level = 3
            gap = SemanticModelGap(
                message="所需 Cube 之间存在多条最短 Join Path，无法无歧义执行",
                ambiguous_model_pairs=ambiguous_pairs,
            )
            return SelectedSemanticCatalog(
                mode=effective_mode,
                catalog=SemanticCatalog(),
                selected_models=[],
                context=None,
                trace=trace,
                gap=gap,
                warnings=warnings,
                index_source=index_source,
            )
        subgraph = graph.minimal_subgraph(required_models)
        if subgraph is None:
            trace.fallback_level = 3
            gap = SemanticModelGap(
                message="所需 Cube 位于断连的 Join Graph 中",
                disconnected_models=required_models,
            )
            return SelectedSemanticCatalog(
                mode=effective_mode,
                catalog=SemanticCatalog(),
                selected_models=[],
                context=None,
                trace=trace,
                gap=gap,
                warnings=warnings,
                index_source=index_source,
            )
        selected_models, join_paths = subgraph
        if len(selected_models) > self.max_cube_models:
            trace.fallback_level = 3
            gap = SemanticModelGap(
                message=f"最小 Join 子图包含 {len(selected_models)} 个 Cube，超过上限 {self.max_cube_models}",
                disconnected_models=selected_models,
            )
            return SelectedSemanticCatalog(
                mode=effective_mode,
                catalog=SemanticCatalog(),
                selected_models=[],
                context=None,
                trace=trace,
                gap=gap,
                warnings=warnings,
                index_source=index_source,
            )
        qualifier_resolution, qualifier_missing = self._resolve_qualifiers(
            selected_models,
            intent,
        )
        operator_resolution, operator_suggestions, operator_missing = (
            self._resolve_temporal(selected_models, intent)
        )
        capability_missing = [*qualifier_missing, *operator_missing]
        if capability_missing:
            trace.fallback_level = 3
            trace.missing_requirements = capability_missing
            gap = SemanticModelGap(
                message="公开语义模型缺少完成查询所需的业务限定或时间能力",
                missing_concepts=capability_missing,
            )
            return SelectedSemanticCatalog(
                mode=effective_mode,
                catalog=SemanticCatalog(),
                selected_models=[],
                context=None,
                trace=trace,
                gap=gap,
                warnings=warnings,
                index_source=index_source,
            )
        trace.fallback_level = fallback_level
        trace.join_paths = join_paths
        trace.fallback_anchor = fallback_anchor
        trace.resolved_requirements = {
            requirement.phrase: best_members[self._requirement_key(requirement)]
            for requirement in requirements
            if self._requirement_key(requirement) in best_members
        }
        trace.missing_requirements = missing
        trace.resolved_qualifiers = qualifier_resolution
        trace.operator_resolution = operator_resolution
        qualifier_members = [
            value
            for value in qualifier_resolution.values()
            if value != "fixed_business_context"
        ]
        suggested_members = self._suggested_members(
            selected_models,
            requirements,
            best_members.values(),
            member_hits,
            question,
            question_vector,
            intent.result_shape,
            [*operator_suggestions, *qualifier_members],
        )
        trace.suggested_members = suggested_members
        allowed_members = self._allowed_members(
            selected_models,
            [*best_members.values(), *suggested_members, *qualifier_members],
            member_hits,
            question,
            question_vector,
        )
        projection_policy = "explicit" if requirements else "model_default"
        context = self._context(
            intent=intent,
            query_mode=QueryMode.CUBE,
            selected_models=selected_models,
            allowed_members=allowed_members,
            suggested_members=suggested_members,
            projection_policy=projection_policy,
            fallback_anchor=fallback_anchor,
            operator_resolution=operator_resolution,
            resolved_qualifiers=qualifier_resolution,
            join_paths=join_paths,
            retrieval_level=fallback_level,
        )
        return SelectedSemanticCatalog(
            mode=effective_mode,
            catalog=self._catalog_subset(selected_models, allowed_members),
            selected_models=selected_models,
            context=context,
            trace=trace,
            warnings=warnings,
            index_source=index_source,
        )
