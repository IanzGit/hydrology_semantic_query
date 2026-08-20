from __future__ import annotations

import json
from collections import deque
from itertools import combinations

from .models import SemanticCatalog, SemanticContext


class SemanticJoinGraph:
    def __init__(self, adjacency: dict[str, set[str]]) -> None:
        self.adjacency = adjacency

    @classmethod
    def from_catalog(cls, catalog: SemanticCatalog) -> SemanticJoinGraph:
        cube_names = {
            name for name, model in catalog.models.items() if model.model_type == "cube"
        }
        adjacency = {name: set() for name in cube_names}
        for name in cube_names:
            for target in catalog.models[name].join_edges:
                if target not in cube_names:
                    continue
                adjacency[name].add(target)
                adjacency[target].add(name)
        return cls(adjacency)

    def shortest_path(self, start: str, end: str) -> list[str] | None:
        paths = self.shortest_paths(start, end, limit=1)
        return paths[0] if paths else None

    def shortest_paths(
        self,
        start: str,
        end: str,
        *,
        limit: int = 2,
    ) -> list[list[str]]:
        if start not in self.adjacency or end not in self.adjacency:
            return []
        if start == end:
            return [[start]]
        queue = deque([(start, [start])])
        shortest_length: int | None = None
        paths: list[list[str]] = []
        while queue:
            node, path = queue.popleft()
            if shortest_length is not None and len(path) >= shortest_length:
                continue
            for neighbor in sorted(self.adjacency[node]):
                if neighbor in path:
                    continue
                next_path = [*path, neighbor]
                if neighbor == end:
                    shortest_length = len(next_path)
                    paths.append(next_path)
                    if len(paths) == limit:
                        return paths
                    continue
                queue.append((neighbor, next_path))
        return paths

    def ambiguous_pairs(self, models: list[str]) -> list[list[str]]:
        return [
            [left, right]
            for left, right in combinations(dict.fromkeys(models), 2)
            if len(self.shortest_paths(left, right, limit=2)) > 1
        ]

    def minimal_subgraph(
        self,
        required_models: list[str],
    ) -> tuple[list[str], list[list[str]]] | None:
        required = list(dict.fromkeys(required_models))
        if not required:
            return [], []
        if len(required) == 1:
            return required, []
        candidates: list[tuple[int, str, str, list[str]]] = []
        for left, right in combinations(required, 2):
            path = self.shortest_path(left, right)
            if path is not None:
                candidates.append((len(path), left, right, path))
        parent = {name: name for name in required}

        def find(name: str) -> str:
            while parent[name] != name:
                parent[name] = parent[parent[name]]
                name = parent[name]
            return name

        def union(left: str, right: str) -> bool:
            left_root = find(left)
            right_root = find(right)
            if left_root == right_root:
                return False
            parent[right_root] = left_root
            return True

        paths: list[list[str]] = []
        for _, left, right, path in sorted(candidates):
            if union(left, right):
                paths.append(path)
        if len({find(name) for name in required}) != 1:
            return None
        nodes: list[str] = []
        for path in paths:
            for name in path:
                if name not in nodes:
                    nodes.append(name)
        return nodes, paths


def context_for_prompt(context: SemanticContext) -> str:
    payload = {
        "retrieval_intent": context.retrieval_intent.model_dump(
            mode="json",
            exclude_none=True,
        ),
        "candidate_models": [
            context.model_details[name] for name in context.candidate_models
        ],
        "members": [
            context.member_details[name] for name in context.allowed_members
        ],
        "binding_candidates": {
            key: [candidate.model_dump(mode="json") for candidate in candidates]
            for key, candidates in context.binding_candidates.items()
        },
        "suggested_members": context.suggested_members,
        "projection_policy": context.projection_policy,
        "fixed_business_context": context.fixed_business_context,
    }
    return json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
