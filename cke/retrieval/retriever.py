"""Sparse graph retrieval engine."""

from __future__ import annotations

from collections import deque
from typing import List

from cke.graph_engine.graph_engine import KnowledgeGraphEngine
from cke.models import Statement
from cke.router.router import QueryRouter


class GraphRetriever:
    """Retrieve minimal graph context around routed entities."""

    def __init__(self, graph_engine: KnowledgeGraphEngine, router: QueryRouter | None = None) -> None:
        self.graph_engine = graph_engine
        self.router = router or QueryRouter()

    def retrieve(self, query: str, max_depth: int = 2) -> List[Statement]:
        seeds = self.router.detect_entities(query, self.graph_engine.all_entities())
        seen_statements: set[tuple[str, str, str]] = set()
        candidates: list[tuple[int, Statement]] = []

        for seed in seeds:
            # Keep bounded BFS behavior while tracking shortest discovery depth.
            queue = deque([(seed, 0)])
            best_depth_for_node: dict[str, int] = {}
            while queue:
                node, depth = queue.popleft()
                if depth > max_depth:
                    continue
                if node in best_depth_for_node and depth > best_depth_for_node[node]:
                    continue
                best_depth_for_node[node] = depth

                for statement in self.graph_engine.get_neighbors(node):
                    key = statement.key()
                    if key not in seen_statements:
                        seen_statements.add(key)
                        candidates.append((depth + 1, statement))
                    if depth + 1 <= max_depth:
                        queue.append((statement.object, depth + 1))

        # Rank by: confidence desc, path length asc, then deterministic text key.
        ranked = sorted(
            candidates,
            key=lambda item: (
                -float(item[1].confidence),
                item[0],
                item[1].subject.lower(),
                item[1].relation.lower(),
                item[1].object.lower(),
            ),
        )
        return [statement for _, statement in ranked]
