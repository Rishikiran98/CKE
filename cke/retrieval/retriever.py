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
        visited_nodes: set[str] = set()
        seen_statements: set[tuple[str, str, str]] = set()
        results: list[Statement] = []

        for seed in seeds:
            queue = deque([(seed, 0)])
            while queue:
                node, depth = queue.popleft()
                if node in visited_nodes or depth > max_depth:
                    continue
                visited_nodes.add(node)

                for statement in self.graph_engine.get_neighbors(node):
                    key = (statement.subject, statement.relation, statement.object)
                    if key not in seen_statements:
                        seen_statements.add(key)
                        results.append(statement)
                    if depth + 1 <= max_depth:
                        queue.append((statement.object, depth + 1))

        return results
