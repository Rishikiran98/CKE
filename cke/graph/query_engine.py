"""High-level query API over the knowledge graph."""

from __future__ import annotations

from collections import deque

from cke.graph_engine.graph_engine import KnowledgeGraphEngine
from cke.models import Statement


class GraphQueryEngine:
    """Query facade that decouples retrieval from storage internals."""

    def __init__(self, graph: KnowledgeGraphEngine) -> None:
        self.graph = graph

    def neighbors(self, entity: str) -> list[Statement]:
        return self.graph.get_neighbors(entity)

    def paths(
        self, entity1: str, entity2: str, cutoff: int = 3
    ) -> list[list[Statement]]:
        return self.graph.find_paths(entity1, entity2, cutoff=cutoff)

    def relations(self, entity: str) -> list[str]:
        return sorted({edge.relation for edge in self.graph.get_neighbors(entity)})

    def subgraph(
        self,
        seed_entities: list[str],
        depth: int = 1,
        max_nodes: int = 200,
    ) -> dict[str, list[Statement] | list[str]]:
        visited: set[str] = set()
        queue: deque[tuple[str, int]] = deque((seed, 0) for seed in seed_entities)
        edges: list[Statement] = []

        while queue and len(visited) < max_nodes:
            node, level = queue.popleft()
            norm = self.graph._normalize_entity(node)
            if norm in visited or level > depth:
                continue
            visited.add(norm)
            for edge in self.graph.get_neighbors(node):
                edges.append(edge)
                if level < depth:
                    queue.append((edge.object, level + 1))

        return {
            "entities": sorted(visited),
            "edges": edges,
        }
