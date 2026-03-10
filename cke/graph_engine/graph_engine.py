"""Knowledge graph storage.

Uses NetworkX when available and a lightweight internal fallback otherwise.
"""

from __future__ import annotations

from collections import defaultdict, deque
from typing import Dict, List

from cke.models import Statement

try:
    import networkx as nx
except Exception:  # pragma: no cover
    nx = None


class KnowledgeGraphEngine:
    """Manages entity-relation graph operations."""

    def __init__(self) -> None:
        self._use_nx = nx is not None
        if self._use_nx:
            self.graph = nx.MultiDiGraph()
        else:
            self.graph: Dict[str, list[tuple[str, str]]] = defaultdict(list)
            self._nodes: set[str] = set()

    def add_statement(self, subject: str, relation: str, object_: str) -> None:
        if self._use_nx:
            self.graph.add_node(subject)
            self.graph.add_node(object_)
            self.graph.add_edge(subject, object_, relation=relation)
        else:
            self._nodes.update([subject, object_])
            self.graph[subject].append((relation, object_))

    def add_statements(self, statements: List[Statement]) -> None:
        for st in statements:
            self.add_statement(st.subject, st.relation, st.object)

    def get_neighbors(self, entity: str) -> List[Statement]:
        if self._use_nx:
            if entity not in self.graph:
                return []
            return [
                Statement(entity, edge_data.get("relation", "related_to"), target)
                for _, target, edge_data in self.graph.out_edges(entity, data=True)
            ]

        return [Statement(entity, relation, obj) for relation, obj in self.graph.get(entity, [])]

    def find_paths(self, entity_a: str, entity_b: str, cutoff: int = 3) -> List[List[Statement]]:
        if self._use_nx:
            if entity_a not in self.graph or entity_b not in self.graph:
                return []
            paths: list[list[Statement]] = []
            for node_path in nx.all_simple_paths(self.graph, source=entity_a, target=entity_b, cutoff=cutoff):
                statement_path: list[Statement] = []
                for i in range(len(node_path) - 1):
                    src, dst = node_path[i], node_path[i + 1]
                    edge_map = self.graph.get_edge_data(src, dst) or {}
                    edge = edge_map[min(edge_map.keys())] if edge_map else {"relation": "related_to"}
                    statement_path.append(Statement(src, edge.get("relation", "related_to"), dst))
                paths.append(statement_path)
            return paths

        if entity_a not in self._nodes or entity_b not in self._nodes:
            return []

        results: list[list[Statement]] = []
        queue = deque([(entity_a, [])])
        while queue:
            node, path = queue.popleft()
            if len(path) > cutoff:
                continue
            if node == entity_b and path:
                results.append(path)
                continue
            for relation, nxt in self.graph.get(node, []):
                if any(step.object == nxt for step in path):
                    continue
                queue.append((nxt, path + [Statement(node, relation, nxt)]))
        return results

    def all_entities(self) -> list[str]:
        if self._use_nx:
            return list(self.graph.nodes)
        return sorted(self._nodes)
