"""Knowledge graph storage.

Uses NetworkX when available and a lightweight internal fallback otherwise.
"""

from __future__ import annotations

from collections import defaultdict, deque
from typing import Any, Dict, List

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
            self.graph: Dict[str, list[dict[str, Any]]] = defaultdict(list)
            self._nodes: set[str] = set()

    def add_statement(
        self,
        subject: str,
        relation: str,
        object_: str,
        context: dict[str, Any] | None = None,
        confidence: float = 1.0,
        source: str | None = None,
        timestamp: str | None = None,
    ) -> None:
        payload = {
            "relation": relation,
            "context": context or {},
            "confidence": confidence,
            "source": source,
            "timestamp": timestamp,
        }

        if self._use_nx:
            self.graph.add_node(subject)
            self.graph.add_node(object_)
            self.graph.add_edge(subject, object_, **payload)
        else:
            self._nodes.update([subject, object_])
            self.graph[subject].append({"object": object_, **payload})

    def add_statements(self, statements: List[Statement]) -> None:
        for st in statements:
            self.add_statement(
                st.subject,
                st.relation,
                st.object,
                context=st.context,
                confidence=st.confidence,
                source=st.source,
                timestamp=st.timestamp,
            )

    def get_neighbors(self, entity: str) -> List[Statement]:
        if self._use_nx:
            if entity not in self.graph:
                return []
            return [
                Statement(
                    subject=entity,
                    relation=edge_data.get("relation", "related_to"),
                    object=target,
                    context=dict(edge_data.get("context", {})),
                    confidence=float(edge_data.get("confidence", 1.0)),
                    source=edge_data.get("source"),
                    timestamp=edge_data.get("timestamp"),
                )
                for _, target, edge_data in self.graph.out_edges(entity, data=True)
            ]

        return [
            Statement(
                subject=entity,
                relation=item.get("relation", "related_to"),
                object=item.get("object", ""),
                context=dict(item.get("context", {})),
                confidence=float(item.get("confidence", 1.0)),
                source=item.get("source"),
                timestamp=item.get("timestamp"),
            )
            for item in self.graph.get(entity, [])
        ]

    def find_paths(
        self, entity_a: str, entity_b: str, cutoff: int = 3
    ) -> List[List[Statement]]:
        if self._use_nx:
            if entity_a not in self.graph or entity_b not in self.graph:
                return []
            paths: list[list[Statement]] = []
            for node_path in nx.all_simple_paths(
                self.graph, source=entity_a, target=entity_b, cutoff=cutoff
            ):
                statement_path: list[Statement] = []
                for i in range(len(node_path) - 1):
                    src, dst = node_path[i], node_path[i + 1]
                    edge_map = self.graph.get_edge_data(src, dst) or {}
                    edge = (
                        edge_map[min(edge_map.keys())]
                        if edge_map
                        else {"relation": "related_to"}
                    )
                    statement_path.append(
                        Statement(
                            subject=src,
                            relation=edge.get("relation", "related_to"),
                            object=dst,
                            context=dict(edge.get("context", {})),
                            confidence=float(edge.get("confidence", 1.0)),
                            source=edge.get("source"),
                            timestamp=edge.get("timestamp"),
                        )
                    )
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
            for item in self.graph.get(node, []):
                nxt = item.get("object", "")
                if any(step.object == nxt for step in path):
                    continue
                queue.append(
                    (
                        nxt,
                        path
                        + [
                            Statement(
                                subject=node,
                                relation=item.get("relation", "related_to"),
                                object=nxt,
                                context=dict(item.get("context", {})),
                                confidence=float(item.get("confidence", 1.0)),
                                source=item.get("source"),
                                timestamp=item.get("timestamp"),
                            )
                        ],
                    )
                )
        return results

    def all_entities(self) -> list[str]:
        if self._use_nx:
            return list(self.graph.nodes)
        return sorted(self._nodes)
