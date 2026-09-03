"""Contextual graph storage built on NetworkX MultiDiGraph."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass, field
from typing import Any

import networkx as nx

from cke.diagnostics import DegradationMixin


@dataclass
class GraphEntity:
    name: str
    attributes: dict[str, Any] = field(default_factory=dict)


@dataclass
class GraphAssertion:
    subject: str
    relation: str
    object: str
    confidence: float = 1.0
    qualifiers: dict[str, Any] = field(default_factory=dict)
    evidence: list[dict[str, Any]] = field(default_factory=list)


#: Stored for an assertion that arrived with no confidence. Maximum
#: certainty for something nothing scored, so it is declared.
_UNSCORED_CONFIDENCE = 1.0


class GraphStore(DegradationMixin):
    def __init__(self, strict: bool = False) -> None:
        self._init_degradation(strict)
        self.graph = nx.MultiDiGraph()

    def add_entity(self, entity: GraphEntity | str) -> None:
        if isinstance(entity, str):
            self.graph.add_node(entity)
        else:
            self.graph.add_node(entity.name, **entity.attributes)

    def add_assertion(self, assertion: GraphAssertion | dict[str, Any]) -> None:
        payload = assertion if isinstance(assertion, dict) else assertion.__dict__
        subject = str(payload["subject"])
        object_ = str(payload["object"])
        self.add_entity(subject)
        self.add_entity(object_)
        self.graph.add_edge(
            subject,
            object_,
            relation=str(payload["relation"]),
            confidence=self._resolve_confidence(payload),
            qualifiers=dict(payload.get("qualifiers", {})),
            evidence=list(payload.get("evidence", [])),
        )

    def _resolve_confidence(self, payload: dict[str, Any]) -> float:
        """Return the payload's confidence, or declare the substitution.

        An edge arriving without one used to be stored at maximum certainty,
        so anything reading trust off the graph read this constant.
        """
        if "confidence" in payload:
            return float(payload["confidence"])
        self._degrade(
            "an assertion arrived with no confidence, so it is stored at "
            f"{_UNSCORED_CONFIDENCE}, the maximum, for something nothing scored"
        )
        return _UNSCORED_CONFIDENCE

    def get_entity(self, name: str) -> dict[str, Any] | None:
        if name not in self.graph:
            return None
        return {"name": name, **self.graph.nodes[name]}

    def neighbors(self, entity: str) -> list[dict[str, Any]]:
        if entity not in self.graph:
            return []
        return [
            {"subject": entity, "object": dst, **attrs}
            for _, dst, attrs in self.graph.out_edges(entity, data=True)
        ]

    def find_paths(
        self, source: str, target: str, max_depth: int = 3
    ) -> list[list[dict[str, Any]]]:
        if source not in self.graph or target not in self.graph:
            return []
        out: list[list[dict[str, Any]]] = []
        for nodes in nx.all_simple_paths(
            self.graph, source=source, target=target, cutoff=max_depth
        ):
            path: list[dict[str, Any]] = []
            for i in range(len(nodes) - 1):
                src, dst = nodes[i], nodes[i + 1]
                edges = self.graph.get_edge_data(src, dst) or {}
                first_key = next(iter(edges), None)
                attrs = edges[first_key] if first_key is not None else {}
                path.append({"subject": src, "object": dst, **attrs})
            out.append(path)
        return out


def example_usage() -> dict[str, Iterable[dict[str, Any]]]:
    """Simple insertion/query example used by docs and smoke tests."""
    store = GraphStore()
    store.add_assertion(
        GraphAssertion(
            subject="Alexander Fleming",
            relation="discovered",
            object="Penicillin",
            confidence=0.98,
            qualifiers={"year": 1928},
            evidence=[{"doc_id": "wiki_fleming", "span": [10, 42]}],
        )
    )
    return {
        "neighbors": store.neighbors("Alexander Fleming"),
        "paths": store.find_paths("Alexander Fleming", "Penicillin", max_depth=2),
    }
