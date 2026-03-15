"""Evidence graph artifact built from top-ranked retrieval paths."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(slots=True)
class EvidenceGraph:
    paths: list[dict] = field(default_factory=list)
    nodes: list[dict] = field(default_factory=list)
    confidence: float = 0.0

    def as_dict(self) -> dict:
        return {
            "paths": self.paths,
            "nodes": self.nodes,
            "confidence": self.confidence,
        }
