"""Graph edge abstraction with trust metadata for resolved facts."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(slots=True)
class Edge:
    subject: str
    relation: str
    object: str
    confidence_score: float = 0.0
    source_count: int = 1
    timestamp: float = 0.0
    contradiction_flag: bool = False
