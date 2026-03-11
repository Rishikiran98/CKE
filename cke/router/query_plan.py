"""Query planning data structure for graph retrieval."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(slots=True)
class QueryPlan:
    seed_entities: list[str] = field(default_factory=list)
    domains: list[str] = field(default_factory=list)
    intent: str = "factoid"
    max_depth: int = 2
    max_results: int = 12
