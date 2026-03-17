"""Query planning data structure for graph retrieval."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(slots=True)
class QueryPlan:
    query_text: str = ""
    seed_entities: list[str] = field(default_factory=list)
    decomposition: list[dict[str, str | float]] = field(default_factory=list)
    domains: list[str] = field(default_factory=list)
    intent: str = "factoid"
    max_depth: int = 2
    max_results: int = 12
    confidence_score: float = 0.0
    reasoning_route: str = "advanced_reasoner"
    operator_hint: str | None = None
