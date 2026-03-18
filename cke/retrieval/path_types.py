"""Structured types for local subgraphs and candidate reasoning paths."""

from __future__ import annotations

from dataclasses import dataclass, field

from cke.models import Statement


@dataclass(slots=True)
class CandidatePath:
    """A scored path candidate assembled from retrieved evidence."""

    statements: list[Statement]
    path_score: float
    entity_overlap_score: float
    relation_match_score: float
    trust_score: float
    summary: str = ""


@dataclass(slots=True)
class LocalSubgraph:
    """A shallow query-local evidence graph used for path generation."""

    seed_entities: list[str]
    entities: list[str]
    edges: list[Statement]
    metadata: dict[str, object] = field(default_factory=dict)
