"""Shared domain models for CKE."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List


@dataclass(slots=True)
class Entity:
    """Canonical entity stored in the graph."""

    name: str
    aliases: List[str] = field(default_factory=list)


@dataclass(slots=True)
class Statement:
    """A subject-relation-object statement with contextual metadata."""

    subject: str
    relation: str
    object: str
    context: Dict[str, Any] = field(default_factory=dict)
    confidence: float = 1.0
    source: str | None = None
    timestamp: str | None = None
    statement_id: str | None = None
    chunk_id: str | None = None
    source_doc_id: str | None = None
    canonical_subject_id: str | None = None
    canonical_object_id: str | None = None
    trust_score: float | None = None
    retrieval_score: float | None = None
    supporting_span: tuple[int, int] | None = None

    def as_text(self) -> str:
        """Render statement in compact text form for metrics/reasoning."""
        return f"{self.subject} {self.relation} {self.object}"

    def key(self) -> tuple[str, str, str]:
        """Canonical identity key for statement de-duplication."""
        return (self.subject, self.relation, self.object)
