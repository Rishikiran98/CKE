"""Assertion and evidence models used by graph update workflows."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any
import time


@dataclass(slots=True)
class Evidence:
    """Evidence span metadata that supports an assertion."""

    text: str = ""
    source: str | None = None
    confidence: float = 1.0
    doc_id: str | None = None
    chunk_id: str | None = None
    span_start: int | None = None
    span_end: int | None = None
    extractor_confidence: float = 1.0


@dataclass(slots=True)
class Assertion:
    """Contextual assertion with quality and provenance metadata."""

    subject: str
    relation: str
    object: str
    qualifiers: dict[str, Any] = field(default_factory=dict)
    trust_score: float = 0.5
    source: str = "unknown"
    timestamp: float = field(default_factory=lambda: float(time.time()))
    evidence_count: int = 1
    extractor_confidence: float = 1.0
    evidence: list[Evidence] = field(default_factory=list)
    secondary: bool = False

    def key(self) -> tuple[str, str, str]:
        """Identity key used for grouping candidate conflicts."""
        return (self.subject, self.relation, self.object)
