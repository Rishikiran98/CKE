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
    qualifiers: Dict[str, Any] = field(default_factory=dict)
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
        base = f"{self.subject} {self.relation} {self.object}"
        if self.qualifiers:
            parts = []
            for key, value in sorted(self.qualifiers.items()):
                if key == "contradiction_flag":
                    continue
                parts.append(f"{key}={value}")
            if parts:
                base += f" [{', '.join(parts)}]"
        return base

    def key(self) -> tuple[str, str, str]:
        """Canonical identity key for statement de-duplication."""
        return (self.subject, self.relation, self.object)

    def to_assertion(self) -> "Any":
        """Convert to Assertion for trust/conflict workflows."""
        from cke.schema.assertion import Assertion, Evidence

        evidence_list: List[Any] = []
        for ev in self.context.get("evidence", []):
            span = None
            if "span_start" in ev and "span_end" in ev:
                span = (int(ev["span_start"]), int(ev["span_end"]))
            evidence_list.append(
                Evidence(
                    chunk_id=ev.get("chunk_id"),
                    span=span,
                    text=ev.get("text", ""),
                    extractor_confidence=float(ev.get("extractor_confidence", 1.0)),
                    source_weight=float(ev.get("source_weight", 1.0)),
                )
            )
        return Assertion(
            subject=self.subject,
            relation=self.relation,
            object=self.object,
            qualifiers=dict(self.qualifiers),
            evidence=evidence_list,
            confidence=self.confidence,
            trust_score=self.trust_score if self.trust_score is not None else self.confidence,
            source=self.source or "unknown",
        )

    @classmethod
    def from_assertion(cls, assertion: "Any") -> "Statement":
        """Create Statement from an Assertion instance."""
        return cls(
            subject=assertion.subject,
            relation=assertion.relation,
            object=assertion.object,
            qualifiers=dict(assertion.qualifiers),
            confidence=assertion.confidence,
            trust_score=assertion.trust_score,
            source=assertion.source,
        )
