"""Typed data contracts for conversational memory lifecycle."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from cke.models import Statement


class MemoryKind(str, Enum):
    """High-level kinds of conversational memory managed by the subsystem."""

    FACT = "fact"
    PREFERENCE = "preference"
    PLAN = "plan"
    STATUS = "status"
    TEMPORAL = "temporal"
    SUMMARY = "summary"
    ALIAS = "alias"
    OBSERVATION = "observation"


class MemoryStatus(str, Enum):
    """Lifecycle state for candidate and canonical memories."""

    PROPOSED = "proposed"
    ACCEPTED = "accepted"
    REJECTED = "rejected"
    EPHEMERAL = "ephemeral"
    SUPERSEDED = "superseded"


class MemoryDecisionType(str, Enum):
    """Possible write outcomes after consolidation."""

    ACCEPT = "accept"
    REJECT = "reject"
    EPHEMERAL = "ephemeral"
    UPDATE = "update"
    MERGE = "merge"


class ConfidenceBand(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


@dataclass(slots=True, frozen=True)
class MemorySourceSpan:
    """Provenance span locating text that supported a memory proposal."""

    turn_id: str
    start: int
    end: int
    text: str
    extractor: str


@dataclass(slots=True, frozen=True)
class ConversationEvent:
    """Immutable raw conversation event stored before any memory promotion."""

    conversation_id: str
    event_id: str
    turn_id: str
    turn_order: int
    role: str
    text: str
    timestamp: str
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class CandidateMemory:
    """Extractor proposal that still requires validation and consolidation."""

    candidate_id: str
    conversation_id: str
    event_id: str
    turn_id: str
    kind: MemoryKind
    subject: str
    relation: str
    object: str
    confidence: float
    confidence_band: ConfidenceBand
    provenance: list[MemorySourceSpan] = field(default_factory=list)
    attributes: dict[str, Any] = field(default_factory=dict)
    status: MemoryStatus = MemoryStatus.PROPOSED
    rejection_reasons: list[str] = field(default_factory=list)

    def as_statement(self) -> Statement:
        """Expose the proposal as a statement-shaped object for compatibility."""
        return Statement(
            subject=self.subject,
            relation=self.relation,
            object=self.object,
            context={
                **self.attributes,
                "candidate_id": self.candidate_id,
                "kind": self.kind.value,
                "status": self.status.value,
                "provenance": [
                    {
                        "turn_id": span.turn_id,
                        "start": span.start,
                        "end": span.end,
                        "text": span.text,
                        "extractor": span.extractor,
                    }
                    for span in self.provenance
                ],
            },
            confidence=self.confidence,
            source=self.conversation_id,
        )


@dataclass(slots=True)
class CanonicalMemory:
    """Durable memory accepted after validation and consolidation."""

    memory_id: str
    conversation_id: str
    kind: MemoryKind
    subject: str
    relation: str
    object: str
    confidence: float
    confidence_band: ConfidenceBand
    provenance: list[MemorySourceSpan] = field(default_factory=list)
    aliases: list[str] = field(default_factory=list)
    attributes: dict[str, Any] = field(default_factory=dict)
    first_seen_at: str | None = None
    last_seen_at: str | None = None
    mention_count: int = 1
    status: MemoryStatus = MemoryStatus.ACCEPTED
    supersedes: str | None = None

    def as_statement(self) -> Statement:
        """Project canonical memory into a graph-friendly statement."""
        turn_id = self.provenance[0].turn_id if self.provenance else None
        return Statement(
            subject=self.subject,
            relation=self.relation,
            object=self.object,
            context={
                **self.attributes,
                "memory_id": self.memory_id,
                "kind": self.kind.value,
                "aliases": list(self.aliases),
                "mention_count": self.mention_count,
                "status": self.status.value,
                "turn_id": turn_id,
            },
            confidence=self.confidence,
            source=self.conversation_id,
            timestamp=self.last_seen_at or self.first_seen_at,
            statement_id=self.memory_id,
            chunk_id=turn_id,
            source_doc_id=self.conversation_id,
        )


@dataclass(slots=True)
class MemoryConflict:
    """Structured conflict detected during consolidation."""

    conflict_type: str
    candidate_id: str
    existing_memory_id: str
    reason: str
    conflicting_fields: tuple[str, ...] = ()


@dataclass(slots=True)
class MemoryWriteDecision:
    """Final write decision for a candidate after consolidation."""

    decision: MemoryDecisionType
    candidate: CandidateMemory
    canonical_memory: CanonicalMemory | None = None
    conflict: MemoryConflict | None = None
    reasons: list[str] = field(default_factory=list)


@dataclass(slots=True)
class ConversationTurn:
    """Backward-compatible turn view combining raw event and accepted memories."""

    conversation_id: str
    turn_id: str
    turn_order: int
    role: str
    text: str
    timestamp: str
    entities: list[str] = field(default_factory=list)
    facts: list[Statement] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)
    event_id: str | None = None


@dataclass(slots=True)
class RetrievedMemory:
    """Scored evidence candidate retrieved at answer time."""

    memory_id: str
    memory_type: str
    text: str
    score: float
    conversation_id: str
    turn_id: str | None = None
    turn_order: int | None = None
    role: str | None = None
    subject: str | None = None
    relation: str | None = None
    object: str | None = None
    entities: list[str] = field(default_factory=list)
    facts: list[Statement] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class EvidenceSet:
    """Evidence assembled for grounded answering."""

    query: str
    rewritten_query: str
    supporting_turns: list[RetrievedMemory] = field(default_factory=list)
    supporting_memories: list[CanonicalMemory] = field(default_factory=list)
    supporting_facts: list[Statement] = field(default_factory=list)
    graph_facts: list[Statement] = field(default_factory=list)
    summaries: list[str] = field(default_factory=list)
    conflicts: list[MemoryConflict] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class RetrievalBundle:
    """Retrieval-stage payload before final evidence selection."""

    rewritten_query: str
    raw_candidates: list[RetrievedMemory] = field(default_factory=list)
    retrieved_turns: list[RetrievedMemory] = field(default_factory=list)
    retrieved_memories: list[CanonicalMemory] = field(default_factory=list)
    retrieved_facts: list[Statement] = field(default_factory=list)
    graph_neighbors: list[Statement] = field(default_factory=list)
    summaries: list[str] = field(default_factory=list)
    evidence: EvidenceSet | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class ConversationAnswer:
    """Grounded answer plus evidence and confidence metadata."""

    answer: str
    confidence: float
    grounded: bool
    evidence: EvidenceSet | None = None
    retrieved_turns: list[RetrievedMemory] = field(default_factory=list)
    retrieved_facts: list[Statement] = field(default_factory=list)
    graph_neighbors: list[Statement] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class TurnIngestionResult:
    """Result of event storage plus memory pipeline execution."""

    event: ConversationEvent
    turn: ConversationTurn
    raw_candidates: list[CandidateMemory] = field(default_factory=list)
    accepted_memories: list[CanonicalMemory] = field(default_factory=list)
    rejected_candidates: list[CandidateMemory] = field(default_factory=list)
    decisions: list[MemoryWriteDecision] = field(default_factory=list)
