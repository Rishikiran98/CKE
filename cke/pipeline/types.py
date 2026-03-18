"""Canonical pipeline data structures for query orchestration."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from cke.models import Statement

if TYPE_CHECKING:
    from cke.router.query_plan import QueryPlan
    from cke.retrieval.path_types import CandidatePath, LocalSubgraph


@dataclass(slots=True)
class ResolvedEntity:
    surface_form: str
    canonical_name: str
    entity_id: str
    link_confidence: float
    aliases_matched: list[str] = field(default_factory=list)


@dataclass(slots=True)
class RetrievedChunk:
    chunk_id: str
    text: str
    source: str
    score_dense: float
    timestamp: float | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class EvidenceFact:
    statement: Statement
    chunk_id: str
    source: str
    trust_score: float
    retrieval_score: float
    entity_alignment_score: float
    supporting_span: tuple[int, int] | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class ReasoningContext:
    query: str
    query_plan: "QueryPlan"
    resolved_entities: list[ResolvedEntity] = field(default_factory=list)
    retrieved_chunks: list[RetrievedChunk] = field(default_factory=list)
    evidence_facts: list[EvidenceFact] = field(default_factory=list)
    candidate_paths: list["CandidatePath"] = field(default_factory=list)
    subgraph: "LocalSubgraph | None" = None
    decomposition: list[Any] = field(default_factory=list)
    trace_metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class ReasonerOutcome:
    answer: str
    confidence: float
    reasoning_path: list[Statement] = field(default_factory=list)
    required_facts: list[tuple[str, str]] = field(default_factory=list)
    operator_checks: list[dict[str, Any]] = field(default_factory=list)
    summary: str = ""


@dataclass(slots=True)
class QueryResult:
    answer: str
    confidence: float
    reasoning_route: str
    evidence_facts: list[EvidenceFact] = field(default_factory=list)
    candidate_paths: list["CandidatePath"] = field(default_factory=list)
    confidence_signals: dict[str, Any] = field(default_factory=dict)
    verification_summary: str = "not_executed"
    trace_id: str = ""
    failure_mode: str | None = None
    debug_info: dict[str, Any] = field(default_factory=dict)
