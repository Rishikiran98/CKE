"""Candidate memory validation and normalization."""

from __future__ import annotations

from cke.conversation.config import ValidationPolicy
from cke.conversation.patterns import normalize_fact_parts
from cke.conversation.types import CandidateMemory, MemoryKind, MemoryStatus


class CandidateMemoryValidator:
    """Normalize and validate proposals before consolidation."""

    def __init__(self, policy: ValidationPolicy | None = None) -> None:
        self.policy = policy or ValidationPolicy()

    def validate(
        self, candidates: list[CandidateMemory]
    ) -> tuple[list[CandidateMemory], list[CandidateMemory]]:
        accepted: list[CandidateMemory] = []
        rejected: list[CandidateMemory] = []
        for candidate in candidates:
            normalized = self._normalize(candidate)
            reasons = self._validate_candidate(normalized)
            if reasons:
                normalized.status = MemoryStatus.REJECTED
                normalized.rejection_reasons.extend(reasons)
                rejected.append(normalized)
            else:
                accepted.append(normalized)
        return accepted, rejected

    def _normalize(self, candidate: CandidateMemory) -> CandidateMemory:
        subject, relation, object_ = normalize_fact_parts(
            candidate.subject,
            candidate.relation,
            candidate.object,
        )
        candidate.subject = subject
        candidate.relation = relation
        candidate.object = object_
        return candidate

    def _validate_candidate(self, candidate: CandidateMemory) -> list[str]:
        reasons: list[str] = []
        if candidate.kind.value not in self.policy.allowed_memory_kinds:
            reasons.append("unsupported_memory_kind")
        if not candidate.subject or not candidate.relation or not candidate.object:
            reasons.append("missing_core_fields")
        if len(candidate.relation.split("_")) > self.policy.max_relation_tokens:
            reasons.append("relation_too_long")
        if candidate.object.lower() in self.policy.reject_placeholder_objects:
            reasons.append("placeholder_object")
        if candidate.confidence < self.policy.min_confidence:
            reasons.append("low_signal_confidence")
        if any(
            (span.end - span.start) < self.policy.min_span_chars
            for span in candidate.provenance
        ):
            reasons.append("provenance_span_too_short")
        if (
            candidate.kind is MemoryKind.OBSERVATION
            and candidate.relation == "mentions"
            and len(candidate.object) < 2
        ):
            reasons.append("weak_entity_mention")
        return reasons
