"""Consolidation logic for dedupe, updates, and conflict handling."""

from __future__ import annotations

from uuid import uuid4

from cke.conversation.config import ConsolidationPolicy
from cke.conversation.types import (
    CandidateMemory,
    CanonicalMemory,
    ConfidenceBand,
    MemoryConflict,
    MemoryDecisionType,
    MemoryKind,
    MemoryStatus,
    MemoryWriteDecision,
)


class MemoryConsolidator:
    """Turn validated candidates into canonical memory write decisions."""

    def __init__(self, policy: ConsolidationPolicy | None = None) -> None:
        self.policy = policy or ConsolidationPolicy()

    def consolidate(
        self,
        candidates: list[CandidateMemory],
        existing: list[CanonicalMemory],
        *,
        timestamp: str,
    ) -> list[MemoryWriteDecision]:
        decisions: list[MemoryWriteDecision] = []
        existing_by_key = {
            (memory.subject, memory.relation, memory.object): memory
            for memory in existing
        }
        existing_by_pair = {
            (memory.subject, memory.relation): memory
            for memory in existing
            if memory.status == MemoryStatus.ACCEPTED
        }

        for candidate in candidates:
            exact = existing_by_key.get(
                (candidate.subject, candidate.relation, candidate.object)
            )
            if exact is not None:
                exact.mention_count += 1
                exact.last_seen_at = timestamp
                exact.provenance.extend(candidate.provenance)
                exact.confidence = max(exact.confidence, candidate.confidence)
                decisions.append(
                    MemoryWriteDecision(
                        decision=MemoryDecisionType.MERGE,
                        candidate=candidate,
                        canonical_memory=exact,
                        reasons=["duplicate_mention_merged"],
                    )
                )
                continue

            if candidate.kind.value in self.policy.ephemeral_kinds:
                candidate.status = MemoryStatus.EPHEMERAL
                decisions.append(
                    MemoryWriteDecision(
                        decision=MemoryDecisionType.EPHEMERAL,
                        candidate=candidate,
                        reasons=["kept_as_ephemeral_observation"],
                    )
                )
                continue

            pair = existing_by_pair.get((candidate.subject, candidate.relation))
            if pair is not None and pair.object != candidate.object:
                conflict = MemoryConflict(
                    conflict_type="contradiction",
                    candidate_id=candidate.candidate_id,
                    existing_memory_id=pair.memory_id,
                    reason="new_candidate_conflicts_with_existing_canonical",
                    conflicting_fields=("object",),
                )
                if (
                    candidate.relation in self.policy.update_relations
                    or candidate.kind
                    in {MemoryKind.STATUS, MemoryKind.TEMPORAL, MemoryKind.PREFERENCE}
                ):
                    pair.status = MemoryStatus.SUPERSEDED
                    canonical = self._build_canonical(
                        candidate, timestamp, supersedes=pair.memory_id
                    )
                    existing_by_key[
                        (canonical.subject, canonical.relation, canonical.object)
                    ] = canonical
                    existing_by_pair[(canonical.subject, canonical.relation)] = (
                        canonical
                    )
                    decisions.append(
                        MemoryWriteDecision(
                            decision=MemoryDecisionType.UPDATE,
                            candidate=candidate,
                            canonical_memory=canonical,
                            conflict=conflict,
                            reasons=["superseded_prior_memory"],
                        )
                    )
                else:
                    candidate.status = MemoryStatus.REJECTED
                    candidate.rejection_reasons.append("unresolved_conflict")
                    decisions.append(
                        MemoryWriteDecision(
                            decision=MemoryDecisionType.REJECT,
                            candidate=candidate,
                            conflict=conflict,
                            reasons=["unresolved_conflict"],
                        )
                    )
                continue

            canonical = self._build_canonical(candidate, timestamp)
            existing_by_key[
                (canonical.subject, canonical.relation, canonical.object)
            ] = canonical
            existing_by_pair[(canonical.subject, canonical.relation)] = canonical
            decisions.append(
                MemoryWriteDecision(
                    decision=MemoryDecisionType.ACCEPT,
                    candidate=candidate,
                    canonical_memory=canonical,
                    reasons=["accepted_new_canonical_memory"],
                )
            )
        return decisions

    def _build_canonical(
        self,
        candidate: CandidateMemory,
        timestamp: str,
        *,
        supersedes: str | None = None,
    ) -> CanonicalMemory:
        candidate.status = MemoryStatus.ACCEPTED
        return CanonicalMemory(
            memory_id=f"mem-{uuid4().hex[:12]}",
            conversation_id=candidate.conversation_id,
            kind=candidate.kind,
            subject=candidate.subject,
            relation=candidate.relation,
            object=candidate.object,
            confidence=candidate.confidence,
            confidence_band=(
                ConfidenceBand.HIGH
                if candidate.confidence >= 0.7
                else (
                    ConfidenceBand.MEDIUM
                    if candidate.confidence >= 0.45
                    else ConfidenceBand.LOW
                )
            ),
            provenance=list(candidate.provenance),
            attributes=dict(candidate.attributes),
            first_seen_at=timestamp,
            last_seen_at=timestamp,
            mention_count=1,
            supersedes=supersedes,
        )
