"""Abstention logic for weak or conflicting evidence."""

from __future__ import annotations

from cke.conversation.config import AnsweringConfig, RetrievalConfig
from cke.conversation.types import EvidenceSet


class AbstentionDecider:
    """Decide when the system should refrain from answering."""

    def __init__(
        self,
        answering_config: AnsweringConfig | None = None,
        retrieval_config: RetrievalConfig | None = None,
    ) -> None:
        self.answering_config = answering_config or AnsweringConfig()
        self.retrieval_config = retrieval_config or RetrievalConfig()

    def should_abstain(self, evidence: EvidenceSet) -> tuple[bool, str]:
        best_score = max(
            (item.score for item in evidence.supporting_turns), default=0.0
        )
        if evidence.conflicts:
            return True, "conflicting_evidence"
        if not evidence.supporting_turns and not evidence.supporting_memories:
            return True, "no_evidence"
        if (
            best_score < self.retrieval_config.weak_match_threshold
            and not evidence.supporting_facts
        ):
            return True, "weak_evidence"
        return False, ""
