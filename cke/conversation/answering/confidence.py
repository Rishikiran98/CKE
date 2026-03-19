"""Confidence estimation for grounded conversational answers."""

from __future__ import annotations

from cke.conversation.config import AnsweringConfig
from cke.conversation.types import ConfidenceBand, EvidenceSet


class ConfidenceEstimator:
    """Estimate numeric confidence and coarse confidence bands."""

    def __init__(self, config: AnsweringConfig | None = None) -> None:
        self.config = config or AnsweringConfig()

    def estimate(
        self, evidence: EvidenceSet, *, grounded: bool
    ) -> tuple[float, ConfidenceBand]:
        if not grounded:
            score = (
                self.config.weak_confidence
                if evidence.supporting_turns or evidence.supporting_memories
                else self.config.abstain_confidence
            )
        else:
            support = min(
                1.0,
                0.4
                + (0.1 * len(evidence.supporting_turns))
                + (0.08 * len(evidence.supporting_memories)),
            )
            score = max(
                0.0,
                min(
                    1.0,
                    self.config.grounded_confidence * support
                    - (len(evidence.conflicts) * self.config.conflict_penalty),
                ),
            )
        band = (
            ConfidenceBand.HIGH
            if score >= 0.75
            else ConfidenceBand.MEDIUM if score >= 0.4 else ConfidenceBand.LOW
        )
        return score, band
