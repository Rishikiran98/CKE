"""Trust scoring utilities for assertions."""

from __future__ import annotations

import math
import time

from cke.graph.assertion import Assertion


class TrustEngine:
    """Compute trust scores from source quality, evidence, and recency."""

    def __init__(
        self,
        tau: float = 60.0 * 60.0 * 24.0 * 365.0,
        source_weights: dict[str, float] | None = None,
    ) -> None:
        self.tau = tau
        self.source_weights = source_weights or {
            "wikipedia": 1.0,
            "paper": 1.2,
            "docs": 1.1,
            "unknown": 0.7,
        }

    @staticmethod
    def _sigmoid(value: float) -> float:
        return 1.0 / (1.0 + math.exp(-value))

    def compute_trust(self, assertion: Assertion, now: float | None = None) -> float:
        """Compute trust score with source/evidence weighting and time decay."""
        source_weight = self.source_weights.get(
            assertion.source, self.source_weights["unknown"]
        )
        evidence_term = math.log(max(assertion.evidence_count, 0) + 1.0)
        extractor_conf = max(0.0, min(assertion.extractor_confidence, 1.0))

        base_trust = self._sigmoid(source_weight * evidence_term * extractor_conf)

        now_ts = float(time.time()) if now is None else float(now)
        delta_time = max(0.0, now_ts - float(assertion.timestamp))
        decayed = base_trust * math.exp(-(delta_time / self.tau))

        assertion.trust_score = decayed
        return decayed
