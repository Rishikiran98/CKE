"""Trust calibration module for data-driven trust scoring."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

from cke.diagnostics import DegradationMixin
from cke.graph.assertion import Assertion


@dataclass(slots=True)
class TrustCalibrationConfig:
    """Linear trust model weights and decay parameters."""

    w_src: float = 0.8
    w_freq: float = 0.6
    w_conf: float = 1.0
    tau: float = 60.0 * 60.0 * 24.0 * 365.0
    low_trust_threshold: float = 0.4


class TrustCalibrator(DegradationMixin):
    """Calibrate trust scoring from graph behavior statistics."""

    def __init__(
        self,
        config: TrustCalibrationConfig | None = None,
        strict: bool = False,
    ) -> None:
        self._init_degradation(strict)
        self.config = config or TrustCalibrationConfig()

    @staticmethod
    def sigmoid(value: float) -> float:
        """Compute numerically-stable sigmoid."""
        if value >= 0:
            exp_neg = math.exp(-value)
            return 1.0 / (1.0 + exp_neg)
        exp_pos = math.exp(value)
        return exp_pos / (1.0 + exp_pos)

    def calibrate(
        self,
        source_weight: float,
        evidence_count: int,
        extractor_confidence: float,
    ) -> float:
        """Compute base trust using weighted source/frequency/confidence terms."""
        score = (
            self.config.w_src * source_weight
            + self.config.w_freq * math.log(1.0 + max(0, evidence_count))
            + self.config.w_conf * max(0.0, min(1.0, extractor_confidence))
        )
        return self.sigmoid(score)

    def apply_decay(self, trust_0: float, now: float, observed_ts: float) -> float:
        """Apply exponential recency decay to a base trust score."""
        delta = max(0.0, now - observed_ts)
        tau = max(self.config.tau, 1e-9)
        return trust_0 * math.exp(-(delta / tau))

    def fit_from_graph(self, graph: Any) -> dict[str, float]:
        """Adjust model weights from contradiction and evidence agreement signals."""
        assertions = self._collect_assertions(graph)
        if not assertions:
            return self.to_dict()

        contradiction_rate = self._contradiction_rate(assertions)
        evidence_agreement = self._evidence_agreement(assertions)
        if evidence_agreement is None:
            # Nothing carried evidence, so the two weights that agreement
            # drives are left where they are rather than fitted to a number
            # nobody measured.
            self._degrade(
                f"none of the {len(assertions)} assertions in this graph carry "
                f"evidence text, so evidence agreement could not be measured "
                f"and w_freq and w_conf are left unfitted"
            )
            self.config.w_src = max(
                0.05, self.config.w_src * (1.0 - 0.4 * contradiction_rate)
            )
            return self.to_dict()

        self.config.w_src = max(
            0.05, self.config.w_src * (1.0 - 0.4 * contradiction_rate)
        )
        self.config.w_freq = max(
            0.05, self.config.w_freq * (0.8 + 0.4 * evidence_agreement)
        )
        self.config.w_conf = max(
            0.05, self.config.w_conf * (0.8 + 0.4 * evidence_agreement)
        )
        return self.to_dict()

    def to_dict(self) -> dict[str, float]:
        """Export current calibration values."""
        return {
            "w_src": self.config.w_src,
            "w_freq": self.config.w_freq,
            "w_conf": self.config.w_conf,
            "tau": self.config.tau,
            "low_trust_threshold": self.config.low_trust_threshold,
        }

    @staticmethod
    def _collect_assertions(graph: Any) -> list[Assertion]:
        if isinstance(graph, list):
            return [item for item in graph if isinstance(item, Assertion)]
        if hasattr(graph, "assertions"):
            return [item for item in graph.assertions if isinstance(item, Assertion)]
        return []

    @staticmethod
    def _contradiction_rate(assertions: list[Assertion]) -> float:
        grouped: dict[tuple[str, str], set[str]] = {}
        for assertion in assertions:
            key = (assertion.subject, assertion.relation)
            grouped.setdefault(key, set()).add(assertion.object)
        contradictory_groups = sum(1 for values in grouped.values() if len(values) > 1)
        total = max(len(grouped), 1)
        return contradictory_groups / total

    def _evidence_agreement(self, assertions: list[Assertion]) -> float | None:
        """The share of evidenced assertions whose evidence agrees with itself.

        ``len(texts) <= 1`` counted an assertion with *no* evidence as
        agreeing, so a graph of unevidenced assertions scored perfect
        agreement and inflated w_freq and w_conf through fit_from_graph. An
        assertion with nothing to agree about is not agreement; it is not a
        measurement at all, so it is left out of both sides of the ratio.

        ``None`` when no assertion carries evidence: there is nothing to
        measure, and a zero there would read as total disagreement.
        """
        agreeing = 0
        evidenced = 0
        for assertion in assertions:
            texts = {
                str(item.text).strip().lower()
                for item in assertion.evidence
                if getattr(item, "text", "").strip()
            }
            if not texts:
                continue
            evidenced += 1
            if len(texts) == 1:
                agreeing += 1
        if not evidenced:
            return None
        return agreeing / evidenced
