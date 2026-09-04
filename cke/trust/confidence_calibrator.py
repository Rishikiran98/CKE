"""Deterministic confidence calibration for Sprint 9 reasoning decisions."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from cke.diagnostics import DegradationMixin


@dataclass(slots=True)
class ConfidenceCalibrationConfig:
    evidence_weight: float = 0.35
    path_weight: float = 0.20
    operator_weight: float = 0.15
    entity_weight: float = 0.15
    route_weight: float = 0.15
    verification_issue_penalty: float = 0.12
    contradiction_penalty: float = 0.45
    low_evidence_cap: float = 0.7
    low_evidence_threshold: int = 1
    abstain_threshold: float = 0.4


#: The signals the weighted sum reads. A caller that omits one is not saying
#: it measured zero; it is saying nothing, and the two must not look alike.
_WEIGHTED_SIGNALS = (
    "path_score",
    "operator_confidence",
    "route_confidence",
)


# Not slots=True: the degradation contract sets its own attributes, and a
# calibrator that cannot record having degraded is the defect this closes.
@dataclass
class ConfidenceCalibrator(DegradationMixin):
    """Interpretably combine structured confidence signals into final confidence."""

    config: ConfidenceCalibrationConfig = field(
        default_factory=ConfidenceCalibrationConfig
    )
    strict: bool = False

    def __post_init__(self) -> None:
        self._init_degradation(self.strict)

    def calibrate(self, signals: dict[str, Any]) -> float:
        absent = [name for name in _WEIGHTED_SIGNALS if name not in signals]
        if absent:
            # Each of these carries a weight in the sum below. Reading an
            # absent one as 0.0 pulls the final confidence down by that
            # weight and reports the result as a calibrated number.
            self._degrade(
                f"the calibrator was given no {', '.join(absent)}, and each "
                f"carries a weight in the confidence it returns, so the "
                f"figure is computed as though those signals were measured "
                f"at zero"
            )
        normalized_evidence_score = self._normalized_evidence_score(signals)
        path_score = self._clip(signals.get("path_score", 0.0))
        operator_confidence = self._clip(signals.get("operator_confidence", 0.0))
        entity_confidence = self._clip(
            signals.get(
                "entity_resolution_confidence", signals.get("entity_confidence", 0.0)
            )
        )
        route_confidence = self._clip(signals.get("route_confidence", 0.0))

        confidence = (
            self.config.evidence_weight * normalized_evidence_score
            + self.config.path_weight * path_score
            + self.config.operator_weight * operator_confidence
            + self.config.entity_weight * entity_confidence
            + self.config.route_weight * route_confidence
        )

        verification_issues = list(signals.get("verification_issues", []))
        if verification_issues:
            confidence -= self.config.verification_issue_penalty * min(
                len(verification_issues), 3
            )

        if bool(signals.get("contradiction_flag", False)):
            confidence -= self.config.contradiction_penalty

        evidence_count = int(signals.get("evidence_count", 0) or 0)
        if evidence_count <= self.config.low_evidence_threshold:
            confidence = min(confidence, self.config.low_evidence_cap)

        if bool(signals.get("verification_pass", True)) is False:
            return 0.0
        if evidence_count <= 0:
            return 0.0
        if bool(signals.get("contradiction_flag", False)):
            confidence = min(confidence, 0.25)

        return self._clip(confidence)

    def should_abstain(self, confidence: float, signals: dict[str, Any]) -> bool:
        threshold = self.config.abstain_threshold
        if not signals.get("verification_pass", False):
            return True
        if (
            signals.get("operator_failed", False)
            and signals.get("route_confidence", 0.0) < 0.6
        ):
            threshold = max(threshold, 0.55)
        if signals.get("verification_issues") and confidence < 0.6:
            return True
        return confidence < threshold

    def _clip(self, value: Any) -> float:
        try:
            return max(0.0, min(1.0, float(value)))
        except (TypeError, ValueError):
            # Returning 0.0 here made a malformed signal indistinguishable
            # from a measured zero, inside the number the pipeline reports as
            # its confidence.
            self._degrade(
                f"a confidence signal was not a number ({value!r}), so it was "
                f"read as zero in the weighted sum"
            )
            return 0.0

    def _normalized_evidence_score(self, signals: dict[str, Any]) -> float:
        evidence_count = int(signals.get("evidence_count", 0) or 0)
        top_evidence_score = self._clip(signals.get("top_evidence_score", 0.0))
        count_factor = min(1.0, evidence_count / 3.0)
        return (0.65 * top_evidence_score) + (0.35 * count_factor)
