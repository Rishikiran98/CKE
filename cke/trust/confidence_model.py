"""Learned/learnable confidence estimator for graph assertions."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any


@dataclass(slots=True)
class ConfidenceFeatures:
    span_quality: float = 0.7
    relation_type: str = "related_to"
    entity_link_confidence: float = 0.7
    llm_logprob: float = -0.5
    source_reliability: float = 0.7


class ConfidenceModel:
    """Predict confidence scores from extraction features."""

    RELATION_PRIOR = {
        "born_in": 0.85,
        "located_in": 0.8,
        "member_of": 0.75,
        "directed": 0.75,
        "acted_in": 0.75,
    }

    def __init__(self) -> None:
        self.weights = {
            "bias": -0.15,
            "span_quality": 1.25,
            "entity_link_confidence": 1.1,
            "llm_logprob": 0.55,
            "source_reliability": 0.8,
            "relation_prior": 0.7,
        }

    def predict(self, assertion: Any) -> float:
        features = self._coerce_features(assertion)
        relation_prior = self.RELATION_PRIOR.get(features.relation_type, 0.6)
        linear = (
            self.weights["bias"]
            + self.weights["span_quality"] * self._clip(features.span_quality)
            + self.weights["entity_link_confidence"] * self._clip(features.entity_link_confidence)
            + self.weights["llm_logprob"] * self._clip_logprob(features.llm_logprob)
            + self.weights["source_reliability"] * self._clip(features.source_reliability)
            + self.weights["relation_prior"] * relation_prior
        )
        return self._sigmoid(linear)

    def _coerce_features(self, assertion: Any) -> ConfidenceFeatures:
        context = getattr(assertion, "context", {}) or {}
        relation = getattr(assertion, "relation", context.get("relation_type", "related_to"))
        return ConfidenceFeatures(
            span_quality=float(context.get("span_quality", 0.7)),
            relation_type=str(relation),
            entity_link_confidence=float(context.get("entity_link_confidence", 0.7)),
            llm_logprob=float(context.get("llm_logprob", -0.5)),
            source_reliability=float(context.get("source_reliability", 0.7)),
        )

    @staticmethod
    def _clip(value: float) -> float:
        return max(0.0, min(1.0, float(value)))

    @staticmethod
    def _clip_logprob(value: float) -> float:
        return max(-3.0, min(0.0, float(value))) / 3.0 + 1.0

    @staticmethod
    def _sigmoid(value: float) -> float:
        if value >= 0:
            exp_neg = math.exp(-value)
            return 1.0 / (1.0 + exp_neg)
        exp_pos = math.exp(value)
        return exp_pos / (1.0 + exp_pos)
