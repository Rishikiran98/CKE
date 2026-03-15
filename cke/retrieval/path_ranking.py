"""Path ranking model for graph retrieval."""

from __future__ import annotations

from dataclasses import dataclass

from cke.models import Statement


@dataclass(slots=True)
class PathFeatures:
    token_overlap: float
    trust: float
    path_length: float
    relation_match: float


DEFAULT_PATH_WEIGHTS = PathFeatures(
    token_overlap=0.25,
    trust=0.20,
    path_length=0.10,
    relation_match=0.45,
)


class PathRankingModel:
    """Lightweight linear ranking model trained via configurable weights."""

    def __init__(self, weights: dict[str, float] | None = None):
        self.weights = self._resolve_weights(weights)

    def _resolve_weights(self, weights: dict[str, float] | None) -> PathFeatures:
        if not weights:
            return PathFeatures(
                token_overlap=DEFAULT_PATH_WEIGHTS.token_overlap,
                trust=DEFAULT_PATH_WEIGHTS.trust,
                path_length=DEFAULT_PATH_WEIGHTS.path_length,
                relation_match=DEFAULT_PATH_WEIGHTS.relation_match,
            )

        return PathFeatures(
            token_overlap=float(
                weights.get("token_overlap", DEFAULT_PATH_WEIGHTS.token_overlap)
            ),
            trust=float(weights.get("trust", DEFAULT_PATH_WEIGHTS.trust)),
            path_length=float(
                weights.get("path_length", DEFAULT_PATH_WEIGHTS.path_length)
            ),
            relation_match=float(
                weights.get("relation_match", DEFAULT_PATH_WEIGHTS.relation_match)
            ),
        )

    def rank_score(self, features: PathFeatures) -> float:
        return (
            self.weights.token_overlap * features.token_overlap
            + self.weights.trust * features.trust
            + self.weights.path_length * features.path_length
            + self.weights.relation_match * features.relation_match
        )

    def train(
        self,
        examples: list[tuple[PathFeatures, float]],
        lr: float = 0.05,
        epochs: int = 30,
    ) -> None:
        """Fit weights with simple gradient descent on MSE for ranking supervision."""
        if not examples:
            return

        keys = ["token_overlap", "trust", "path_length", "relation_match"]
        for _ in range(max(1, epochs)):
            grads = {k: 0.0 for k in keys}
            for features, label in examples:
                pred = self.rank_score(features)
                error = pred - label
                grads["token_overlap"] += error * features.token_overlap
                grads["trust"] += error * features.trust
                grads["path_length"] += error * features.path_length
                grads["relation_match"] += error * features.relation_match
            n = float(len(examples))
            for key in keys:
                value = getattr(self.weights, key) - lr * grads[key] / n
                setattr(self.weights, key, value)


def relation_match_score(
    path: list[Statement], decomposition: list[dict[str, str | float]]
) -> float:
    if not decomposition:
        return 0.0
    rel_targets = {
        str(step.get("value", ""))
        for step in decomposition
        if str(step.get("type", "")) == "relation"
    }
    if not rel_targets:
        return 0.0
    matched = sum(1 for edge in path if edge.relation in rel_targets)
    return matched / max(1, len(rel_targets))
