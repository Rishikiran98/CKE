"""Knowledge drift computation across graph snapshots."""

from __future__ import annotations

from cke.graph.snapshot_manager import GraphSnapshot


class DriftMonitor:
    """Compute raw and smoothed drift scores."""

    def __init__(self, alpha: float = 0.2) -> None:
        self.alpha = alpha

    @staticmethod
    def compute_drift(current: GraphSnapshot, previous: GraphSnapshot) -> float:
        current_set = current.assertion_hash_set
        previous_set = previous.assertion_hash_set
        union = current_set.union(previous_set)
        if not union:
            return 0.0
        intersection = current_set.intersection(previous_set)
        return 1.0 - (len(intersection) / len(union))

    def smooth_drift(self, delta: float, previous_delta: float) -> float:
        return (self.alpha * delta) + ((1 - self.alpha) * previous_delta)
