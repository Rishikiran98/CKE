"""Domain convergence state classification."""

from __future__ import annotations

from collections import deque


class ConvergenceEngine:
    """Track drift windows and classify domain evolution state."""

    def __init__(self, epsilon: float = 0.05, window_size: int = 5) -> None:
        self.epsilon = epsilon
        self.window_size = window_size
        self._domain_drifts: dict[str, deque[float]] = {}

    def update(self, domain_name: str, smoothed_delta: float) -> str:
        history = self._domain_drifts.setdefault(
            domain_name, deque(maxlen=self.window_size)
        )
        history.append(smoothed_delta)
        return self.get_state(domain_name)

    def get_state(self, domain_name: str) -> str:
        history = self._domain_drifts.get(domain_name, deque())
        if len(history) >= self.window_size and all(
            value < self.epsilon for value in history
        ):
            return "stable"

        if history and history[-1] >= max(self.epsilon * 2, 0.25):
            return "volatile"

        return "evolving"
