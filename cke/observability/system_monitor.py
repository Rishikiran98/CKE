"""Runtime observability primitives for CKE."""

from __future__ import annotations

import time
from dataclasses import dataclass, field


@dataclass
class SystemMonitor:
    query_latency_ms: float = 0.0
    retrieval_steps: int = 0
    graph_nodes_traversed: int = 0
    assertions_generated: int = 0
    conflicts_detected: int = 0
    _start: float = field(default=0.0, init=False, repr=False)

    def start_query(self) -> None:
        self._start = time.perf_counter()

    def end_query(self) -> None:
        if self._start:
            self.query_latency_ms = (time.perf_counter() - self._start) * 1000

    def record_retrieval(self, steps: int, nodes_traversed: int = 0) -> None:
        self.retrieval_steps += int(steps)
        self.graph_nodes_traversed += int(nodes_traversed)

    def record_assertions(self, count: int) -> None:
        self.assertions_generated += int(count)

    def record_conflicts(self, count: int) -> None:
        self.conflicts_detected += int(count)

    def snapshot(self) -> dict[str, float | int]:
        return {
            "query_latency_ms": round(self.query_latency_ms, 3),
            "retrieval_steps": self.retrieval_steps,
            "graph_nodes_traversed": self.graph_nodes_traversed,
            "assertions_added": self.assertions_generated,
            "conflicts_resolved": self.conflicts_detected,
        }
