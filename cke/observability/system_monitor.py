from __future__ import annotations

import time
import logging
from collections import defaultdict
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)

def _safe_metric(func):
    """Decorator to guarantee metrics emission never throws an exception."""
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            logger.debug(f"Monitor exception swallowed: {e}")
    return wrapper

@dataclass
class SystemMonitor:
    query_latency_ms: float = 0.0
    retrieval_steps: int = 0
    graph_nodes_traversed: int = 0
    assertions_generated: int = 0
    conflicts_detected: int = 0
    bridge_nodes_found: int = 0
    neighborhood_nodes_expanded: int = 0
    _start: float = field(default=0.0, init=False, repr=False)
    
    # New Latency Metrics
    ingestion_latency_ms: float = 0.0
    extraction_latency_ms: float = 0.0
    validation_latency_ms: float = 0.0
    consolidation_latency_ms: float = 0.0
    answering_latency_ms: float = 0.0
    retrieval_latency_ms: float = 0.0
    
    # Error Taxonomies & Quality Metrics
    extractor_exceptions: int = 0
    duplicate_candidates: int = 0
    canonical_updates: int = 0
    canonical_supersedes: int = 0
    unresolved_references: int = 0
    stale_rejects: int = 0
    
    # Distributions & Counts
    validation_rejections_by_reason: dict[str, int] = field(default_factory=lambda: defaultdict(int))
    candidate_count_by_kind: dict[str, int] = field(default_factory=lambda: defaultdict(int))
    canonical_write_count_by_kind: dict[str, int] = field(default_factory=lambda: defaultdict(int))
    abstention_count_by_reason: dict[str, int] = field(default_factory=lambda: defaultdict(int))
    confidence_buckets: dict[str, int] = field(default_factory=lambda: defaultdict(int))

    def start_query(self) -> None:
        self._start = time.perf_counter()

    @_safe_metric
    def end_query(self) -> None:
        if self._start:
            self.query_latency_ms = (time.perf_counter() - self._start) * 1000

    @_safe_metric
    def record_retrieval(self, steps: int, nodes_traversed: int = 0) -> None:
        self.retrieval_steps += int(steps)
        self.graph_nodes_traversed += int(nodes_traversed)

    @_safe_metric
    def record_assertions(self, count: int) -> None:
        self.assertions_generated += int(count)

    @_safe_metric
    def record_conflicts(self, count: int) -> None:
        self.conflicts_detected += int(count)

    @_safe_metric
    def record_bridge_nodes(self, count: int) -> None:
        self.bridge_nodes_found += int(count)

    @_safe_metric
    def record_neighborhood_expansion(self, count: int) -> None:
        self.neighborhood_nodes_expanded += int(count)

    @_safe_metric
    def record_latency(self, metric_name: str, start_time: float) -> None:
        elapsed_ms = (time.perf_counter() - start_time) * 1000
        setattr(self, metric_name, elapsed_ms)
        
    @_safe_metric
    def record_extractor_exception(self) -> None:
        self.extractor_exceptions += 1

    @_safe_metric
    def record_candidate(self, kind: str) -> None:
        self.candidate_count_by_kind[kind] += 1
        
    @_safe_metric
    def record_validation_rejection(self, reason: str) -> None:
        self.validation_rejections_by_reason[reason] += 1
        
    @_safe_metric
    def record_consolidation_outcome(self, decision: str, kind: str) -> None:
        if decision == "merge":
            self.duplicate_candidates += 1
        elif decision == "update":
            self.canonical_updates += 1
            self.canonical_supersedes += 1
            self.canonical_write_count_by_kind[kind] += 1
        elif decision == "accept":
            self.canonical_write_count_by_kind[kind] += 1

    @_safe_metric
    def record_answering(self, grounded: bool, confidence: float, abstained: bool, reason: str = "", unresolved_refs: bool = False) -> None:
        if unresolved_refs:
            self.unresolved_references += 1
        
        if abstained:
            self.abstention_count_by_reason[reason] += 1
            
        bucket = "high" if confidence >= 0.7 else ("medium" if confidence >= 0.4 else "low")
        self.confidence_buckets[bucket] += 1

    def snapshot(self) -> dict[str, float | int]:
        return {
            "query_latency_ms": round(self.query_latency_ms, 3),
            "ingestion_latency_ms": round(self.ingestion_latency_ms, 3),
            "answering_latency_ms": round(self.answering_latency_ms, 3),
            "retrieval_steps": self.retrieval_steps,
            "graph_nodes_traversed": self.graph_nodes_traversed,
            "assertions_added": self.assertions_generated,
            "conflicts_resolved": self.conflicts_detected,
            "bridge_nodes_found": self.bridge_nodes_found,
            "neighborhood_nodes_expanded": self.neighborhood_nodes_expanded,
            "extractor_exceptions": self.extractor_exceptions,
            "duplicate_candidates": self.duplicate_candidates,
            "canonical_updates": self.canonical_updates,
        }
