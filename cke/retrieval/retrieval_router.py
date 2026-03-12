"""Router that executes graph retrieval with dense fallback."""

from __future__ import annotations

import logging
from dataclasses import dataclass

from cke.models import Statement
from cke.retrieval.hybrid_retrieval import EvidencePack, HybridRetrievalMerger
from cke.retrieval.rag_baseline import RAGRetriever
from cke.retrieval.retriever import GraphRetriever

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class HybridRetrievalMetrics:
    """Counters for fallback behavior in hybrid retrieval."""

    total_queries: int = 0
    fallback_triggered: int = 0

    @property
    def fallback_rate(self) -> float:
        if self.total_queries == 0:
            return 0.0
        return self.fallback_triggered / self.total_queries


class RetrievalRouter:
    """Hybrid retrieval coordinator with graph-first execution."""

    def __init__(
        self,
        graph_retriever: GraphRetriever,
        dense_retriever: RAGRetriever,
        evidence_threshold: int = 2,
        dense_top_k: int = 3,
    ) -> None:
        self.graph_retriever = graph_retriever
        self.dense_retriever = dense_retriever
        self.evidence_threshold = evidence_threshold
        self.dense_top_k = dense_top_k
        self.metrics = HybridRetrievalMetrics()
        self.merger = HybridRetrievalMerger()

    def retrieve(self, query: str, max_depth: int = 2) -> EvidencePack:
        self.metrics.total_queries += 1

        graph_statements = self.graph_retriever.retrieve(query, max_depth=max_depth)
        dense_chunks: list[str] = []
        fallback_used = len(graph_statements) < self.evidence_threshold

        if fallback_used:
            self.metrics.fallback_triggered += 1
            dense_chunks = self._dense_fallback(query)

        evidence_pack, _ = self.merger.merge(graph_statements, dense_chunks)

        logger.info(
            "hybrid_retrieval query=%r fallback_triggered=%s fallback_rate=%.3f",
            query,
            fallback_used,
            self.metrics.fallback_rate,
        )
        return evidence_pack

    def _dense_fallback(self, query: str) -> list[str]:
        dense_results = self.dense_retriever.retrieve(query, k=self.dense_top_k)
        return [str(item.get("text", "")) for item in dense_results if item.get("text")]

    def metrics_snapshot(self) -> dict[str, int | float]:
        return {
            "total_queries": self.metrics.total_queries,
            "fallback_triggered": self.metrics.fallback_triggered,
            "fallback_rate": self.metrics.fallback_rate,
        }

    @staticmethod
    def statement_count(statements: list[Statement]) -> int:
        return len(statements)
