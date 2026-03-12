"""Tests for hybrid graph+dense retrieval routing."""

from cke.graph_engine.graph_engine import KnowledgeGraphEngine
from cke.models import Statement
from cke.retrieval.hybrid_retrieval import (
    DENSE_WEIGHT,
    GRAPH_WEIGHT,
    HybridRetrievalMerger,
)
from cke.retrieval.rag_baseline import RAGRetriever
from cke.retrieval.retrieval_router import RetrievalRouter
from cke.retrieval.retriever import GraphRetriever


class StubDenseRetriever(RAGRetriever):
    def __init__(self, payload: list[dict[str, str | float]]) -> None:
        self.payload = payload

    def retrieve(self, query: str, k: int = 5) -> list[dict[str, str | float]]:
        return self.payload[:k]


def _build_graph() -> KnowledgeGraphEngine:
    graph = KnowledgeGraphEngine()
    graph.add_statements(
        [
            Statement("Redis", "supports", "PubSub", confidence=0.94),
            Statement("PubSub", "implemented_via", "RESP", confidence=0.9),
        ]
    )
    return graph


def test_graph_retrieval_succeeds_without_dense_fallback():
    router = RetrievalRouter(
        graph_retriever=GraphRetriever(_build_graph()),
        dense_retriever=StubDenseRetriever(
            [{"text": "Dense fallback chunk", "score": 0.5}]
        ),
        evidence_threshold=1,
        dense_top_k=3,
    )

    pack = router.retrieve("What protocol does Redis pubsub use?", max_depth=2)

    assert len(pack.graph_statements) >= 1
    assert pack.fallback_chunks == []
    metrics = router.metrics_snapshot()
    assert metrics["fallback_triggered"] == 0
    assert metrics["fallback_rate"] == 0.0


def test_graph_retrieval_failure_activates_dense_fallback():
    router = RetrievalRouter(
        graph_retriever=GraphRetriever(_build_graph()),
        dense_retriever=StubDenseRetriever(
            [
                {"text": "Redis supports pub/sub messaging", "score": 0.91},
                {"text": "RESP is a protocol used by Redis", "score": 0.9},
                {"text": "PubSub can broadcast to subscribers", "score": 0.88},
                {"text": "Extra chunk should be truncated", "score": 0.5},
            ]
        ),
        evidence_threshold=2,
        dense_top_k=3,
    )

    pack = router.retrieve("What does Kafka use for compaction?")

    assert pack.graph_statements == []
    assert len(pack.fallback_chunks) == 3
    metrics = router.metrics_snapshot()
    assert metrics["fallback_triggered"] == 1
    assert metrics["fallback_rate"] == 1.0


def test_merge_removes_duplicates_and_assigns_weights():
    merger = HybridRetrievalMerger()
    graph_statements = [
        Statement("Redis", "supports", "PubSub"),
        Statement("Redis", "supports", "PubSub"),
    ]
    dense_chunks = [
        "Redis supports PubSub",
        "redis supports pubsub",
        "RESP is used by Redis",
    ]

    pack, weighted = merger.merge(graph_statements, dense_chunks)

    assert len(pack.graph_statements) == 1
    assert pack.fallback_chunks == ["RESP is used by Redis"]
    assert sum(item.source == "graph" for item in weighted) == 1
    assert sum(item.source == "dense" for item in weighted) == 1
    assert {item.weight for item in weighted} == {GRAPH_WEIGHT, DENSE_WEIGHT}
