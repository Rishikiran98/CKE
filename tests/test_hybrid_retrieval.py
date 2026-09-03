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


def _two_hop_graph() -> KnowledgeGraphEngine:
    """Redis and Kafka both exist; only Redis has anything said about it."""
    graph = KnowledgeGraphEngine()
    graph.add_statements(
        [
            Statement("Redis", "supports", "PubSub", confidence=0.94),
            Statement("Kafka", "category", "Broker", confidence=0.9),
        ]
    )
    return graph


def _dense_stub() -> StubDenseRetriever:
    return StubDenseRetriever([{"text": "dense chunk", "score": 0.5}])


def test_fallback_is_decided_by_coverage_not_by_count():
    """One statement about each seed is enough; many about one seed is not.

    The old rule fell back below two statements. Two statements about Redis
    would have satisfied it for a question about Redis and Kafka, and the
    graph would have been trusted on a relation it had said nothing about.
    """
    graph = KnowledgeGraphEngine()
    graph.add_statements(
        [
            Statement("Redis", "supports", "PubSub", confidence=0.9),
            Statement("Redis", "category", "Database", confidence=0.9),
            Statement("Redis", "written_in", "C", confidence=0.9),
            Statement("Kafka", "category", "Broker", confidence=0.9),
        ]
    )

    class OneSided(GraphRetriever):
        def retrieve(self, query, max_depth=2):
            return [
                s for s in super().retrieve(query, max_depth) if s.subject == "Redis"
            ]

    router = RetrievalRouter(
        graph_retriever=OneSided(graph), dense_retriever=_dense_stub(), dense_top_k=1
    )
    pack = router.retrieve("Do Redis and Kafka both support PubSub?")

    assert len(pack.graph_statements) >= 2, "count alone would have sufficed"
    assert pack.fallback_chunks == ["dense chunk"]


def test_no_fallback_when_every_seed_is_covered():
    router = RetrievalRouter(
        graph_retriever=GraphRetriever(_two_hop_graph()),
        dense_retriever=_dense_stub(),
        dense_top_k=1,
    )
    pack = router.retrieve("Do Redis and Kafka both support PubSub?")

    assert pack.fallback_chunks == []
    assert router.metrics_snapshot()["fallback_triggered"] == 0


def test_fallback_when_the_question_names_no_known_entity():
    """Nothing to judge sufficiency by, so the dense retriever is consulted.

    The seeds are forced empty rather than left to the entity detector, which
    finds *something* in almost any question. An earlier version of this test
    passed because the detector returned an uncovered seed, not because the
    no-seed branch ran, and a mutation deleting that branch survived it.
    """

    class Seedless(GraphRetriever):
        def seed_entities(self, query):
            return []

        def retrieve(self, query, max_depth=2):
            return [Statement("Redis", "supports", "PubSub", confidence=0.9)]

    router = RetrievalRouter(
        graph_retriever=Seedless(_two_hop_graph()),
        dense_retriever=_dense_stub(),
        dense_top_k=1,
    )
    pack = router.retrieve("What year did the treaty get signed?")

    assert pack.graph_statements, "the graph did return something"
    assert pack.fallback_chunks == ["dense chunk"]


def test_the_router_carries_no_count_threshold():
    router = RetrievalRouter(
        graph_retriever=GraphRetriever(_two_hop_graph()), dense_retriever=_dense_stub()
    )
    assert not hasattr(router, "evidence_threshold")
