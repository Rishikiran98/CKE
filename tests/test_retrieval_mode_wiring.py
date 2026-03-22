"""Integration tests for retrieval mode wiring in QueryOrchestrator."""

from __future__ import annotations

from unittest.mock import MagicMock

from cke.pipeline.query_orchestrator import QueryOrchestrator
from cke.retrieval.default_evidence_retriever import DefaultEvidenceRetriever
from cke.retrieval.dense_evidence_retriever import DenseEvidenceRetriever
from cke.retrieval.hybrid_evidence_retriever import HybridEvidenceRetriever


def _stub_router():
    router = MagicMock()
    router.route.return_value = MagicMock(
        reasoning_route="factoid",
        route_confidence=0.8,
        target_relations=[],
        seed_entities=[],
        decomposition=[],
    )
    router.detect_entities.return_value = []
    return router


def _stub_graph_engine():
    engine = MagicMock()
    engine.get_neighbors.return_value = []
    engine.all_entities.return_value = []
    engine.edges_for_relation.return_value = []
    return engine


def _stub_dense_retriever():
    dense = MagicMock()
    dense.retrieve.return_value = [{"text": "some chunk", "score": 0.7, "doc_id": "d1"}]
    dense.build_index = MagicMock()
    return dense


class TestHybridModeDefault:
    def test_uses_hybrid_when_both_engines_available(self):
        orch = QueryOrchestrator(
            graph_engine=_stub_graph_engine(),
            router=_stub_router(),
            retrieval_mode="hybrid",
            dense_retriever=_stub_dense_retriever(),
        )
        assert isinstance(orch.retriever, HybridEvidenceRetriever)

    def test_stores_retrieval_mode(self):
        orch = QueryOrchestrator(
            graph_engine=_stub_graph_engine(),
            router=_stub_router(),
            retrieval_mode="hybrid",
            dense_retriever=_stub_dense_retriever(),
        )
        assert orch.retrieval_mode == "hybrid"


class TestGraphOnlyMode:
    def test_falls_back_to_graph_only_without_dense(self):
        orch = QueryOrchestrator(
            graph_engine=_stub_graph_engine(),
            router=_stub_router(),
            retrieval_mode="hybrid",
            # no dense_retriever → falls back to graph_only
        )
        assert isinstance(orch.retriever, DefaultEvidenceRetriever)

    def test_explicit_graph_only_mode(self):
        orch = QueryOrchestrator(
            graph_engine=_stub_graph_engine(),
            router=_stub_router(),
            retrieval_mode="graph_only",
            dense_retriever=_stub_dense_retriever(),
        )
        assert isinstance(orch.retriever, DefaultEvidenceRetriever)


class TestDenseOnlyMode:
    def test_dense_only_mode(self):
        orch = QueryOrchestrator(
            graph_engine=None,
            router=_stub_router(),
            retrieval_mode="dense_only",
            dense_retriever=_stub_dense_retriever(),
        )
        assert isinstance(orch.retriever, DenseEvidenceRetriever)


class TestExplicitRetrieverOverride:
    def test_explicit_retriever_takes_precedence(self):
        custom = MagicMock()
        orch = QueryOrchestrator(
            graph_engine=_stub_graph_engine(),
            router=_stub_router(),
            retriever=custom,
            retrieval_mode="hybrid",
            dense_retriever=_stub_dense_retriever(),
        )
        assert orch.retriever is custom

    def test_no_retriever_when_nothing_provided(self):
        orch = QueryOrchestrator(
            graph_engine=None,
            router=_stub_router(),
            retrieval_mode="graph_only",
        )
        assert orch.retriever is None


class TestEvidenceThreshold:
    def test_evidence_threshold_flows_to_router(self):
        orch = QueryOrchestrator(
            graph_engine=_stub_graph_engine(),
            router=_stub_router(),
            retrieval_mode="hybrid",
            dense_retriever=_stub_dense_retriever(),
            evidence_threshold=5,
        )
        assert isinstance(orch.retriever, HybridEvidenceRetriever)
        inner_router = orch.retriever.retrieval_router
        assert inner_router.evidence_threshold == 5


class TestDebugInfo:
    def test_retrieval_mode_in_debug_info(self):
        orch = QueryOrchestrator(
            graph_engine=_stub_graph_engine(),
            router=_stub_router(),
            retrieval_mode="hybrid",
            dense_retriever=_stub_dense_retriever(),
        )
        query_plan = MagicMock(
            reasoning_route="factoid",
            route_confidence=0.8,
        )
        from cke.pipeline.types import ReasoningContext

        ctx = ReasoningContext(
            query="test",
            query_plan=query_plan,
            trace_metadata={},
        )
        debug = orch._build_debug_info(ctx, query_plan, "trace123")
        assert debug["retrieval_mode"] == "hybrid"
