"""Tests for the HybridEvidenceRetriever adapter."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from cke.models import Statement
from cke.pipeline.types import ResolvedEntity
from cke.retrieval.hybrid_retrieval import EvidencePack
from cke.retrieval.hybrid_evidence_retriever import HybridEvidenceRetriever
from cke.retrieval.retrieval_router import RetrievalRouter


def _make_statement(subject: str, relation: str, obj: str, **kw) -> Statement:
    return Statement(subject=subject, relation=relation, object=obj, **kw)


def _make_entity(name: str, eid: str = "") -> ResolvedEntity:
    return ResolvedEntity(
        surface_form=name,
        canonical_name=name,
        entity_id=eid or name.lower().replace(" ", "_"),
        link_confidence=0.9,
    )


def _mock_router(graph_stmts: list[Statement], chunks: list[str]) -> RetrievalRouter:
    router = MagicMock(spec=RetrievalRouter)
    router.retrieve.return_value = EvidencePack(
        graph_statements=graph_stmts,
        fallback_chunks=chunks,
    )
    router.metrics_snapshot.return_value = {
        "total_queries": 1,
        "fallback_triggered": 1 if chunks else 0,
        "fallback_rate": 1.0 if chunks else 0.0,
    }
    return router


class TestGraphStatementConversion:
    def test_converts_graph_statements_to_chunks_and_facts(self):
        stmts = [
            _make_statement("Alice", "born_in", "Paris", confidence=0.9),
            _make_statement("Bob", "works_at", "CERN", confidence=0.8),
        ]
        router = _mock_router(stmts, [])
        adapter = HybridEvidenceRetriever(router)

        chunks, facts = adapter.retrieve("Where was Alice born?")

        assert len(facts) == 2
        assert all(f.metadata.get("retriever") == "hybrid_graph" for f in facts)
        assert facts[0].statement.subject in {"Alice", "Bob"}
        assert len(chunks) <= 5  # default top_k

    def test_graph_statement_chunk_ids(self):
        stmts = [_make_statement("X", "rel", "Y", chunk_id="chunk_42")]
        router = _mock_router(stmts, [])
        adapter = HybridEvidenceRetriever(router)

        chunks, facts = adapter.retrieve("query")

        assert chunks[0].chunk_id == "chunk_42"
        assert facts[0].chunk_id == "chunk_42"

    def test_graph_statement_without_chunk_id_gets_generated(self):
        stmts = [_make_statement("X", "rel", "Y")]
        router = _mock_router(stmts, [])
        adapter = HybridEvidenceRetriever(router)

        chunks, facts = adapter.retrieve("query")

        assert chunks[0].chunk_id.startswith("hybrid_graph::")


class TestDenseFallbackConversion:
    def test_converts_fallback_chunks_to_synthetic_facts(self):
        chunks_text = ["Alice was born in Paris.", "Bob works at CERN."]
        router = _mock_router([], chunks_text)
        adapter = HybridEvidenceRetriever(router)

        chunks, facts = adapter.retrieve("Where was Alice born?")

        dense_facts = [f for f in facts if f.metadata.get("retriever") == "hybrid_dense"]
        assert len(dense_facts) == 2
        assert all(f.metadata.get("synthetic") is True for f in dense_facts)
        assert all(f.statement.relation == "dense_fallback" for f in dense_facts)
        assert dense_facts[0].statement.object in chunks_text

    def test_dense_chunk_ids(self):
        router = _mock_router([], ["chunk text"])
        adapter = HybridEvidenceRetriever(router)

        chunks, facts = adapter.retrieve("query")

        assert chunks[0].chunk_id.startswith("hybrid_dense::")
        assert chunks[0].source == "dense_fallback"


class TestMergedOutput:
    def test_graph_and_dense_combined(self):
        stmts = [_make_statement("Alice", "born_in", "Paris")]
        dense = ["Bob works at CERN."]
        router = _mock_router(stmts, dense)
        adapter = HybridEvidenceRetriever(router)

        chunks, facts = adapter.retrieve("query")

        sources = {f.metadata["retriever"] for f in facts}
        assert "hybrid_graph" in sources
        assert "hybrid_dense" in sources

    def test_respects_top_k_for_chunks(self):
        stmts = [_make_statement(f"E{i}", "rel", f"O{i}") for i in range(10)]
        router = _mock_router(stmts, [])
        adapter = HybridEvidenceRetriever(router)

        chunks, facts = adapter.retrieve("query", top_k=3)

        assert len(chunks) == 3
        # facts may be more than top_k (up to top_k * 3)
        assert len(facts) <= 10


class TestEntityScoring:
    def test_entity_match_boosts_score(self):
        stmts = [
            _make_statement("Alice", "born_in", "Paris"),
            _make_statement("Zulu", "capital_of", "Delta"),
        ]
        router = _mock_router(stmts, [])
        adapter = HybridEvidenceRetriever(router)
        entities = [_make_entity("Alice")]

        chunks, facts = adapter.retrieve(
            "Where was Alice born?",
            resolved_entities=entities,
        )

        # Alice-related fact should score higher
        alice_facts = [f for f in facts if f.statement.subject == "Alice"]
        other_facts = [f for f in facts if f.statement.subject != "Alice"]
        assert alice_facts[0].retrieval_score > other_facts[0].retrieval_score

    def test_relation_match_boosts_score(self):
        stmts = [
            _make_statement("Alice", "born_in", "Paris"),
            _make_statement("Alice", "works_at", "CERN"),
        ]
        router = _mock_router(stmts, [])
        adapter = HybridEvidenceRetriever(router)

        chunks, facts = adapter.retrieve(
            "Where was Alice born?",
            target_relations=["born_in"],
        )

        born_facts = [f for f in facts if f.statement.relation == "born_in"]
        work_facts = [f for f in facts if f.statement.relation == "works_at"]
        assert born_facts[0].retrieval_score > work_facts[0].retrieval_score


class TestMetricsSnapshot:
    def test_delegates_to_router(self):
        router = _mock_router([], ["text"])
        adapter = HybridEvidenceRetriever(router)
        adapter.retrieve("query")

        metrics = adapter.metrics_snapshot()

        assert metrics["fallback_triggered"] == 1
        router.metrics_snapshot.assert_called_once()
