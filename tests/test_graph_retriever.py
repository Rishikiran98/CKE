"""Tests for graph retrieval traversal, scoring, and evidence selection."""

from cke.graph_engine.graph_engine import KnowledgeGraphEngine
from cke.models import Statement
from cke.retrieval.graph_retriever import GraphRetriever
from cke.router.query_plan import QueryPlan


def _build_graph() -> KnowledgeGraphEngine:
    graph = KnowledgeGraphEngine()
    graph.add_statements(
        [
            Statement("Redis", "supports", "PubSub", confidence=0.95),
            Statement("PubSub", "implemented_via", "RESP", confidence=0.9),
            Statement("RESP", "used_by", "Redis", confidence=0.7),
            Statement("Redis", "category", "Database", confidence=0.8),
            Statement("Kafka", "supports", "PubSub", confidence=0.85),
        ]
    )
    return graph


def test_bfs_traversal_collects_paths():
    retriever = GraphRetriever(_build_graph())
    paths = retriever._bfs_traversal(["Redis"], max_depth=2, max_nodes=50)
    triples = {
        (edge.subject, edge.relation, edge.object) for path in paths for edge in path
    }
    assert ("Redis", "supports", "PubSub") in triples
    assert ("PubSub", "implemented_via", "RESP") in triples


def test_path_scoring_penalizes_long_or_repeated_entities():
    retriever = GraphRetriever(_build_graph())
    short_path = [Statement("Redis", "supports", "PubSub", confidence=0.9)]
    longer_repeating_path = [
        Statement("Redis", "supports", "PubSub", confidence=0.9),
        Statement("PubSub", "implemented_via", "RESP", confidence=0.9),
        Statement("RESP", "used_by", "Redis", confidence=0.9),
    ]

    assert retriever._score_path(short_path) > retriever._score_path(
        longer_repeating_path
    )


def test_evidence_selection_returns_ranked_assertions():
    retriever = GraphRetriever(_build_graph())
    plan = QueryPlan(seed_entities=["Redis"], domains=["databases"], intent="multi-hop")
    result = retriever.retrieve(plan, mode="beam", beam_width=3)

    assert {"evidence", "paths", "entities"}.issubset(result.keys())
    assert result["evidence"]
    assert 1 <= len(result["evidence"]) <= 20
    top = result["evidence"][0]
    assert {"subject", "relation", "object", "trust_score"}.issubset(top.keys())
