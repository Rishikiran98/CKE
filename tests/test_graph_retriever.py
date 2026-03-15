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
            Statement("pubsub", "implemented_via", "resp", confidence=0.9),
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
    assert ("PubSub", "implemented_via", "resp") in triples


def test_path_scoring_penalizes_long_or_repeated_entities():
    retriever = GraphRetriever(_build_graph())
    short_path = [Statement("redis", "supports", "pubsub", confidence=0.9)]
    longer_repeating_path = [
        Statement("redis", "supports", "pubsub", confidence=0.9),
        Statement("pubsub", "implemented_via", "resp", confidence=0.9),
        Statement("resp", "used_by", "redis", confidence=0.9),
    ]

    assert retriever._score_path(short_path) > retriever._score_path(
        longer_repeating_path
    )


def test_evidence_selection_returns_ranked_assertions():
    retriever = GraphRetriever(_build_graph())
    plan = QueryPlan(
        query_text="How does Redis do pubsub?",
        seed_entities=["Redis"],
        domains=["databases"],
        intent="multi-hop",
    )
    result = retriever.retrieve(plan, mode="beam", beam_width=3)

    assert {"evidence", "paths", "entities"}.issubset(result.keys())
    assert result["evidence"]
    assert 1 <= len(result["evidence"]) <= 20
    top = result["evidence"][0]
    assert {"subject", "relation", "object", "trust_score"}.issubset(top.keys())


def test_query_aware_ranking_prefers_relevant_edges():
    graph = KnowledgeGraphEngine()
    graph.add_statements(
        [
            Statement("Redis", "supports", "PubSub", confidence=0.9),
            Statement("Redis", "created_by", "Salvatore Sanfilippo", confidence=0.99),
        ]
    )
    retriever = GraphRetriever(graph)
    plan = QueryPlan(
        query_text="How does redis pubsub work?",
        seed_entities=["redis"],
        intent="definition",
        max_results=5,
    )

    result = retriever.retrieve(plan)

    assert result["evidence"][0]["relation"] == "supports"


def test_astar_traversal_returns_ranked_paths():
    retriever = GraphRetriever(_build_graph())
    plan = QueryPlan(
        query_text="How is Redis connected to RESP via pubsub?",
        seed_entities=["Redis"],
        intent="multi-hop",
        decomposition=[
            {"type": "relation", "value": "supports"},
            {"type": "relation", "value": "implemented_via"},
        ],
        max_depth=3,
    )

    result = retriever.retrieve(plan, mode="astar")

    assert result["paths"]
    assert "evidence_graph" in result
    assert result["evidence_graph"]["paths"]


def test_path_ranking_prefers_relation_match():
    graph = KnowledgeGraphEngine()
    graph.add_statements(
        [
            Statement("Film", "directed_by", "Director", confidence=0.75),
            Statement("Film", "genre", "Action", confidence=0.98),
        ]
    )
    retriever = GraphRetriever(graph)
    plan = QueryPlan(
        query_text="Who directed the film?",
        seed_entities=["Film"],
        intent="multi-hop",
        decomposition=[{"type": "relation", "value": "directed_by"}],
        max_depth=1,
        max_results=5,
    )

    result = retriever.retrieve(plan, mode="astar")

    assert result["evidence"][0]["relation"] == "directed_by"
