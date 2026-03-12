"""Tests for bridge and neighborhood retrieval modes."""

from cke.graph_engine.graph_engine import KnowledgeGraphEngine
from cke.models import Statement
from cke.retrieval.graph_retriever import GraphRetriever
from cke.router.query_plan import QueryPlan


def _build_bridge_graph() -> KnowledgeGraphEngine:
    graph = KnowledgeGraphEngine()
    graph.add_statements(
        [
            Statement(
                "Einstein",
                "affiliated_with",
                "Princeton University",
                confidence=0.97,
            ),
            Statement(
                "Princeton",
                "contains",
                "Princeton University",
                confidence=0.95,
            ),
            Statement(
                "Einstein",
                "won",
                "Nobel Prize",
                confidence=0.9,
            ),
            Statement(
                "Princeton University",
                "located_in",
                "Princeton",
                confidence=0.94,
            ),
        ]
    )
    return graph


def test_bridge_mode_finds_connector_for_comparison_intent():
    retriever = GraphRetriever(_build_bridge_graph())
    plan = QueryPlan(
        seed_entities=["Einstein", "Princeton"],
        intent="comparison",
        max_depth=2,
        max_results=12,
    )

    result = retriever.retrieve(plan)

    assert result["paths"], "Expected at least one bridge candidate path"
    assert "Princeton University" in result["entities"]

    flattened = [edge for path in result["paths"] for edge in path["assertions"]]
    assert any(
        edge["subject"] == "Einstein"
        and edge["relation"] == "affiliated_with"
        and edge["object"] == "Princeton University"
        for edge in flattened
    )


def test_neighborhood_mode_returns_local_context_only():
    retriever = GraphRetriever(_build_bridge_graph())
    plan = QueryPlan(
        seed_entities=["Einstein"],
        intent="definition",
        max_results=12,
    )

    result = retriever.retrieve(plan)

    assert result["evidence"]
    assert len(result["evidence"]) <= 12
    assert all(item["subject"] == "Einstein" for item in result["evidence"])
    assert all(
        {"id", "subject", "relation", "object", "trust"}.issubset(item)
        for item in result["evidence"]
    )
