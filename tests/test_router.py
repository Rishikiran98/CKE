"""Tests for query routing and planning."""

from cke.graph_engine.graph_engine import KnowledgeGraphEngine
from cke.models import Statement
from cke.router.entity_linker import EntityLinker
from cke.router.intent_classifier import IntentClassifier
from cke.router.query_router import QueryRouter


def test_entity_extraction_from_query_uses_graph_clues():
    graph = KnowledgeGraphEngine()
    graph.add_statements(
        [
            Statement("Redis", "supports", "PubSub"),
            Statement("Postgres", "supports", "SQL"),
        ]
    )

    linker = EntityLinker(graph)
    entities = linker.extract_entities("Which database supports pubsub?")
    assert "Redis" in entities


def test_intent_detection_multi_hop():
    classifier = IntentClassifier()
    assert classifier.classify("Which system uses Redis?") == "multi-hop"


def test_query_router_builds_query_plan():
    graph = KnowledgeGraphEngine()
    graph.add_statement("Redis", "supports", "PubSub")

    router = QueryRouter(graph)
    plan = router.route("Which database supports pubsub?")

    assert plan.intent == "multi-hop"
    assert "databases" in plan.domains
    assert plan.max_depth == 3
    assert "Redis" in plan.seed_entities


def test_query_router_uses_question_entities_when_linker_has_no_match():
    router = QueryRouter()
    plan = router.route("How are Scott Derrickson and Ed Wood connected?")

    assert "Scott Derrickson" in plan.seed_entities
    assert "Ed Wood" in plan.seed_entities
    assert plan.max_depth == 3


def test_query_router_adapts_depth_for_complex_multi_hop_query():
    router = QueryRouter()
    query = "How is Redis connected to RESP via PubSub and then through API services?"

    plan = router.route(query)

    assert plan.intent == "multi-hop"
    assert plan.max_depth >= 4


def test_query_router_respects_manual_depth_when_query_is_simple():
    router = QueryRouter()

    plan = router.route("What is Redis?", max_depth=6)

    assert plan.max_depth <= 4


def test_query_router_populates_decomposition_steps():
    router = QueryRouter()
    plan = router.route("What nationality is the director of Top Gun?")

    assert plan.decomposition
    relation_values = {
        step["value"] for step in plan.decomposition if step["type"] == "relation"
    }
    assert "directed_by" in relation_values
    assert "nationality" in relation_values
