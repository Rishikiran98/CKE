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
