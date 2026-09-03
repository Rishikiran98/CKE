"""Default-runtime smoke coverage for entity cleanup and seeded graph QA."""

from cke.graph_engine.graph_engine import KnowledgeGraphEngine
from cke.pipeline.query_orchestrator import QueryOrchestrator
from cke.router.query_router import QueryRouter


def _build_seeded_graph() -> KnowledgeGraphEngine:
    graph = KnowledgeGraphEngine()
    graph.add_statement("Albert Einstein", "nationality", "German")
    graph.add_statement("Scott Derrickson", "nationality", "American")
    graph.add_statement("Ed Wood", "nationality", "American")
    graph.add_statement("Person X", "child", "A")
    graph.add_statement("Person X", "child", "B")
    graph.add_statement("Person X", "child", "C")
    graph.add_statement("Christopher Nolan", "directed", "Inception")
    graph.add_statement("A", "located_in", "B")
    graph.add_statement("B", "located_in", "C")
    return graph


def test_query_router_entity_cleanup_avoids_scaffolding_and_single_letter_overmatch():
    graph = _build_seeded_graph()
    router = QueryRouter(graph_engine=graph)

    assert router.detect_entities("What is the nationality of Albert Einstein?") == [
        "Albert Einstein"
    ]
    assert set(
        router.detect_entities(
            "Were Scott Derrickson and Ed Wood of the same nationality?"
        )
    ) == {"Scott Derrickson", "Ed Wood"}
    assert router.detect_entities("How many children did Person X have?") == [
        "Person X"
    ]
    assert set(router.detect_entities("Did Christopher Nolan direct Inception?")) == {
        "Christopher Nolan",
        "Inception",
    }
    assert router.detect_entities("Where is A located in?") == ["A"]


def test_default_orchestrator_runtime_answers_seeded_smoke_queries():
    graph = _build_seeded_graph()
    router = QueryRouter(graph_engine=graph)
    orchestrator = QueryOrchestrator(graph_engine=graph, router=router)

    assert orchestrator.retriever is not None
    assert orchestrator.assembler is not None

    assert (
        orchestrator.answer("What is the nationality of Albert Einstein?").answer
        == "German"
    )
    assert (
        orchestrator.answer(
            "Were Scott Derrickson and Ed Wood of the same nationality?"
        ).answer
        == "yes"
    )
    assert orchestrator.answer("How many children did Person X have?").answer == "3"
    assert (
        orchestrator.answer("Did Christopher Nolan direct Inception?").answer == "yes"
    )
    assert orchestrator.answer("Where is A located in?").answer == "C"
