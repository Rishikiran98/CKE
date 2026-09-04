"""Strict reaches the components the README says it reaches.

The README states that every benchmark, evaluation and experiment entry point
constructs its components with strict=True, excepting only the API server and
one deprecation shim. Three entry points did not, and the conversation
subsystem had no strict parameter at all — so it ran on the hashed embedder by
construction whatever the caller asked for.
"""

from __future__ import annotations

import inspect

import pytest

from cke.diagnostics import DegradedComponentError, clear_runtime_state


@pytest.fixture(autouse=True)
def _clean_runtime_state():
    clear_runtime_state()
    yield
    clear_runtime_state()


# ---------------------------------------------------------------------------
# The check that could not tell a healthy component from a degraded one
# ---------------------------------------------------------------------------


def test_a_component_that_cannot_report_degradation_is_refused():
    """getattr(supplied, "degraded", False) returned the default for anything
    that set self.strict by hand, so an already degraded instance passed."""
    from cke.diagnostics import require_strict_component

    class CannotSay:
        strict = True  # claims strictness, cannot report degradation

    with pytest.raises(DegradedComponentError, match="cannot report whether"):
        require_strict_component("Pipeline", CannotSay(), "reasoner", True)


def test_the_two_classes_that_used_to_slip_through_can_now_answer():
    from cke.retrieval.retriever import GraphRetriever
    from cke.router.query_router import QueryRouter

    for component in (QueryRouter(strict=True), GraphRetriever(None, strict=True)):
        assert component.strict is True
        assert component.degraded is False
        assert hasattr(component, "degraded_reason")


# ---------------------------------------------------------------------------
# The orchestrator holds its injectables to the same bar
# ---------------------------------------------------------------------------


def test_a_strict_orchestrator_refuses_a_non_strict_injected_component():
    """It accepted eight prebuilt collaborators and checked none of them,
    while reporting itself strict."""
    from cke.graph_engine.graph_engine import KnowledgeGraphEngine
    from cke.pipeline.query_orchestrator import QueryOrchestrator
    from cke.reasoning.path_reasoner import PathReasoner
    from cke.router.query_router import QueryRouter

    with pytest.raises(DegradedComponentError, match="does not declare"):
        QueryOrchestrator(
            graph_engine=KnowledgeGraphEngine(strict=True),
            router=QueryRouter(strict=True),
            reasoner=PathReasoner(strict=False),
            strict=True,
        )


def test_a_strict_orchestrator_accepts_a_strict_one():
    from cke.graph_engine.graph_engine import KnowledgeGraphEngine
    from cke.pipeline.query_orchestrator import QueryOrchestrator
    from cke.reasoning.path_reasoner import PathReasoner
    from cke.router.query_router import QueryRouter

    orchestrator = QueryOrchestrator(
        graph_engine=KnowledgeGraphEngine(strict=True),
        router=QueryRouter(strict=True),
        reasoner=PathReasoner(strict=True),
        strict=True,
    )

    assert orchestrator.strict is True
    assert orchestrator.reasoner.strict is True


# ---------------------------------------------------------------------------
# The conversation subsystem can be strict at all
# ---------------------------------------------------------------------------


def test_the_conversational_orchestrator_takes_strict():
    from cke.pipeline.conversational_orchestrator import ConversationalOrchestrator

    assert "strict" in inspect.signature(ConversationalOrchestrator.__init__).parameters


def test_strict_reaches_the_ingestion_pipeline_and_the_embedder():
    """Two components inside the chain dropped it: the store built its
    ingestion pipeline without strict, and the retriever built its candidate
    generator without strict, so the embedder was never strict either."""
    from cke.conversation.memory import ConversationalMemoryStore
    from cke.conversation.retriever import ConversationalRetriever

    store = ConversationalMemoryStore(strict=True)
    assert store.ingestion_pipeline.strict is True

    retriever = ConversationalRetriever(store, strict=True)
    assert retriever.candidate_generator.strict is True
    assert retriever.candidate_generator.embedding_model.strict is True


# ---------------------------------------------------------------------------
# The entry points
# ---------------------------------------------------------------------------


def test_every_entry_point_passes_strict_to_what_it_builds():
    """A source check, because these are constructor call sites rather than
    behaviour — but a narrow one: each name below is a component with a
    degradation path that the driver used to build without strict."""
    from pathlib import Path

    root = Path(__file__).resolve().parents[1]
    experiment = (root / "cke" / "experiments" / "run_experiment.py").read_text(
        encoding="utf-8"
    )
    assert "GraphRetriever(graph, strict=strict)" in experiment


def test_the_monitor_declares_a_metric_it_could_not_record():
    """It warned and set no flag, so a snapshot could not say that one of its
    figures had stopped recording."""
    from cke.observability.system_monitor import SystemMonitor

    monitor = SystemMonitor()
    monitor.record_retrieval("not a number")  # type: ignore[arg-type]

    assert monitor.degraded is True
    assert "undercounts" in monitor.degraded_reason

    with pytest.raises(DegradedComponentError):
        SystemMonitor(strict=True).record_retrieval(
            "not a number"  # type: ignore[arg-type]
        )


def test_an_experiment_over_no_data_refuses_rather_than_scoring_zero():
    """max(len(dataset), 1) reported exact_match 0.0 for a run that scored
    nothing, which reads as a measured failure."""
    from cke.evaluation.experiment_runner import ExperimentRunner
    from cke.retrieval.rag_baseline import RAGRetriever

    runner = ExperimentRunner(retriever=RAGRetriever())

    with pytest.raises(ValueError, match="nothing to score"):
        runner.run([])
