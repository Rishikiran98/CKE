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

from cke.diagnostics import DegradedComponentError

#: One sentence the rule extractor can read, so evaluate() has work to do.
CONTEXT = "Salvatore Sanfilippo developed Redis."


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


def test_a_strict_orchestrator_refuses_a_non_strict_injected_component(
    offline_embedder,
):
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


def test_a_strict_orchestrator_accepts_a_strict_one(offline_embedder):
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


def test_strict_reaches_the_ingestion_pipeline_and_the_embedder(offline_embedder):
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


def test_every_entry_point_passes_strict_to_what_it_builds(offline_embedder):
    """run_experiment.evaluate built its GraphRetriever non-strict.

    Its own signature says strict=True, so a run launched strict got a
    retriever that would accept its own fallbacks. I first wrote this as a
    source check for the literal call, with a note admitting it — which is
    the shape of test this branch exists to remove. Watch the construction
    instead: every GraphRetriever the function builds must be strict when it
    is.
    """
    from cke.experiments import run_experiment as module

    built = []
    real = module.GraphRetriever

    def _record(graph, *args, **kwargs):
        retriever = real(graph, *args, **kwargs)
        built.append(retriever)
        return retriever

    monkeypatch = pytest.MonkeyPatch()
    try:
        monkeypatch.setattr(module, "GraphRetriever", _record)
        module.evaluate(
            [module.QAItem(question="Who made Redis?", context=CONTEXT, answer="x")],
            strict=True,
        )
    finally:
        monkeypatch.undo()

    assert built, "evaluate() built no retriever, so nothing was checked"
    for retriever in built:
        assert retriever.strict is True
        assert retriever.degraded is False, retriever.degraded_reason


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
