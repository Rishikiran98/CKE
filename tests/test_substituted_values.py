"""Substituted constants must announce themselves.

A benchmark that ran on a hashed embedder was one failure mode. The other is a
number that was never measured at all: a missing key becoming a confidence, a
trust score, or a routing certainty. These assert that each such substitution
declares itself and is refused under strict.
"""

from __future__ import annotations

import pytest

from cke.diagnostics import (
    DegradedComponentError,
    clear_runtime_state,
    environment_report,
)


@pytest.fixture(autouse=True)
def _clean_runtime_state():
    clear_runtime_state()
    yield
    clear_runtime_state()


def test_dense_retriever_declares_a_missing_score():
    """One missing key became confidence, trust and retrieval score alike."""
    from cke.retrieval.dense_evidence_retriever import DenseEvidenceRetriever

    class ScorelessRetriever:
        def retrieve(self, query, k=5):
            return [{"text": "a passage", "doc_id": "d1"}]

    retriever = DenseEvidenceRetriever(ScorelessRetriever())
    retriever.retrieve("a question")

    reasons = [r.reason for r in environment_report().degradations]
    assert any("carried no score" in r for r in reasons)
    assert any("None of the three is a measurement" in r for r in reasons)


def test_dense_retriever_strict_refuses_a_missing_score():
    from cke.retrieval.dense_evidence_retriever import DenseEvidenceRetriever

    class ScorelessRetriever:
        def retrieve(self, query, k=5):
            return [{"text": "a passage", "doc_id": "d1"}]

    retriever = DenseEvidenceRetriever(ScorelessRetriever(), strict=True)
    with pytest.raises(DegradedComponentError):
        retriever.retrieve("a question")


def test_a_scored_result_is_not_a_degradation():
    from cke.retrieval.dense_evidence_retriever import DenseEvidenceRetriever

    class ScoredRetriever:
        def retrieve(self, query, k=5):
            return [{"text": "a passage", "doc_id": "d1", "score": 0.9}]

    chunks, facts = DenseEvidenceRetriever(ScoredRetriever(), strict=True).retrieve("q")

    assert facts[0].trust_score == 0.9
    assert environment_report().is_degraded is False


def test_path_confidence_no_longer_counts_an_unscored_edge_as_maximum():
    """A missing trust score used to read as 1.0, so absent instrumentation
    raised the metric instead of lowering it."""
    from cke.evaluation.extended_metrics import EvaluationMetrics

    scored_only = [[{"trust_score": 0.4}]]
    with_unscored = [[{"trust_score": 0.4}, {"subject": "no score here"}]]

    assert EvaluationMetrics.path_confidence(scored_only) == pytest.approx(0.4)
    # The unscored edge is excluded rather than averaged in as 1.0, which
    # would have produced 0.7.
    assert EvaluationMetrics.path_confidence(with_unscored) == pytest.approx(0.4)

    reasons = [r.reason for r in environment_report().degradations]
    assert any("neither trust_score nor confidence" in r for r in reasons)


def test_path_confidence_strict_refuses_unscored_edges():
    from cke.evaluation.extended_metrics import EvaluationMetrics

    with pytest.raises(DegradedComponentError):
        EvaluationMetrics.path_confidence([[{"subject": "no score"}]], strict=True)


def test_reasoner_adapter_declares_a_substituted_confidence():
    """A bare answer string got a constant reported as the reasoner's own
    certainty."""
    import inspect

    from cke.reasoning import reasoner_adapter as module

    source = inspect.getsource(module.ReasonerAdapter.reason)
    assert "declare_degradation" in source
    assert "_SUBSTITUTED_CONFIDENCE" in source
    assert module._SUBSTITUTED_CONFIDENCE == 0.8


def test_orchestrator_declares_a_substituted_route_confidence():
    from cke.pipeline import query_orchestrator as module

    class PlanWithNoConfidence:
        pass

    orchestrator = module.QueryOrchestrator.__new__(module.QueryOrchestrator)
    orchestrator.strict = False

    value = orchestrator._route_confidence(PlanWithNoConfidence())

    assert value == module._SUBSTITUTED_ROUTE_CONFIDENCE
    reasons = [r.reason for r in environment_report().degradations]
    assert any("neither route_confidence nor confidence_score" in r for r in reasons)


def test_orchestrator_uses_a_real_route_confidence_when_present():
    from cke.pipeline import query_orchestrator as module

    class Plan:
        route_confidence = 0.42

    orchestrator = module.QueryOrchestrator.__new__(module.QueryOrchestrator)
    orchestrator.strict = True

    assert orchestrator._route_confidence(Plan()) == pytest.approx(0.42)
    assert environment_report().is_degraded is False
