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
    environment_report,
)


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
    assert "self._degrade" in source
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


# ---------------------------------------------------------------------------
# The flag must be on the object, and strict must reach it
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "module_path, class_name",
    [
        ("cke.retrieval.dense_evidence_retriever", "DenseEvidenceRetriever"),
        ("cke.retrieval.hybrid_evidence_retriever", "HybridEvidenceRetriever"),
        ("cke.reasoning.reasoner_adapter", "ReasonerAdapter"),
        ("cke.pipeline.query_orchestrator", "QueryOrchestrator"),
    ],
)
def test_substituting_components_expose_the_instance_flag(module_path, class_name):
    """declare_degradation only logs and records; it sets no flag on the object.

    Using it inside a method left obligation two of the contract unmet:
    inspecting the component itself raised AttributeError.
    """
    import importlib

    from cke.diagnostics import DegradationMixin

    component = getattr(importlib.import_module(module_path), class_name)
    assert issubclass(
        component, DegradationMixin
    ), f"{class_name} substitutes values but does not carry the instance flag"


def test_orchestrator_forwards_strict_to_every_adapter():
    """QueryOrchestrator(strict=True) built adapters that defaulted to False,
    so a strict run only warned where it should have raised."""
    import inspect

    from cke.pipeline import query_orchestrator as module

    init = inspect.getsource(module.QueryOrchestrator.__init__)
    assert "ReasonerAdapter(self.reasoner, strict=strict)" in init
    assert "strict=strict," in init

    builder = inspect.getsource(module.QueryOrchestrator._build_retriever)
    assert "HybridEvidenceRetriever(router, strict=strict)" in builder
    assert "DenseEvidenceRetriever(dense_retriever, strict=strict)" in builder


def test_hybrid_retriever_strict_refuses_synthetic_trust():
    from cke.retrieval.hybrid_evidence_retriever import HybridEvidenceRetriever

    class Pack:
        graph_statements: list = []
        fallback_chunks = ["a chunk with no trust of its own"]

    class Router:
        def retrieve(self, *args, **kwargs):
            return Pack()

    with pytest.raises(DegradedComponentError, match="substituted confidence"):
        HybridEvidenceRetriever(Router(), strict=True).retrieve("a question")


def test_no_class_declares_a_degradation_without_carrying_the_flag():
    """declare_degradation is for module-level functions.

    Called from a method it logs and records but sets nothing on the object,
    so a caller inspecting the component cannot see the degraded state. This
    walks the package rather than trusting a hand-kept list, because that is
    exactly how these were missed.
    """
    import ast
    import pathlib

    root = pathlib.Path(__file__).resolve().parents[1] / "cke"
    offenders = []

    for path in sorted(root.rglob("*.py")):
        if "tests" in path.parts:
            continue
        source = path.read_text(encoding="utf-8")
        if "declare_degradation(" not in source:
            continue
        for node in ast.walk(ast.parse(source)):
            if not isinstance(node, ast.ClassDef):
                continue
            bases = [ast.unparse(base) for base in node.bases]
            if "DegradationMixin" in bases:
                continue
            for method in node.body:
                if not isinstance(method, ast.FunctionDef):
                    continue
                takes_self = method.args.args and method.args.args[0].arg == "self"
                if takes_self and "declare_degradation(" in ast.unparse(method):
                    offenders.append(
                        f"{path}:{method.lineno} {node.name}.{method.name}"
                    )

    assert not offenders, (
        "instance methods declaring a degradation without the instance flag: "
        + ", ".join(offenders)
    )
