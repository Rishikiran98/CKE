"""Values that stood where a measurement belonged, and now say so.

Each of these fed a number the pipeline reports. None of them declared.
"""

from __future__ import annotations

import pytest

from cke.diagnostics import DegradedComponentError


# ---------------------------------------------------------------------------
# An assertion with no evidence is not an assertion whose evidence agrees
# ---------------------------------------------------------------------------


def _assertion(*texts: str):
    from cke.schema.assertion import Assertion, Evidence

    return Assertion(
        subject="A",
        relation="r",
        object="B",
        evidence=[Evidence(text=text) for text in texts],
    )


def test_no_evidence_is_not_perfect_agreement():
    """len(texts) <= 1 counted zero evidence as agreeing, so a graph of
    unevidenced assertions scored 1.0 and inflated w_freq and w_conf."""
    from cke.trust.calibration import TrustCalibrator

    calibrator = TrustCalibrator()

    assert calibrator._evidence_agreement([_assertion()]) is None
    assert calibrator._evidence_agreement([_assertion("one span")]) == 1.0
    assert calibrator._evidence_agreement([_assertion("a", "b")]) == 0.0
    # The unevidenced assertion is left out of both sides of the ratio.
    mixed = [_assertion(), _assertion("one span"), _assertion("a", "b")]
    assert calibrator._evidence_agreement(mixed) == 0.5


def test_fitting_on_a_graph_with_no_evidence_declares_and_leaves_the_weights():
    from cke.trust.calibration import TrustCalibrator

    class _Graph:
        assertions = [_assertion(), _assertion()]

    calibrator = TrustCalibrator()
    before = calibrator.config.w_freq
    calibrator.fit_from_graph(_Graph())

    assert calibrator.degraded is True
    assert "could not be measured" in calibrator.degraded_reason
    assert calibrator.config.w_freq == before, "unfitted, not fitted to nothing"

    with pytest.raises(DegradedComponentError):
        TrustCalibrator(strict=True).fit_from_graph(_Graph())


# ---------------------------------------------------------------------------
# The calibrator says when a signal it weights was never given
# ---------------------------------------------------------------------------


def test_an_absent_weighted_signal_is_declared():
    from cke.trust.confidence_calibrator import ConfidenceCalibrator

    calibrator = ConfidenceCalibrator()
    calibrator.calibrate({"evidence_count": 2, "top_evidence_score": 0.9})

    assert calibrator.degraded is True
    for name in ("path_score", "operator_confidence", "route_confidence"):
        assert name in calibrator.degraded_reason


def test_a_signal_present_and_zero_is_not_declared():
    """Zero is a measurement. Only absence is the substitution."""
    from cke.trust.confidence_calibrator import ConfidenceCalibrator

    calibrator = ConfidenceCalibrator()
    calibrator.calibrate(
        {
            "path_score": 0.0,
            "operator_confidence": 0.0,
            "route_confidence": 0.0,
            "evidence_count": 0,
        }
    )

    assert calibrator.degraded is False


def test_a_signal_that_is_not_a_number_is_declared():
    from cke.trust.confidence_calibrator import ConfidenceCalibrator

    calibrator = ConfidenceCalibrator()
    calibrator.calibrate(
        {
            "path_score": "not a number",
            "operator_confidence": 0.5,
            "route_confidence": 0.5,
        }
    )

    assert calibrator.degraded is True
    assert "not a number" in calibrator.degraded_reason


# ---------------------------------------------------------------------------
# A comparison whose direction the question never stated
# ---------------------------------------------------------------------------


def test_a_comparison_with_no_stated_direction_is_refused():
    """ ">" was assumed, so a question that never said which way was answered
    under a guess — and the opposite answer was equally available."""
    from cke.models import Statement
    from cke.pipeline.types import ResolvedEntity
    from cke.reasoning.operator_executor import OperatorExecutor

    def _entity(name):
        return ResolvedEntity(
            surface_form=name,
            canonical_name=name,
            entity_id=name,
            link_confidence=1.0,
        )

    facts = [
        Statement("Film A", "release_year", "1990"),
        Statement("Film B", "release_year", "2001"),
    ]
    entities = [_entity("Film A"), _entity("Film B")]
    executor = OperatorExecutor()

    stated = executor._execute_compare(
        "temporal_compare",
        "Which film was released later, Film A or Film B?",
        facts,
        entities,
    )
    assert stated is not None and stated.result_value == "Film B"

    unstated = executor._execute_compare(
        "temporal_compare",
        "Compare Film A and Film B.",
        facts,
        entities,
    )
    assert unstated is None, "no direction was stated, so none is assumed"


def test_the_verifier_rechecks_a_selection_instead_of_skipping_it():
    """The executor renamed the operator to compare_selection, which the
    verifier did not know, so it skipped the recheck — and passed was forced
    True on the way past."""
    from cke.reasoning.verifier import ReasoningVerifier

    verifier = ReasoningVerifier()
    honest = {
        "operator": "compare_selection",
        "inputs": ("1990", "2001", ">", "Film A", "Film B", "date_compare"),
        "result": "Film B",
    }
    tampered = {**honest, "result": "Film A"}

    assert verifier._check_logical_validity([honest]) == (True, "")
    assert verifier._check_logical_validity([tampered])[0] is False


def test_evidence_facts_with_no_score_are_declared():
    """A fact carrying none of trust, retrieval or statement confidence fed
    the calibrator a zero as top_evidence_score, indistinguishable from a fact
    measured as worthless."""
    from cke.graph_engine.graph_engine import KnowledgeGraphEngine
    from cke.models import Statement
    from cke.pipeline.query_orchestrator import QueryOrchestrator
    from cke.pipeline.types import EvidenceFact, ReasoningContext
    from cke.router.query_router import QueryRouter

    def _fact(trust: float, retrieval: float, confidence: float) -> EvidenceFact:
        return EvidenceFact(
            statement=Statement("A", "r", "B", confidence=confidence),
            chunk_id="c0",
            source="d0",
            trust_score=trust,
            retrieval_score=retrieval,
            entity_alignment_score=0.0,
        )

    orchestrator = QueryOrchestrator(
        graph_engine=KnowledgeGraphEngine(), router=QueryRouter()
    )
    plan = QueryRouter().route("Where is A?")

    scored = ReasoningContext(
        query="Where is A?", query_plan=plan, evidence_facts=[_fact(0.9, 0.0, 0.0)]
    )
    orchestrator._initial_confidence_signals(scored, plan, [])
    assert orchestrator.degraded is False, "a scored fact is not a substitution"

    unscored = ReasoningContext(
        query="Where is A?", query_plan=plan, evidence_facts=[_fact(0.0, 0.0, 0.0)]
    )
    orchestrator._initial_confidence_signals(unscored, plan, [])

    assert orchestrator.degraded is True
    assert "measured zeros" in orchestrator.degraded_reason
