"""The reasoner reports what it measured, and a strict run can use it.

`PathReasoner` computed a reasoning confidence and returned a bare string,
throwing it away one line before returning. `ReasonerAdapter` then substituted
a constant, 0.8, which happens to sit above `ReasoningVerifier`'s 0.7
threshold — so weak evidence cleared verification on the strength of the
constant, and a strict orchestrator refused every query it could answer.

The whole suite passed either way, which is why these tests exist.
"""

from __future__ import annotations

import pytest

from cke.diagnostics import DegradedComponentError
from cke.graph_engine.graph_engine import KnowledgeGraphEngine
from cke.models import Statement
from cke.pipeline.query_orchestrator import QueryOrchestrator
from cke.reasoning.path_reasoner import PathReasoner
from cke.reasoning.reasoner_adapter import ReasonerAdapter
from cke.reasoning.verifier import ReasoningVerifier
from cke.router.query_router import QueryRouter

_QUESTION = "Which country is Hagia Sophia located in?"


def _chain(confidence: float) -> list[Statement]:
    return [
        Statement("Hagia Sophia", "located_in", "Istanbul", confidence=confidence),
        Statement("Istanbul", "located_in", "Turkey", confidence=confidence),
    ]


def _engine(confidence: float = 0.9) -> KnowledgeGraphEngine:
    engine = KnowledgeGraphEngine()
    for statement in _chain(confidence):
        engine.add_statement(
            statement.subject,
            statement.relation,
            statement.object,
            confidence=statement.confidence,
        )
    return engine


# ---------------------------------------------------------------------------
# The confidence is the one that was computed
# ---------------------------------------------------------------------------


def test_the_reasoner_reports_the_confidence_it_computed():
    outcome = PathReasoner().reason(_QUESTION, _chain(0.9))

    assert outcome.answer == "Turkey"
    # 0.9 * 0.9 over a one-edge inferred path. Not a constant.
    assert outcome.confidence == pytest.approx(0.81, abs=1e-4)
    assert outcome.summary == "path_reasoning_verified"
    assert outcome.reasoning_path, "the path behind the answer travels with it"


def test_the_confidence_moves_with_the_evidence():
    """A constant cannot do this, and that is the whole point of the finding."""
    strong = PathReasoner().reason(_QUESTION, _chain(0.9)).confidence
    weaker = PathReasoner().reason(_QUESTION, _chain(0.8)).confidence

    assert strong > weaker > 0.0


def test_weak_evidence_no_longer_clears_the_verifier_on_a_constant():
    """0.8 sat above the 0.7 threshold, so evidence this weak used to be
    reported as verified."""
    assert ReasoningVerifier().confidence_threshold == 0.7

    reasoner = PathReasoner()
    outcome = reasoner.reason(_QUESTION, _chain(0.5))

    assert outcome.confidence < 0.7
    assert outcome.summary == "advanced_reasoner_fallback"
    assert reasoner.degraded is True
    assert "no confidence of its own" in reasoner.degraded_reason


def test_a_reasoner_that_cannot_verify_refuses_under_strict(offline_embedder):
    with pytest.raises(DegradedComponentError, match="path reasoning"):
        PathReasoner(strict=True).reason(_QUESTION, _chain(0.5))


def test_an_abstention_carries_zero_and_says_which_route_it_took():
    reasoner = PathReasoner()

    assert reasoner.reason(_QUESTION, []).summary == "no_evidence_provided"
    assert reasoner.reason(_QUESTION, []).confidence == 0.0
    unrelated = [Statement("Redis", "uses", "TCP", confidence=0.9)]
    assert reasoner.reason(_QUESTION, unrelated).confidence == 0.0


# ---------------------------------------------------------------------------
# The adapter stops substituting, but only where a real number exists
# ---------------------------------------------------------------------------


def test_the_adapter_does_not_substitute_when_the_reasoner_reports_one():
    adapter = ReasonerAdapter(PathReasoner(), strict=True)

    outcome = adapter.reason(_QUESTION, _chain(0.9))

    assert outcome is not None
    assert outcome.confidence == pytest.approx(0.81, abs=1e-4)
    assert adapter.degraded is False


def test_the_adapter_still_declares_for_a_reasoner_that_has_no_confidence():
    """The substitution is correct for a reasoner that genuinely reports none;
    it must not be quietly deleted along with the case that had one."""

    class _StringOnlyReasoner:
        def answer(self, query, context):
            return "Turkey"

    adapter = ReasonerAdapter(_StringOnlyReasoner())
    outcome = adapter.reason(_QUESTION, _chain(0.9))

    assert outcome.confidence == 0.8
    assert adapter.degraded is True
    assert "substituted" in adapter.degraded_reason

    with pytest.raises(DegradedComponentError):
        ReasonerAdapter(_StringOnlyReasoner(), strict=True).reason(
            _QUESTION, _chain(0.9)
        )


@pytest.mark.parametrize("answer", ["", None], ids=["empty", "none"])
def test_a_reasoner_that_returns_no_answer_abstains_rather_than_vanishing(answer):
    """Why the branch below it was unreachable.

    The adapter carried `if not answer: return None` immediately after the
    branch that already returns INSUFFICIENT_EVIDENCE for a falsy answer, so
    nothing could reach it. Deleting an unreachable line is safe only while
    the line above keeps handling the case; this holds it there.

    Abstaining rather than returning None matters downstream: None reads as
    "the adapter had nothing to say", and an abstention is something the
    reasoner said.
    """

    class _SilentReasoner:
        def answer(self, query, context):
            return answer

    outcome = ReasonerAdapter(_SilentReasoner()).reason(_QUESTION, _chain(0.9))

    assert outcome is not None
    assert outcome.answer == "INSUFFICIENT_EVIDENCE"
    assert outcome.confidence == 0.0
    assert outcome.summary == "reasoner_abstained"


# ---------------------------------------------------------------------------
# The test that did not exist
# ---------------------------------------------------------------------------


def test_a_strict_orchestrator_answers_a_query_it_can_answer(offline_embedder):
    """No test anywhere constructed a strict QueryOrchestrator and called
    .answer(). It raised on every query the adapter handled."""
    orchestrator = QueryOrchestrator(
        graph_engine=_engine(), router=QueryRouter(strict=True), strict=True
    )

    result = orchestrator.answer(_QUESTION)

    assert result.answer == "Turkey"
    assert 0.0 < result.confidence <= 1.0


def test_the_strict_and_permissive_orchestrators_agree(offline_embedder):
    """They differ only in what they refuse, never in what they report."""
    strict = QueryOrchestrator(
        graph_engine=_engine(), router=QueryRouter(strict=True), strict=True
    ).answer(_QUESTION)
    permissive = QueryOrchestrator(
        graph_engine=_engine(), router=QueryRouter(), strict=False
    ).answer(_QUESTION)

    assert strict.answer == permissive.answer
    assert strict.confidence == pytest.approx(permissive.confidence)
