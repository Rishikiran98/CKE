"""No value may stand in for a measurement without saying so.

The dependency fallbacks were one half of Phase 2. This is the other: a
missing key, a dropped row, or an unreadable blob becoming a number that a
reader would take for something the system observed.
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


def _reasons() -> list[str]:
    return [record.reason for record in environment_report().degradations]


def test_ablation_runner_declares_rows_with_no_prediction(tmp_path):
    """A row missing a key was scored as empty against empty."""
    from cke.evaluation.ablation_runner import AblationRunner

    runner = AblationRunner(evaluator=lambda item, variant: {"answer": "Paris"})
    runner.run([{"question": "q"}], output_dir=tmp_path)

    assert any("carried no prediction" in r for r in _reasons())


def test_compare_summaries_excludes_a_one_sided_metric():
    """A metric on one side only was differenced against zero."""
    from cke.evaluation.eval_types import EvaluationSummary
    from cke.evaluation.reporting import compare_summaries

    def summary(metrics):
        return EvaluationSummary(
            total_cases=1,
            exact_matches=1,
            acceptable_matches=1,
            abstentions=0,
            failed_cases=0,
            accuracy=1.0,
            acceptable_accuracy=1.0,
            retrieval_metrics=metrics,
        )

    baseline = summary({"recall": 0.4})
    candidate = summary({"recall": 0.6, "precision": 0.9})

    deltas = compare_summaries(baseline, candidate)

    assert deltas == {"recall": pytest.approx(0.2)}
    assert "precision" not in deltas
    assert any("only one side" in r for r in _reasons())


def test_token_tracker_reports_no_cost_rather_than_zero():
    """cost_estimate was emitted as 0.0 with no pricing configured."""
    from cke.observability.token_tracker import TokenTracker

    tracker = TokenTracker()
    tracker.add_usage(1000, 500)
    payload = tracker.to_dict()

    assert payload["cost_estimate"] is None
    assert payload["cost_estimate_is_computed"] is False
    assert tracker.degraded is True

    priced = TokenTracker(cost_per_1k_prompt=0.01)
    priced.add_usage(1000, 0)
    assert priced.to_dict()["cost_estimate"] == pytest.approx(0.01)
    assert priced.degraded is False


def test_sqlite_store_declares_an_unreadable_context(tmp_path):
    """An undecodable blob silently dropped qualifiers and evidence."""
    from cke.storage.sqlite_store import SQLiteStore

    store = SQLiteStore(tmp_path / "t.db")
    assert store._decode_context("{not json") == {}
    assert any("could not be decoded" in r for r in _reasons())

    strict_store = SQLiteStore(tmp_path / "s.db", strict=True)
    with pytest.raises(DegradedComponentError):
        strict_store._decode_context("{not json")


def test_confidence_model_declares_substituted_features():
    """Rule-extracted statements carry none of these features."""
    from cke.models import Statement
    from cke.trust.confidence_model import ConfidenceModel

    ConfidenceModel().predict(Statement("A", "uses", "B"))

    assert any("substituted constants" in r for r in _reasons())

    with pytest.raises(DegradedComponentError):
        ConfidenceModel(strict=True).predict(Statement("A", "uses", "B"))


def test_domain_classifier_never_labels_the_unclassifiable_with_a_real_domain():
    """The substitution is gone, so there is no longer one to declare.

    This used to return "programming" for anything it could not place, and
    declare a degradation to say so. Declaring it was an improvement on
    silence, but the label was still a real domain, indistinguishable
    downstream from a match, and it made strict mode unusable on any corpus
    outside the taxonomy: a benchmark of encyclopedic questions matched no
    keyword, so a strict run could not reach its first result.

    Saying "unclassified" substitutes nothing, so it is an answer rather than
    a degradation, and the assertion is the stronger one — not that the lie is
    announced, but that it is not told.
    """
    from cke.graph.domain_classifier import (
        UNCLASSIFIED_DOMAIN,
        DomainClassifier,
    )

    classifier = DomainClassifier(strict=True)
    label = classifier.classify_entity("zzzqqq unclassifiable")

    assert label == UNCLASSIFIED_DOMAIN
    assert label not in DomainClassifier.DOMAIN_KEYWORDS
    assert classifier.degraded is False

    # A query it can place is still placed.
    assert classifier.classify_entity("redis database index") == "databases"


def test_locomo_loader_declares_a_conversation_with_no_sessions():
    """A record with no session_N turn list loads with zero documents."""
    from cke.datasets.locomo_loader import LoCoMoDataset

    LoCoMoDataset()._conversation_documents({"speaker_a": "A"})

    assert any("no session_N turn lists" in r for r in _reasons())


def test_hotpot_loader_declares_dropped_context_entries():
    from cke.datasets.hotpot_loader import HotpotDataset

    HotpotDataset()._context_to_documents([["Title", ["a sentence"]], ["malformed"]])

    assert any("were dropped" in r for r in _reasons())


def test_graph_store_declares_an_assertion_with_no_confidence():
    from cke.graph.graph_store import GraphStore

    GraphStore().add_assertion({"subject": "A", "relation": "uses", "object": "B"})

    assert any("no confidence" in r for r in _reasons())

    with pytest.raises(DegradedComponentError):
        GraphStore(strict=True).add_assertion(
            {"subject": "A", "relation": "uses", "object": "B"}
        )


def test_default_evidence_retriever_declares_a_whole_graph_scan():
    """No seed matched, so it returned the entire graph as if retrieved."""
    from cke.graph_engine.graph_engine import KnowledgeGraphEngine
    from cke.retrieval.default_evidence_retriever import DefaultEvidenceRetriever

    engine = KnowledgeGraphEngine()
    engine.add_statement("Redis", "uses", "RESP")
    retriever = DefaultEvidenceRetriever(engine)
    retriever.retrieve("a query matching nothing at all")

    assert any("every statement in the graph is returned" in r for r in _reasons())
