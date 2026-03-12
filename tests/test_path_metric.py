"""Tests for path completeness metric integration."""

from cke.evaluation.ablation_runner import AblationRunner
from cke.evaluation.extended_metrics import EvaluationMetrics


def test_path_completeness_metric() -> None:
    score = EvaluationMetrics.path_completeness(
        predicted_nodes=["Redis", "RESP", "PubSub"],
        gold_nodes=["Redis", "RESP", "Protocol"],
    )
    assert score == 2 / 3


def test_ablation_aggregate_includes_path_completeness() -> None:
    runner = AblationRunner(lambda item, variant: item)
    rows = [
        {
            "prediction": "A",
            "answer": "A",
            "predicted_evidence": ["e1"],
            "gold_evidence": ["e1", "e2"],
            "retrieved_nodes": ["n1", "n2"],
            "gold_nodes": ["n1", "n3"],
        }
    ]
    metrics = runner._aggregate(rows)
    assert "path_completeness" in metrics
    assert metrics["path_completeness"] == 0.5
