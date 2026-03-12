"""Extended evaluation metrics for CKE benchmarking."""

from __future__ import annotations

import re
from collections import Counter
from typing import Any


class EvaluationMetrics:
    """Collection of static evaluation helpers."""

    @staticmethod
    def _normalize(text: str) -> list[str]:
        return re.findall(r"\w+", str(text).lower())

    @classmethod
    def exact_match(cls, prediction: str, answer: str) -> float:
        pred = " ".join(cls._normalize(prediction))
        gold = " ".join(cls._normalize(answer))
        return float(pred == gold)

    @classmethod
    def f1_score(cls, prediction: str, answer: str) -> float:
        pred_tokens = cls._normalize(prediction)
        gold_tokens = cls._normalize(answer)
        if not pred_tokens or not gold_tokens:
            return 0.0
        overlap = sum((Counter(pred_tokens) & Counter(gold_tokens)).values())
        if overlap == 0:
            return 0.0
        precision = overlap / len(pred_tokens)
        recall = overlap / len(gold_tokens)
        return 2 * precision * recall / (precision + recall)

    @staticmethod
    def evidence_recall(
        predicted_evidence: list[str], gold_evidence: list[str]
    ) -> float:
        gold = set(gold_evidence)
        if not gold:
            return 0.0
        return len(set(predicted_evidence).intersection(gold)) / len(gold)

    @staticmethod
    def evidence_precision(
        predicted_evidence: list[str], gold_evidence: list[str]
    ) -> float:
        predicted = set(predicted_evidence)
        if not predicted:
            return 0.0
        return len(predicted.intersection(set(gold_evidence))) / len(predicted)

    @staticmethod
    def graph_hop_count(evidence_paths: list[list[dict[str, Any]]]) -> float:
        if not evidence_paths:
            return 0.0
        return sum(len(path) for path in evidence_paths) / len(evidence_paths)

    @staticmethod
    def path_confidence(evidence_paths: list[list[dict[str, Any]]]) -> float:
        if not evidence_paths:
            return 0.0
        per_path: list[float] = []
        for path in evidence_paths:
            if not path:
                continue
            per_path.append(
                sum(
                    float(edge.get("trust_score", edge.get("confidence", 1.0)))
                    for edge in path
                )
                / len(path)
            )
        return sum(per_path) / len(per_path) if per_path else 0.0

    @staticmethod
    def path_completeness(
        predicted_nodes: list[str],
        gold_nodes: list[str],
    ) -> float:
        """Share of gold evidence nodes contained in retrieved nodes."""
        gold = set(gold_nodes)
        if not gold:
            return 0.0
        predicted = set(predicted_nodes)
        return len(predicted.intersection(gold)) / len(gold)

    @staticmethod
    def latency_ms(start_time: float, end_time: float) -> float:
        return max(0.0, (end_time - start_time) * 1000)

    @staticmethod
    def retrieval_steps(evidence_paths: list[list[dict[str, Any]]]) -> int:
        return sum(len(path) for path in evidence_paths)
