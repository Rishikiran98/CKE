"""Run ablation studies across CKE variants."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable

from cke.evaluation.extended_metrics import EvaluationMetrics


@dataclass(slots=True)
class AblationConfig:
    use_graph: bool = True
    use_trust: bool = True
    use_conflict: bool = True
    use_rag: bool = False

    @property
    def name(self) -> str:
        if self.use_rag and not self.use_graph:
            return "rag_only"
        if self.use_graph and not self.use_trust:
            return "graph_only"
        if self.use_graph and self.use_trust and not self.use_conflict:
            return "graph_trust"
        if self.use_graph and self.use_trust and self.use_conflict and self.use_rag:
            return "full_cke"
        if self.use_graph and self.use_trust and self.use_conflict:
            return "graph_trust_conflict"
        return "custom"


class AblationRunner:
    """Execute variant configurations and persist metrics.json."""

    DEFAULT_VARIANTS = [
        AblationConfig(use_graph=False, use_trust=False, use_conflict=False, use_rag=True),
        AblationConfig(use_graph=True, use_trust=False, use_conflict=False, use_rag=False),
        AblationConfig(use_graph=True, use_trust=True, use_conflict=False, use_rag=False),
        AblationConfig(use_graph=True, use_trust=True, use_conflict=True, use_rag=False),
        AblationConfig(use_graph=True, use_trust=True, use_conflict=True, use_rag=True),
    ]

    def __init__(self, evaluator: Callable[[dict[str, Any], AblationConfig], dict[str, Any]]) -> None:
        self.evaluator = evaluator

    def run(
        self,
        dataset: list[dict[str, Any]],
        output_dir: str | Path,
        variants: list[AblationConfig] | None = None,
    ) -> dict[str, Any]:
        active = variants or self.DEFAULT_VARIANTS
        results: dict[str, Any] = {}

        for variant in active:
            per_item = [self.evaluator(item, variant) for item in dataset]
            metrics = self._aggregate(per_item)
            results[variant.name] = {
                "config": asdict(variant),
                "metrics": metrics,
            }

        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        metrics_file = output_path / "metrics.json"
        metrics_file.write_text(json.dumps(results, indent=2), encoding="utf-8")
        return results

    def _aggregate(self, rows: list[dict[str, Any]]) -> dict[str, float]:
        total = max(len(rows), 1)
        em = sum(EvaluationMetrics.exact_match(r.get("prediction", ""), r.get("answer", "")) for r in rows) / total
        f1 = sum(EvaluationMetrics.f1_score(r.get("prediction", ""), r.get("answer", "")) for r in rows) / total
        evidence_recall = sum(
            EvaluationMetrics.evidence_recall(r.get("predicted_evidence", []), r.get("gold_evidence", []))
            for r in rows
        ) / total
        evidence_precision = sum(
            EvaluationMetrics.evidence_precision(r.get("predicted_evidence", []), r.get("gold_evidence", []))
            for r in rows
        ) / total
        return {
            "exact_match": em,
            "f1_score": f1,
            "evidence_recall": evidence_recall,
            "evidence_precision": evidence_precision,
        }
