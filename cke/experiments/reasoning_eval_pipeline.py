"""Benchmark harness for deterministic reasoning on HotpotQA, MS MARCO, and LoCoMo."""

from __future__ import annotations

import argparse
import re
import time
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from cke.datasets.registry import load_dataset
from cke.models import Statement
from cke.reasoning.path_reasoner import PathReasoner


@dataclass(slots=True)
class DatasetMetrics:
    exact_match: float
    f1: float
    reasoning_depth: float
    edges_traversed: float
    latency: float


class ReasoningEvalPipeline:
    """Evaluates path reasoning quality and runtime behavior across datasets."""

    def __init__(self, reasoner: PathReasoner | None = None) -> None:
        self.reasoner = reasoner or PathReasoner()

    def evaluate_dataset(
        self,
        dataset_name: str,
        dataset_path: str,
        max_samples: int | None = None,
    ) -> DatasetMetrics:
        loader = load_dataset(dataset_name, dataset_path)
        items = loader.items[:max_samples] if max_samples else loader.items
        if not items:
            return DatasetMetrics(0.0, 0.0, 0.0, 0.0, 0.0)

        total_em = 0.0
        total_f1 = 0.0
        total_depth = 0.0
        total_edges = 0.0
        total_latency = 0.0

        for item in items:
            question = str(item.get("question") or "")
            gold_answer = str(item.get("answer") or "")
            context = self._documents_to_statements(item.get("documents") or [])

            start = time.perf_counter()
            prediction = self.reasoner.answer(question, context)
            latency = time.perf_counter() - start

            total_em += float(self._exact_match(prediction, gold_answer))
            total_f1 += self._f1_score(prediction, gold_answer)
            total_depth += self._reasoning_depth()
            total_edges += self._edges_traversed()
            total_latency += latency

        count = float(len(items))
        return DatasetMetrics(
            exact_match=total_em / count,
            f1=total_f1 / count,
            reasoning_depth=total_depth / count,
            edges_traversed=total_edges / count,
            latency=total_latency / count,
        )

    def evaluate(
        self,
        hotpot_path: str,
        msmarco_path: str,
        locomo_path: str,
        max_samples: int | None = None,
    ) -> dict[str, DatasetMetrics]:
        return {
            "hotpotqa": self.evaluate_dataset("hotpotqa", hotpot_path, max_samples),
            "msmarco": self.evaluate_dataset("msmarco", msmarco_path, max_samples),
            "locomo": self.evaluate_dataset("locomo", locomo_path, max_samples),
        }

    def _reasoning_depth(self) -> int:
        trace = self.reasoner.format_reasoning_path()
        return sum(1 for line in trace.splitlines() if " -> " in line)

    def _edges_traversed(self) -> int:
        trace = self.reasoner.format_reasoning_path()
        ranked = [
            line for line in trace.splitlines() if line.startswith("Ranked expansion")
        ]
        return len(ranked)

    @staticmethod
    def _normalize(text: str) -> list[str]:
        return re.findall(r"\w+", text.lower())

    def _exact_match(self, prediction: str, target: str) -> bool:
        return " ".join(self._normalize(prediction)) == " ".join(
            self._normalize(target)
        )

    def _f1_score(self, prediction: str, target: str) -> float:
        pred_tokens = self._normalize(prediction)
        target_tokens = self._normalize(target)
        if not pred_tokens or not target_tokens:
            return 0.0
        common = Counter(pred_tokens) & Counter(target_tokens)
        overlap = sum(common.values())
        if overlap == 0:
            return 0.0
        precision = overlap / len(pred_tokens)
        recall = overlap / len(target_tokens)
        return 2 * precision * recall / (precision + recall)

    @staticmethod
    def _documents_to_statements(documents: list[dict[str, Any]]) -> list[Statement]:
        statements: list[Statement] = []
        for doc in documents:
            title = str(doc.get("title") or doc.get("doc_id") or "document")
            text = str(doc.get("text") or "")
            if not text.strip():
                continue
            relation = "contains"
            statements.append(
                Statement(subject=title, relation=relation, object=text, confidence=0.8)
            )
        return statements


def main() -> None:
    parser = argparse.ArgumentParser(description="Run CKE reasoning benchmark harness")
    parser.add_argument("--hotpot", required=True, help="Path to HotpotQA JSON")
    parser.add_argument("--msmarco", required=True, help="Path to MS MARCO TSV")
    parser.add_argument("--locomo", required=True, help="Path to LoCoMo JSON")
    parser.add_argument("--max-samples", type=int, default=100)
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()

    pipeline = ReasoningEvalPipeline()
    metrics = pipeline.evaluate(
        hotpot_path=args.hotpot,
        msmarco_path=args.msmarco,
        locomo_path=args.locomo,
        max_samples=args.max_samples,
    )

    lines = []
    for dataset_name, values in metrics.items():
        lines.append(
            f"{dataset_name}: "
            f"ExactMatch={values.exact_match:.4f}, "
            f"F1={values.f1:.4f}, "
            f"reasoning_depth={values.reasoning_depth:.2f}, "
            f"edges_traversed={values.edges_traversed:.2f}, "
            f"latency={values.latency:.4f}"
        )

    report = "\n".join(lines)
    print(report)
    if args.out:
        args.out.write_text(report + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
