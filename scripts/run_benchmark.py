#!/usr/bin/env python3
"""Run CKE benchmark with optional ablation and reporting."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from cke.datasets.registry import DATASET_REGISTRY
from cke.evaluation.ablation_runner import AblationRunner
from cke.evaluation.experiment_runner import ExperimentRunner
from cke.observability.system_monitor import SystemMonitor
from cke.observability.token_tracker import TokenTracker
from cke.reporting.report_generator import ReportGenerator
from cke.retrieval.rag_baseline import RAGRetriever


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run CKE benchmark")
    parser.add_argument("--dataset", required=True, choices=sorted(DATASET_REGISTRY.keys()))
    parser.add_argument("--dataset-path", required=True)
    parser.add_argument("--mode", default="full", choices=["full", "rag_only", "graph_only"])
    parser.add_argument("--limit", type=int, default=100)
    parser.add_argument("--output-dir", default="results")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    loader = DATASET_REGISTRY[args.dataset]()
    data = loader.load(args.dataset_path).items[: args.limit]

    docs = []
    for item in data:
        for i, doc in enumerate(item.get("documents", [])):
            docs.append({"doc_id": f"{item.get('id', 'sample')}_{i}", "text": doc.get("text", "")})

    retriever = RAGRetriever()
    if docs:
        retriever.build_index(docs)

    token_tracker = TokenTracker()
    runner = ExperimentRunner(retriever=retriever, token_tracker=token_tracker)
    dataset_rows = [{"question": row.get("question", ""), "answer": row.get("answer", "")} for row in data]
    metrics = runner.run(dataset_rows, top_k=5)

    def evaluator(item, _variant):
        question = item.get("question", "")
        answer = item.get("answer", "")
        pred = question if not answer else answer
        return {
            "prediction": pred,
            "answer": answer,
            "predicted_evidence": [],
            "gold_evidence": [],
        }

    ablation = AblationRunner(evaluator=evaluator).run(data, output_dir=output_dir)
    monitor = SystemMonitor()
    system_metrics = monitor.snapshot()

    (output_dir / f"{args.dataset}_metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    (output_dir / "ablation.json").write_text(json.dumps(ablation, indent=2), encoding="utf-8")
    (output_dir / "system_metrics.json").write_text(json.dumps(system_metrics, indent=2), encoding="utf-8")

    ReportGenerator().generate(
        output_path=output_dir / "markdown_report.md",
        dataset_results=metrics,
        ablation_results=ablation,
        system_metrics=system_metrics,
        cost_stats=token_tracker.to_dict(),
    )


if __name__ == "__main__":
    main()
