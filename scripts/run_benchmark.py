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

from cke.datasets.registry import DATASET_REGISTRY  # noqa: E402
from cke.diagnostics import environment_report  # noqa: E402
from cke.evaluation.ablation_runner import AblationRunner  # noqa: E402
from cke.evaluation.experiment_runner import ExperimentRunner  # noqa: E402
from cke.observability.system_monitor import SystemMonitor  # noqa: E402
from cke.observability.token_tracker import TokenTracker  # noqa: E402
from cke.reporting.report_generator import ReportGenerator  # noqa: E402
from cke.retrieval.rag_baseline import RAGRetriever  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run CKE benchmark")
    parser.add_argument(
        "--dataset", required=True, choices=sorted(DATASET_REGISTRY.keys())
    )
    parser.add_argument("--dataset-path", required=True)
    parser.add_argument(
        "--mode", default="full", choices=["full", "rag_only", "graph_only"]
    )
    parser.add_argument("--limit", type=int, default=100)
    parser.add_argument("--output-dir", default="results")
    parser.add_argument(
        "--skip-ablation",
        action="store_true",
        help=(
            "Skip the ablation stage. Required until a real ablation "
            "evaluator exists; see _ablation_evaluator."
        ),
    )
    parser.add_argument(
        "--allow-degraded",
        action="store_true",
        help=(
            "Permit components to run degraded. Off by default: a benchmark "
            "that ran on a hashed embedder is not a benchmark."
        ),
    )
    return parser.parse_args()


class AblationNotImplementedError(NotImplementedError):
    """Raised because this script has no real ablation evaluator."""


def _ablation_evaluator(item, variant):
    """Placeholder that refuses to score rather than fabricating a result.

    The previous implementation returned the gold answer as the prediction and
    ignored the variant, so AblationRunner scored exact_match=1.0 and
    f1_score=1.0 identically across all five variants, for any dataset. That is
    a perfect-score generator, not an ablation.

    A real evaluator has to run the pipeline under each variant's
    configuration and return what the pipeline actually predicted. Until one
    exists this raises, because a missing ablation is recoverable and a
    fabricated one is not.
    """
    raise AblationNotImplementedError(
        "This script has no ablation evaluator. The previous one returned the "
        f"gold answer as its own prediction and ignored the variant "
        f"({variant!r}), so every variant scored exact_match=1.0 by "
        "construction. Implement an evaluator that runs the pipeline under "
        "each variant and returns its real prediction, or run this script "
        "with --skip-ablation."
    )


def main() -> None:
    args = parse_args()
    print(environment_report().render(), flush=True)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    loader = DATASET_REGISTRY[args.dataset]()
    data = loader.load(args.dataset_path).items[: args.limit]

    docs = []
    for item in data:
        for i, doc in enumerate(item.get("documents", [])):
            docs.append(
                {
                    "doc_id": f"{item.get('id', 'sample')}_{i}",
                    "text": doc.get("text", ""),
                }
            )

    retriever = RAGRetriever(strict=not args.allow_degraded)
    if docs:
        retriever.build_index(docs)

    token_tracker = TokenTracker()
    runner = ExperimentRunner(retriever=retriever, token_tracker=token_tracker)
    dataset_rows = [
        {"question": row.get("question", ""), "answer": row.get("answer", "")}
        for row in data
    ]
    metrics = runner.run(dataset_rows, top_k=5)

    ablation: dict = {}
    if not args.skip_ablation:
        ablation = AblationRunner(evaluator=_ablation_evaluator).run(
            data, output_dir=output_dir
        )
    monitor = SystemMonitor()
    system_metrics = monitor.snapshot()

    (output_dir / f"{args.dataset}_metrics.json").write_text(
        json.dumps(metrics, indent=2), encoding="utf-8"
    )
    if ablation:
        (output_dir / "ablation.json").write_text(
            json.dumps(ablation, indent=2), encoding="utf-8"
        )
    (output_dir / "system_metrics.json").write_text(
        json.dumps(system_metrics, indent=2), encoding="utf-8"
    )

    ReportGenerator().generate(
        output_path=output_dir / "markdown_report.md",
        dataset_results=metrics,
        ablation_results=ablation,
        system_metrics=system_metrics,
        cost_stats=token_tracker.to_dict(),
    )


if __name__ == "__main__":
    main()
