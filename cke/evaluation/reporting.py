"""Lightweight human and machine-readable evaluation reporting."""

from __future__ import annotations

import csv
import json
from dataclasses import asdict
from pathlib import Path

from cke.evaluation.eval_types import CaseEvaluationResult, EvaluationSummary


def generate_text_report(
    results: list[CaseEvaluationResult],
    summary: EvaluationSummary,
    max_failed_cases: int = 10,
) -> str:
    lines = [
        "CKE Sprint 8 Evaluation Report",
        f"Total cases: {summary.total_cases}",
        f"Exact accuracy: {summary.accuracy:.2%} ({summary.exact_matches}/{summary.total_cases})",
        "Acceptable accuracy: "
        f"{summary.acceptable_accuracy:.2%} ({summary.acceptable_matches}/{summary.total_cases})",
        f"Abstentions: {summary.abstentions}",
        f"Failed cases: {summary.failed_cases}",
        "",
        "Failure breakdown:",
    ]

    if summary.failure_breakdown:
        lines.extend(
            f"  - {failure_mode}: {count}"
            for failure_mode, count in summary.failure_breakdown.items()
        )
    else:
        lines.append("  - none")

    lines.append("")
    lines.append("Stage failure breakdown:")
    if summary.stage_failure_breakdown:
        lines.extend(
            f"  - {stage}: {count}"
            for stage, count in summary.stage_failure_breakdown.items()
        )
    else:
        lines.append("  - none")

    failed = [result for result in results if not result.acceptable_match][
        :max_failed_cases
    ]
    lines.append("")
    lines.append("Failed case summaries:")
    if failed:
        lines.extend(
            "  - "
            f"{result.case_id}: predicted={result.predicted_answer!r}, "
            f"expected={result.expected_answer!r}, failure_mode={result.failure_mode}"
            for result in failed
        )
    else:
        lines.append("  - none")

    return "\n".join(lines)


def export_json(
    results: list[CaseEvaluationResult],
    summary: EvaluationSummary,
    output_path: str | Path,
) -> Path:
    path = Path(output_path)
    payload = {
        "summary": asdict(summary),
        "results": [asdict(result) for result in results],
    }
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return path


def export_csv(results: list[CaseEvaluationResult], output_path: str | Path) -> Path:
    path = Path(output_path)
    fieldnames = [
        "case_id",
        "query",
        "predicted_answer",
        "expected_answer",
        "correct",
        "acceptable_match",
        "abstained",
        "failure_mode",
        "verification_summary",
        "reasoning_route",
        "trace_id",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for result in results:
            row = {name: getattr(result, name) for name in fieldnames}
            writer.writerow(row)
    return path
