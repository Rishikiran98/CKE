"""Experiment report generation utilities."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


class ReportGenerator:
    def generate(
        self,
        output_path: str | Path,
        dataset_results: dict[str, Any],
        ablation_results: dict[str, Any],
        system_metrics: dict[str, Any],
        cost_stats: dict[str, Any],
    ) -> Path:
        output = Path(output_path)
        output.parent.mkdir(parents=True, exist_ok=True)

        lines = [
            "# CKE Benchmark Report",
            "",
            "## Dataset Results",
            "```json",
            json.dumps(dataset_results, indent=2),
            "```",
            "",
            "## Ablation Results",
            "```json",
            json.dumps(ablation_results, indent=2),
            "```",
            "",
            "## System Metrics",
            "```json",
            json.dumps(system_metrics, indent=2),
            "```",
            "",
            "## Cost Statistics",
            "```json",
            json.dumps(cost_stats, indent=2),
            "```",
            "",
        ]
        output.write_text("\n".join(lines), encoding="utf-8")
        return output
