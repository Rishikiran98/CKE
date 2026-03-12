"""Experiment report generation utilities."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


class ReportGenerator:
    @staticmethod
    def _path_completeness_summary(ablation_results: dict[str, Any]) -> str:
        """Return a readable summary for path completeness across variants."""
        lines: list[str] = []
        for variant, payload in ablation_results.items():
            metrics = payload.get("metrics", {})
            if "path_completeness" in metrics:
                lines.append(f"- {variant}: {metrics['path_completeness']:.3f}")
        return "\n".join(lines) if lines else "- Not available"

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
            "### Path Completeness",
            self._path_completeness_summary(ablation_results),
            "",
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
