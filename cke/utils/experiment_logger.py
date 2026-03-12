"""Experiment logging utilities for JSONL-based run tracking."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


class ExperimentLogger:
    """Collect and persist experiment query logs as JSONL."""

    def __init__(self) -> None:
        self._logs: list[dict[str, Any]] = []

    @property
    def logs(self) -> list[dict[str, Any]]:
        """Return in-memory log entries."""
        return list(self._logs)

    def log_query(
        self,
        question: str,
        retrieved_items: list[dict[str, Any]] | list[str],
        answer: str,
        latency: float,
        tokens: int,
        correct: bool,
        retrieved_nodes: list[str] | None = None,
        gold_nodes: list[str] | None = None,
    ) -> None:
        """Append one query-level experiment record."""
        self._logs.append(
            {
                "question": question,
                "retrieval_size": len(retrieved_items),
                "answer": answer,
                "latency": float(latency),
                "tokens": int(tokens),
                "correct": bool(correct),
                "retrieved_nodes": list(retrieved_nodes or []),
                "gold_nodes": list(gold_nodes or []),
            }
        )

    def save_logs(self, path: str | Path) -> None:
        """Write logs in JSONL format to *path*."""
        output_path = Path(path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("w", encoding="utf-8") as handle:
            for entry in self._logs:
                handle.write(json.dumps(entry, ensure_ascii=False) + "\n")
