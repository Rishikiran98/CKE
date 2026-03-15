"""Reasoning-trace capture for observability and debugging."""

from __future__ import annotations

import json
import time
import uuid
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

TRACE_DIR = Path("cke/logs/reasoning_traces")


@dataclass(slots=True)
class ReasoningTrace:
    """Structured reasoning trace used across retrieval/reasoning/extraction."""

    query: str
    entities: list[str] = field(default_factory=list)
    retrieved_facts: list[dict[str, Any]] = field(default_factory=list)
    graph_paths: list[list[dict[str, Any]]] = field(default_factory=list)
    operators_used: list[str] = field(default_factory=list)
    confidence_score: float = 0.0
    final_answer: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class ReasoningTraceLogger:
    """Filesystem-backed trace logger."""

    def __init__(self, trace_dir: Path | None = None) -> None:
        self.trace_dir = trace_dir or TRACE_DIR
        self.trace_dir.mkdir(parents=True, exist_ok=True)

    def write(self, trace: ReasoningTrace, stage: str = "reasoning") -> Path:
        ts = int(time.time() * 1000)
        suffix = uuid.uuid4().hex[:8]
        path = self.trace_dir / f"{ts}_{stage}_{suffix}.json"
        payload = trace.to_dict()
        payload["stage"] = stage
        payload["logged_at"] = ts
        path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
        return path
