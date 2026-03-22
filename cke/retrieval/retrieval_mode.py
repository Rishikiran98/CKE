"""Retrieval mode configuration for query orchestration."""

from __future__ import annotations

from enum import Enum


class RetrievalMode(str, Enum):
    """Controls which retrieval strategy the orchestrator uses."""

    GRAPH_ONLY = "graph_only"
    DENSE_ONLY = "dense_only"
    HYBRID = "hybrid"
