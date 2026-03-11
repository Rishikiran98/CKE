"""Utilities for trust/conflict reporting from graph data."""

from __future__ import annotations

from collections import Counter
from typing import Any

from cke.graph_engine.graph_engine import KnowledgeGraphEngine


def _all_statements(graph: KnowledgeGraphEngine) -> list:
    statements = []
    for entity in graph.all_entities():
        statements.extend(graph.get_neighbors(entity))
    return statements


def trust_distribution(graph: KnowledgeGraphEngine) -> dict[str, int]:
    """Bucket statement confidence values into coarse trust ranges."""
    buckets = Counter()
    for st in _all_statements(graph):
        score = float(st.confidence)
        if score < 0.25:
            buckets["low"] += 1
        elif score < 0.6:
            buckets["medium"] += 1
        else:
            buckets["high"] += 1
    return dict(buckets)


def top_trusted_assertions(
    graph: KnowledgeGraphEngine, limit: int = 10
) -> list[dict[str, Any]]:
    """Return top-K assertions sorted by trust/confidence."""
    statements = sorted(_all_statements(graph), key=lambda st: float(st.confidence), reverse=True)
    return [
        {
            "subject": st.subject,
            "relation": st.relation,
            "object": st.object,
            "trust_score": st.confidence,
            "source": st.source,
        }
        for st in statements[:limit]
    ]


def conflict_summary(graph: KnowledgeGraphEngine) -> dict[str, Any]:
    """Summarize potential conflicts found in stored statements."""
    grouped: dict[tuple[str, str], set[str]] = {}
    for st in _all_statements(graph):
        key = (st.subject, st.relation)
        grouped.setdefault(key, set()).add(st.object)

    conflicts = {
        f"{subject}::{relation}": sorted(list(objects))
        for (subject, relation), objects in grouped.items()
        if len(objects) > 1
    }
    return {"conflict_pairs": conflicts, "total": len(conflicts)}
