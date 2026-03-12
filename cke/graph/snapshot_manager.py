"""Graph snapshot storage for convergence/drift monitoring."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone

from cke.models import Statement


@dataclass(slots=True)
class GraphSnapshot:
    timestamp: str
    assertion_hash_set: frozenset[str]
    assertion_count: int


class GraphSnapshotManager:
    """Store and load graph snapshots for fast set-based comparisons."""

    def __init__(self) -> None:
        self._snapshots: dict[str, GraphSnapshot] = {}

    @staticmethod
    def _statement_hash(statement: Statement) -> str:
        return statement.key().__repr__()

    @staticmethod
    def _all_statements(graph) -> list[Statement]:
        statements: list[Statement] = []
        for entity in graph.all_entities():
            statements.extend(graph.get_neighbors(entity))
        return statements

    def create_snapshot(self, graph, timestamp: str | None = None) -> GraphSnapshot:
        snapshot_ts = timestamp or datetime.now(timezone.utc).isoformat()
        statement_hashes = frozenset(
            self._statement_hash(statement) for statement in self._all_statements(graph)
        )
        snapshot = GraphSnapshot(
            timestamp=snapshot_ts,
            assertion_hash_set=statement_hashes,
            assertion_count=len(statement_hashes),
        )
        self._snapshots[snapshot_ts] = snapshot
        return snapshot

    def load_snapshot(self, timestamp: str) -> GraphSnapshot | None:
        return self._snapshots.get(timestamp)
