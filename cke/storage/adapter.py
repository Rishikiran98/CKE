"""Abstract storage adapter interface for CKE graph persistence.

Implement this interface to add new storage backends
(e.g. SQLite, Neo4j, Postgres) without touching the graph engine.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import List, Optional

from cke.models import Statement


class StorageAdapter(ABC):
    """Minimal interface every CKE storage backend must satisfy."""

    # ------------------------------------------------------------------
    # Schema / lifecycle
    # ------------------------------------------------------------------

    @abstractmethod
    def init_schema(self) -> None:
        """Create all required tables / collections if they don't exist."""

    @abstractmethod
    def clear(self) -> None:
        """Remove all persisted data.  Useful for tests and resetting state."""

    # ------------------------------------------------------------------
    # Entity management
    # ------------------------------------------------------------------

    @abstractmethod
    def upsert_entity(self, canonical_name: str) -> int:
        """Ensure *canonical_name* exists and return its integer entity_id."""

    @abstractmethod
    def resolve_entity_by_name(self, name: str) -> Optional[int]:
        """Return entity_id for *name* (canonical or alias), or None."""

    @abstractmethod
    def add_alias(self, alias: str, entity_id: int) -> None:
        """Map *alias* -> *entity_id* (idempotent)."""

    # ------------------------------------------------------------------
    # Statement management
    # ------------------------------------------------------------------

    @abstractmethod
    def upsert_statement(
        self,
        subject: str,
        relation: str,
        object_: str,
        context: dict | None = None,
        confidence: float = 1.0,
        source: str | None = None,
        timestamp: str | None = None,
    ) -> int:
        """Persist a statement and return its statement_id.

        Repeated calls with the same (subject, relation, object_) triple
        should update metadata rather than duplicate the row.
        """

    # ------------------------------------------------------------------
    # Graph traversal
    # ------------------------------------------------------------------

    @abstractmethod
    def get_neighbors(self, entity_name: str) -> List[Statement]:
        """Return all statements where *entity_name* is the subject."""

    @abstractmethod
    def find_paths(
        self, entity_a: str, entity_b: str, cutoff: int = 3
    ) -> List[List[Statement]]:
        """Return all simple paths from *entity_a* to *entity_b* up to *cutoff* hops."""

    # ------------------------------------------------------------------
    # Bulk read
    # ------------------------------------------------------------------

    @abstractmethod
    def load_all_statements(self) -> List[Statement]:
        """Return every statement currently persisted in the store."""

    @abstractmethod
    def all_entities(self) -> List[str]:
        """Return sorted list of all canonical entity names."""
