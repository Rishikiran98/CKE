"""SQLite-backed storage adapter for CKE.

Schema
------
entities  : entity_id (PK), canonical_name (UNIQUE), entity_type
aliases   : alias (PK), entity_id (FK -> entities)
statements: statement_id (PK), subject, relation, object, context (JSON),
            confidence, source, timestamp

Deduplication
-------------
* entities and aliases use INSERT OR IGNORE.
* statements use INSERT OR IGNORE on (subject, relation, object) and then
  UPDATE to refresh confidence/source/timestamp on conflict.
"""

from __future__ import annotations

import json
import sqlite3
from collections import deque
from pathlib import Path
from typing import Any, Dict, List, Optional

from cke.models import Statement
from cke.storage.adapter import StorageAdapter

_SCHEMA = """
PRAGMA journal_mode=WAL;
PRAGMA foreign_keys=ON;

CREATE TABLE IF NOT EXISTS entities (
    entity_id     INTEGER PRIMARY KEY AUTOINCREMENT,
    canonical_name TEXT    NOT NULL UNIQUE,
    entity_type    TEXT
);

CREATE TABLE IF NOT EXISTS aliases (
    alias      TEXT    NOT NULL PRIMARY KEY,
    entity_id  INTEGER NOT NULL REFERENCES entities(entity_id)
);

CREATE TABLE IF NOT EXISTS statements (
    statement_id INTEGER PRIMARY KEY AUTOINCREMENT,
    subject      TEXT    NOT NULL,
    relation     TEXT    NOT NULL,
    object       TEXT    NOT NULL,
    context      TEXT    NOT NULL DEFAULT '{}',
    confidence   REAL    NOT NULL DEFAULT 1.0,
    source       TEXT,
    timestamp    TEXT,
    UNIQUE (subject, relation, object)
);

CREATE INDEX IF NOT EXISTS idx_statements_subject ON statements(subject);
CREATE INDEX IF NOT EXISTS idx_statements_object  ON statements(object);
"""


class SQLiteStore(StorageAdapter):
    """SQLite-backed storage using only the Python standard library."""

    def __init__(self, db_path: str | Path) -> None:
        self._db_path = str(db_path)
        self._conn: sqlite3.Connection | None = None

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _connect(self) -> sqlite3.Connection:
        if self._conn is None:
            self._conn = sqlite3.connect(self._db_path, check_same_thread=False)
            self._conn.row_factory = sqlite3.Row
        return self._conn

    def _execute(
        self,
        sql: str,
        params: tuple = (),
    ) -> sqlite3.Cursor:
        return self._connect().execute(sql, params)

    def _commit(self) -> None:
        self._connect().commit()

    # ------------------------------------------------------------------
    # Schema / lifecycle
    # ------------------------------------------------------------------

    def init_schema(self) -> None:
        conn = self._connect()
        conn.executescript(_SCHEMA)
        conn.commit()

    def clear(self) -> None:
        conn = self._connect()
        conn.executescript(
            "DELETE FROM statements; DELETE FROM aliases; DELETE FROM entities;"
        )
        conn.commit()

    def close(self) -> None:
        """Close the underlying connection (optional clean-up)."""
        if self._conn is not None:
            self._conn.close()
            self._conn = None

    # ------------------------------------------------------------------
    # Entity management
    # ------------------------------------------------------------------

    def upsert_entity(self, canonical_name: str) -> int:
        """Ensure entity exists; return its entity_id."""
        self._execute(
            "INSERT OR IGNORE INTO entities (canonical_name) VALUES (?)",
            (canonical_name,),
        )
        self._commit()
        row = self._execute(
            "SELECT entity_id FROM entities WHERE canonical_name = ?",
            (canonical_name,),
        ).fetchone()
        return int(row["entity_id"])

    def resolve_entity_by_name(self, name: str) -> Optional[int]:
        """Resolve by canonical name first, then alias."""
        row = self._execute(
            "SELECT entity_id FROM entities WHERE canonical_name = ?",
            (name,),
        ).fetchone()
        if row:
            return int(row["entity_id"])
        row = self._execute(
            "SELECT entity_id FROM aliases WHERE alias = ?",
            (name,),
        ).fetchone()
        return int(row["entity_id"]) if row else None

    def add_alias(self, alias: str, entity_id: int) -> None:
        self._execute(
            "INSERT OR IGNORE INTO aliases (alias, entity_id) VALUES (?, ?)",
            (alias, entity_id),
        )
        self._commit()

    # ------------------------------------------------------------------
    # Statement management
    # ------------------------------------------------------------------

    def upsert_statement(
        self,
        subject: str,
        relation: str,
        object_: str,
        context: Dict[str, Any] | None = None,
        confidence: float = 1.0,
        source: str | None = None,
        timestamp: str | None = None,
    ) -> int:
        ctx_json = json.dumps(context or {})
        # Insert or ignore on the unique triple, then update metadata.
        self._execute(
            """
            INSERT INTO statements
                (subject, relation, object, context, confidence, source, timestamp)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(subject, relation, object) DO UPDATE SET
                context    = excluded.context,
                confidence = excluded.confidence,
                source     = excluded.source,
                timestamp  = excluded.timestamp
            """,
            (subject, relation, object_, ctx_json, confidence, source, timestamp),
        )
        self._commit()
        row = self._execute(
            """
            SELECT statement_id FROM statements
            WHERE subject = ? AND relation = ? AND object = ?
            """,
            (subject, relation, object_),
        ).fetchone()
        return int(row["statement_id"])

    # ------------------------------------------------------------------
    # Graph traversal
    # ------------------------------------------------------------------

    def get_neighbors(self, entity_name: str) -> List[Statement]:
        rows = self._execute(
            """
            SELECT subject, relation, object, context, confidence, source, timestamp
            FROM statements
            WHERE subject = ?
            """,
            (entity_name,),
        ).fetchall()
        return [self._row_to_statement(r) for r in rows]

    def find_paths(
        self, entity_a: str, entity_b: str, cutoff: int = 3
    ) -> List[List[Statement]]:
        """BFS over persisted statements; mirrors in-memory graph engine logic."""
        known = {
            r["subject"]
            for r in self._execute("SELECT DISTINCT subject FROM statements").fetchall()
        } | {
            r["object"]
            for r in self._execute("SELECT DISTINCT object FROM statements").fetchall()
        }

        if entity_a not in known or entity_b not in known:
            return []

        results: List[List[Statement]] = []
        queue: deque[tuple[str, list[Statement]]] = deque([(entity_a, [])])

        while queue:
            node, path = queue.popleft()
            if len(path) > cutoff:
                continue
            if node == entity_b and path:
                results.append(path)
                continue
            for st in self.get_neighbors(node):
                if any(step.object == st.object for step in path):
                    continue
                queue.append((st.object, path + [st]))

        return results

    # ------------------------------------------------------------------
    # Bulk read
    # ------------------------------------------------------------------

    def load_all_statements(self) -> List[Statement]:
        rows = self._execute(
            "SELECT subject, relation, object, context, confidence, source, timestamp"
            " FROM statements"
        ).fetchall()
        return [self._row_to_statement(r) for r in rows]

    def all_entities(self) -> List[str]:
        rows = self._execute(
            """
            SELECT DISTINCT name FROM (
                SELECT subject AS name FROM statements
                UNION
                SELECT object  AS name FROM statements
            ) ORDER BY name
            """
        ).fetchall()
        return [r["name"] for r in rows]

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _row_to_statement(row: sqlite3.Row) -> Statement:
        ctx = json.loads(row["context"] or "{}")
        return Statement(
            subject=row["subject"],
            relation=row["relation"],
            object=row["object"],
            context=ctx,
            confidence=float(row["confidence"]),
            source=row["source"],
            timestamp=row["timestamp"],
        )
