"""SQLite-backed storage adapter for persistent CKE graph state."""

from __future__ import annotations

import json
import logging
import sqlite3
from pathlib import Path
from typing import List, Optional

logger = logging.getLogger(__name__)

from cke.models import Statement
from cke.storage.adapter import StorageAdapter


class SQLiteStore(StorageAdapter):
    """Persist graph entities, aliases, and statements in SQLite."""

    def __init__(self, db_path: str | Path) -> None:
        self.db_path = str(db_path)
        self._conn = sqlite3.connect(self.db_path)
        self._conn.row_factory = sqlite3.Row
        self._conn.execute("PRAGMA foreign_keys = ON")

    @staticmethod
    def _normalize_name(value: str) -> str:
        return " ".join(str(value).strip().lower().split())

    @staticmethod
    def _decode_context(raw_context: str | None) -> dict:
        if not raw_context:
            return {}
        try:
            parsed = json.loads(raw_context)
        except json.JSONDecodeError:
            return {}
        return parsed if isinstance(parsed, dict) else {}

    def close(self) -> None:
        self._conn.close()

    def __del__(self) -> None:  # pragma: no cover - defensive cleanup only
        try:
            self.close()
        except Exception:  # noqa: B110
            logger.debug(
                "Failed to close SQLite connection during cleanup", exc_info=True
            )

    def init_schema(self) -> None:
        self._conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS entities (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                canonical_name TEXT NOT NULL,
                normalized_name TEXT NOT NULL UNIQUE
            );

            CREATE TABLE IF NOT EXISTS aliases (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                alias TEXT NOT NULL,
                normalized_alias TEXT NOT NULL UNIQUE,
                entity_id INTEGER NOT NULL,
                FOREIGN KEY(entity_id) REFERENCES entities(id) ON DELETE CASCADE
            );

            CREATE TABLE IF NOT EXISTS statements (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                subject_entity_id INTEGER NOT NULL,
                relation TEXT NOT NULL,
                object_entity_id INTEGER NOT NULL,
                context_json TEXT,
                confidence REAL NOT NULL DEFAULT 1.0,
                source TEXT,
                timestamp TEXT,
                FOREIGN KEY(subject_entity_id)
                    REFERENCES entities(id) ON DELETE CASCADE,
                FOREIGN KEY(object_entity_id) REFERENCES entities(id) ON DELETE CASCADE,
                UNIQUE(subject_entity_id, relation, object_entity_id)
            );

            CREATE INDEX IF NOT EXISTS idx_statements_subject
                ON statements(subject_entity_id);
            CREATE INDEX IF NOT EXISTS idx_statements_object
                ON statements(object_entity_id);
            CREATE INDEX IF NOT EXISTS idx_statements_relation
                ON statements(relation);
            """
        )
        self._conn.commit()

    def clear(self) -> None:
        self._conn.execute("DELETE FROM statements")
        self._conn.execute("DELETE FROM aliases")
        self._conn.execute("DELETE FROM entities")
        self._conn.commit()

    def upsert_entity(self, canonical_name: str) -> int:
        normalized_name = self._normalize_name(canonical_name)
        self._conn.execute(
            """
            INSERT INTO entities (canonical_name, normalized_name)
            VALUES (?, ?)
            ON CONFLICT(normalized_name) DO UPDATE SET
                canonical_name = excluded.canonical_name
            """,
            (canonical_name, normalized_name),
        )
        self._conn.commit()
        entity_id = self.resolve_entity_by_name(canonical_name)
        if entity_id is None:  # pragma: no cover - defensive invariant check
            raise RuntimeError(f"failed to upsert entity: {canonical_name}")
        return entity_id

    def resolve_entity_by_name(self, name: str) -> Optional[int]:
        normalized_name = self._normalize_name(name)
        row = self._conn.execute(
            """
            SELECT id
            FROM entities
            WHERE normalized_name = ?
            """,
            (normalized_name,),
        ).fetchone()
        if row is not None:
            return int(row["id"])

        alias_row = self._conn.execute(
            """
            SELECT entity_id
            FROM aliases
            WHERE normalized_alias = ?
            """,
            (normalized_name,),
        ).fetchone()
        if alias_row is None:
            return None
        return int(alias_row["entity_id"])

    def add_alias(self, alias: str, entity_id: int) -> None:
        self._conn.execute(
            """
            INSERT INTO aliases (alias, normalized_alias, entity_id)
            VALUES (?, ?, ?)
            ON CONFLICT(normalized_alias) DO UPDATE SET
                alias = excluded.alias,
                entity_id = excluded.entity_id
            """,
            (alias, self._normalize_name(alias), entity_id),
        )
        self._conn.commit()

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
        subject_id = self.upsert_entity(subject)
        object_id = self.upsert_entity(object_)
        context_json = json.dumps(context or {}, sort_keys=True)
        self._conn.execute(
            """
            INSERT INTO statements (
                subject_entity_id,
                relation,
                object_entity_id,
                context_json,
                confidence,
                source,
                timestamp
            )
            VALUES (?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(subject_entity_id, relation, object_entity_id) DO UPDATE SET
                context_json = excluded.context_json,
                confidence = excluded.confidence,
                source = excluded.source,
                timestamp = excluded.timestamp
            """,
            (
                subject_id,
                relation,
                object_id,
                context_json,
                confidence,
                source,
                timestamp,
            ),
        )
        self._conn.commit()
        row = self._conn.execute(
            """
            SELECT id
            FROM statements
            WHERE subject_entity_id = ? AND relation = ? AND object_entity_id = ?
            """,
            (subject_id, relation, object_id),
        ).fetchone()
        if row is None:  # pragma: no cover - defensive invariant check
            raise RuntimeError("failed to upsert statement")
        return int(row["id"])

    def _statement_from_row(self, row: sqlite3.Row) -> Statement:
        return Statement(
            subject=row["subject_name"],
            relation=row["relation"],
            object=row["object_name"],
            context=self._decode_context(row["context_json"]),
            confidence=float(row["confidence"]),
            source=row["source"],
            timestamp=row["timestamp"],
            statement_id=str(row["id"]),
        )

    def get_neighbors(self, entity_name: str) -> List[Statement]:
        entity_id = self.resolve_entity_by_name(entity_name)
        if entity_id is None:
            return []
        rows = self._conn.execute(
            """
            SELECT
                st.id,
                subj.canonical_name AS subject_name,
                st.relation,
                obj.canonical_name AS object_name,
                st.context_json,
                st.confidence,
                st.source,
                st.timestamp
            FROM statements AS st
            JOIN entities AS subj ON subj.id = st.subject_entity_id
            JOIN entities AS obj ON obj.id = st.object_entity_id
            WHERE st.subject_entity_id = ?
            ORDER BY st.id ASC
            """,
            (entity_id,),
        ).fetchall()
        return [self._statement_from_row(row) for row in rows]

    def find_paths(
        self, entity_a: str, entity_b: str, cutoff: int = 3
    ) -> List[List[Statement]]:
        source_id = self.resolve_entity_by_name(entity_a)
        target_id = self.resolve_entity_by_name(entity_b)
        if source_id is None or target_id is None or cutoff < 1:
            return []

        adjacency: dict[int, list[tuple[int, Statement]]] = {}
        rows = self._conn.execute(
            """
            SELECT
                st.id,
                st.subject_entity_id,
                st.object_entity_id,
                subj.canonical_name AS subject_name,
                st.relation,
                obj.canonical_name AS object_name,
                st.context_json,
                st.confidence,
                st.source,
                st.timestamp
            FROM statements AS st
            JOIN entities AS subj ON subj.id = st.subject_entity_id
            JOIN entities AS obj ON obj.id = st.object_entity_id
            ORDER BY st.id ASC
            """
        ).fetchall()
        for row in rows:
            adjacency.setdefault(int(row["subject_entity_id"]), []).append(
                (int(row["object_entity_id"]), self._statement_from_row(row))
            )

        paths: list[list[Statement]] = []

        def dfs(current_id: int, seen: set[int], path: list[Statement]) -> None:
            if len(path) > cutoff:
                return
            if current_id == target_id and path:
                paths.append(list(path))
                return
            for next_id, statement in adjacency.get(current_id, []):
                if next_id in seen:
                    continue
                seen.add(next_id)
                path.append(statement)
                dfs(next_id, seen, path)
                path.pop()
                seen.remove(next_id)

        dfs(source_id, {source_id}, [])
        return paths

    def load_all_statements(self) -> List[Statement]:
        rows = self._conn.execute(
            """
            SELECT
                st.id,
                subj.canonical_name AS subject_name,
                st.relation,
                obj.canonical_name AS object_name,
                st.context_json,
                st.confidence,
                st.source,
                st.timestamp
            FROM statements AS st
            JOIN entities AS subj ON subj.id = st.subject_entity_id
            JOIN entities AS obj ON obj.id = st.object_entity_id
            ORDER BY st.id ASC
            """
        ).fetchall()
        return [self._statement_from_row(row) for row in rows]

    def all_entities(self) -> List[str]:
        rows = self._conn.execute(
            """
            SELECT canonical_name
            FROM entities
            ORDER BY canonical_name ASC
            """
        ).fetchall()
        return [str(row["canonical_name"]) for row in rows]
