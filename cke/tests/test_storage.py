"""Tests for SQLite storage backend and graph engine persistence."""

from __future__ import annotations

import tempfile
from pathlib import Path

from cke.graph_engine.graph_engine import KnowledgeGraphEngine
from cke.storage.sqlite_store import SQLiteStore


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _tmp_db() -> Path:
    """Return a path to a fresh temporary SQLite file."""
    f = tempfile.NamedTemporaryFile(suffix=".db", delete=False)
    f.close()
    return Path(f.name)


# ---------------------------------------------------------------------------
# Task G-1: Schema initialisation
# ---------------------------------------------------------------------------


def test_init_schema_creates_tables():
    db = _tmp_db()
    store = SQLiteStore(db)
    store.init_schema()
    conn = store._connect()
    tables = {
        r[0]
        for r in conn.execute(
            "SELECT name FROM sqlite_master " "WHERE type='table'"
        ).fetchall()
    }
    assert "entities" in tables
    assert "aliases" in tables
    assert "statements" in tables
    store.close()


# ---------------------------------------------------------------------------
# Task G-2: Entity persistence across re-instantiation
# ---------------------------------------------------------------------------


def test_entity_persisted_across_restart():
    db = _tmp_db()

    engine1 = KnowledgeGraphEngine(db_path=db)
    engine1.add_statement("Redis", "uses", "RESP")

    # New engine instance, same DB file
    engine2 = KnowledgeGraphEngine(db_path=db)
    assert "Redis" in engine2.all_entities()
    assert "RESP" in engine2.all_entities()


# ---------------------------------------------------------------------------
# Task G-3: Alias persistence and canonical lookup
# ---------------------------------------------------------------------------


def test_alias_persisted_and_resolvable():
    db = _tmp_db()
    store = SQLiteStore(db)
    store.init_schema()

    eid = store.upsert_entity("Redis")
    store.add_alias("Redis DB", eid)
    store.add_alias("redis", eid)

    assert store.resolve_entity_by_name("Redis") == eid
    assert store.resolve_entity_by_name("Redis DB") == eid
    assert store.resolve_entity_by_name("redis") == eid
    assert store.resolve_entity_by_name("unknown") is None
    store.close()


def test_alias_idempotent():
    db = _tmp_db()
    store = SQLiteStore(db)
    store.init_schema()
    eid = store.upsert_entity("PubSub")
    # Inserting the same alias twice must not raise
    store.add_alias("pub-sub", eid)
    store.add_alias("pub-sub", eid)
    assert store.resolve_entity_by_name("pub-sub") == eid
    store.close()


# ---------------------------------------------------------------------------
# Task G-4: Statement persistence with full metadata
# ---------------------------------------------------------------------------


def test_statement_persistence_with_metadata():
    db = _tmp_db()
    store = SQLiteStore(db)
    store.init_schema()

    store.upsert_statement(
        "Redis",
        "implements",
        "RESP",
        context={"version": 3},
        confidence=0.9,
        source="rfc",
        timestamp="2024-01-01",
    )

    stmts = store.load_all_statements()
    assert len(stmts) == 1
    st = stmts[0]
    assert st.subject == "Redis"
    assert st.relation == "implements"
    assert st.object == "RESP"
    assert st.context == {"version": 3}
    assert abs(st.confidence - 0.9) < 1e-6
    assert st.source == "rfc"
    assert st.timestamp == "2024-01-01"
    store.close()


def test_statement_upsert_deduplicates():
    db = _tmp_db()
    store = SQLiteStore(db)
    store.init_schema()

    store.upsert_statement("A", "knows", "B", confidence=0.5)
    store.upsert_statement("A", "knows", "B", confidence=0.9)

    stmts = store.load_all_statements()
    # Must have exactly one row, with the updated confidence
    assert len(stmts) == 1
    assert abs(stmts[0].confidence - 0.9) < 1e-6
    store.close()


# ---------------------------------------------------------------------------
# Task G-5: Neighbor / path retrieval from persisted storage
# ---------------------------------------------------------------------------


def test_get_neighbors_from_storage():
    db = _tmp_db()
    store = SQLiteStore(db)
    store.init_schema()

    store.upsert_statement("Redis", "supports", "PubSub")
    store.upsert_statement("Redis", "uses", "RESP")

    neighbors = store.get_neighbors("Redis")
    relations = {st.relation for st in neighbors}
    assert "supports" in relations
    assert "uses" in relations
    store.close()


def test_find_paths_from_storage():
    db = _tmp_db()
    store = SQLiteStore(db)
    store.init_schema()

    store.upsert_statement("Redis", "supports", "PubSub")
    store.upsert_statement("PubSub", "implemented_via", "RESP")

    paths = store.find_paths("Redis", "RESP", cutoff=3)
    assert len(paths) >= 1
    assert paths[0][0].subject == "Redis"
    assert paths[0][-1].object == "RESP"
    store.close()


def test_engine_get_neighbors_after_reload():
    db = _tmp_db()

    engine1 = KnowledgeGraphEngine(db_path=db)
    engine1.add_statement("Redis", "uses", "RESP", confidence=0.95)

    engine2 = KnowledgeGraphEngine(db_path=db)
    neighbors = engine2.get_neighbors("Redis")
    assert len(neighbors) == 1
    assert neighbors[0].relation == "uses"
    assert abs(neighbors[0].confidence - 0.95) < 1e-6


# ---------------------------------------------------------------------------
# Task G-6: Backward compatibility – in-memory mode unchanged
# ---------------------------------------------------------------------------


def test_in_memory_mode_unchanged():
    """No db_path = original in-memory behaviour; no side effects."""
    engine = KnowledgeGraphEngine()
    engine.add_statement("A", "rel", "B")
    assert engine.get_neighbors("A")[0].object == "B"
    # Confirm storage is not present
    assert engine._storage is None


def test_clear_wipes_all_data():
    db = _tmp_db()
    store = SQLiteStore(db)
    store.init_schema()
    store.upsert_entity("Redis")
    store.upsert_statement("Redis", "uses", "RESP")
    store.clear()
    assert store.all_entities() == []
    assert store.load_all_statements() == []
    store.close()
