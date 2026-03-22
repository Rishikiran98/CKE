from __future__ import annotations

import subprocess
import sys
from pathlib import Path

from cke.graph_engine.graph_engine import KnowledgeGraphEngine
from cke.storage.sqlite_store import SQLiteStore


def test_sqlite_store_implements_storage_adapter_contract(tmp_path: Path):
    db_path = tmp_path / "store.db"
    store = SQLiteStore(db_path)
    store.init_schema()

    redis_id = store.upsert_entity("Redis")
    assert store.resolve_entity_by_name("Redis") == redis_id
    assert store.resolve_entity_by_name("redis") == redis_id

    store.add_alias("redis-cache", redis_id)
    assert store.resolve_entity_by_name("redis-cache") == redis_id

    statement_id = store.upsert_statement(
        "Redis",
        "supports",
        "PubSub",
        context={"source_type": "demo", "valid_from": "2024-01-01"},
        confidence=0.9,
        source="unit-test",
        timestamp="2025-01-02T00:00:00Z",
    )
    assert statement_id > 0

    updated_statement_id = store.upsert_statement(
        "Redis",
        "supports",
        "PubSub",
        context={"source_type": "updated"},
        confidence=0.95,
        source="unit-test-updated",
        timestamp="2025-01-03T00:00:00Z",
    )
    assert updated_statement_id == statement_id

    store.upsert_statement("PubSub", "implemented_via", "RESP")

    neighbors = store.get_neighbors("redis-cache")
    assert len(neighbors) == 1
    assert neighbors[0].subject == "Redis"
    assert neighbors[0].relation == "supports"
    assert neighbors[0].object == "PubSub"
    assert neighbors[0].context == {"source_type": "updated"}
    assert neighbors[0].confidence == 0.95
    assert neighbors[0].source == "unit-test-updated"
    assert neighbors[0].timestamp == "2025-01-03T00:00:00Z"

    paths = store.find_paths("Redis", "RESP", cutoff=2)
    assert len(paths) == 1
    assert [edge.object for edge in paths[0]] == ["PubSub", "RESP"]

    statements = store.load_all_statements()
    assert len(statements) == 2
    assert [statement.as_text() for statement in statements] == [
        "Redis supports PubSub",
        "PubSub implemented_via RESP",
    ]

    assert store.all_entities() == ["PubSub", "RESP", "Redis"]

    store.clear()
    assert store.load_all_statements() == []
    assert store.all_entities() == []


def test_knowledge_graph_engine_persists_and_reloads_from_sqlite(tmp_path: Path):
    db_path = tmp_path / "graph.db"

    graph = KnowledgeGraphEngine(db_path=db_path)
    graph.add_statement("Redis", "supports", "PubSub")
    graph.add_statement("PubSub", "implemented_via", "RESP")

    reloaded = KnowledgeGraphEngine(db_path=db_path)

    redis_neighbors = reloaded.get_neighbors("Redis")
    assert len(redis_neighbors) == 1
    assert redis_neighbors[0].object == "PubSub"

    paths = reloaded.find_paths("Redis", "RESP", cutoff=2)
    assert len(paths) == 1
    assert [edge.object for edge in paths[0]] == ["PubSub", "RESP"]


def test_demo_supports_db_path_end_to_end(tmp_path: Path):
    db_path = tmp_path / "demo.db"
    result = subprocess.run(
        [sys.executable, "demo.py", "--db-path", str(db_path)],
        cwd=Path(__file__).resolve().parents[2],
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert db_path.exists()
    assert f"DB path: {db_path}" in result.stdout
    assert "Answer:" in result.stdout
