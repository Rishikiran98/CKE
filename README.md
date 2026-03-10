# Convergent Knowledge Engine (CKE)

CKE is a research prototype showing how **structured knowledge graphs + sparse retrieval** can answer questions with less context than traditional dense RAG.

## Features

- Rule-based semantic extraction (`extractor`)
- Optional LLM-based semantic extraction with rule fallback (`extractor/llm_extractor.py`)
- Canonical entity resolution (aliases + hybrid string/embedding similarity) (`entity_resolution`)
- Knowledge graph engine powered by NetworkX (`graph_engine`)
- **SQLite-backed persistence** – save and reload the graph across restarts (`storage`)
- Query-to-entity routing (`router`)
- BFS graph retrieval for minimal context (`retrieval/retriever.py`)
- Template-based reasoning (`reasoning`)
- Optional LLM-based reasoner with template fallback (`reasoning/llm_reasoner.py`)
- Baseline RAG retriever with embeddings + FAISS fallback (`retrieval/rag_baseline.py`)
- Experiment runner comparing CKE vs RAG (`experiments/run_experiment.py`)
- Unit tests for core behavior (`tests`)

## Project Layout

```text
cke/
  extractor/
  entity_resolution/
  graph_engine/
  storage/
  router/
  retrieval/
  reasoning/
  experiments/
  tests/
```

## Python Version

- Python 3.11

## Dependencies

- networkx
- sentence-transformers
- faiss-cpu
- numpy
- pytest
- pydantic
- fastapi (optional)

Install with pip:

```bash
pip install networkx sentence-transformers faiss-cpu numpy pytest pydantic fastapi
```


## Canonical Entity Resolution

`EntityResolver` now maps multiple surface forms to a single canonical entity.

Example:
- `Redis`
- `Redis DB`
- `Redis database`

All resolve to the same canonical node when possible using:
- alias registration
- canonical-form normalization
- hybrid string + embedding similarity (with deterministic offline fallback embeddings)

## Statement Schema

`Statement` supports core triples plus contextual metadata:

- `subject: str`
- `relation: str`
- `object: str`
- `context: dict = {}`
- `confidence: float = 1.0`
- `source: str | None = None`
- `timestamp: str | None = None`

Existing code remains backward compatible: creating `Statement(subject, relation, object)` still works.

## Confidence-aware Graph Retrieval

Graph retrieval remains bounded BFS but results are ranked to prefer:
1. higher-confidence statements
2. shorter path distance from detected query entities

This keeps retrieval sparse while prioritizing higher quality evidence.


## Reasoner Modes

CKE supports two reasoner modes:

- `template` (default): deterministic, offline-safe heuristic reasoning
- `llm`: OpenAI-compatible LLM reasoning grounded on retrieved graph statements

LLM reasoner prompt behavior:
- constrained to provided graph context
- asks model to report insufficient evidence instead of guessing
- includes evidence statements in prompt
- requests JSON output: `{"answer": "...", "used_evidence": [...]}`

Fallback behavior for reliability:
- if API key is missing
- if optional client import fails
- if request fails
- if model output is malformed

Then CKE automatically falls back to `TemplateReasoner`.

## Persistence

By default, CKE stores the knowledge graph **in-memory only**; data is lost when the process exits.  Pass `--db-path` to enable a **SQLite-backed persistent store** that survives restarts.

### Storage module layout

```text
cke/storage/
  adapter.py       # Abstract StorageAdapter interface
  sqlite_store.py  # SQLite backend (stdlib sqlite3 only)
```

### How it works

- `KnowledgeGraphEngine(db_path="path/to/cke.db")` initialises the SQLite backend and pre-warms the in-memory graph from stored statements.
- Every `add_statement` call writes through to both the in-memory graph and the database.
- On the next startup, the same `db_path` reloads all persisted statements automatically.

### SQLite schema

| Table | Key columns |
|---|---|
| `entities` | `entity_id`, `canonical_name`, `entity_type` |
| `aliases` | `alias`, `entity_id` |
| `statements` | `subject`, `relation`, `object`, `context` (JSON), `confidence`, `source`, `timestamp` |

## Run Demo

```bash
python demo.py
python demo.py --extractor rule
python demo.py --extractor llm
python demo.py --reasoner template
python demo.py --reasoner llm

# Persistent mode – graph is saved and reloaded across restarts
python demo.py --db-path .\data\cke.db
python demo.py --extractor llm --reasoner llm --db-path .\data\cke.db
```

Expected flow:

1. Ingest small corpus
2. Build knowledge graph
3. Ask a query
4. Print reasoning path and answer

## Run Tests

```bash
pytest cke/tests -q
```

## Run Experiment

```bash
python -m cke.experiments.run_experiment
python -m cke.experiments.run_experiment --extractor llm
python -m cke.experiments.run_experiment --reasoner template
python -m cke.experiments.run_experiment --reasoner llm
```

Optional dataset input:

```bash
python -m cke.experiments.run_experiment --dataset path/to/hotpot_sample.json
```

Dataset format (list of dicts):

```json
[
  {
    "question": "What protocol does Redis pub/sub use?",
    "context": "Redis supports PubSub. PubSub implemented_via RESP.",
    "answer": "RESP"
  }
]
```
