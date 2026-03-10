# Convergent Knowledge Engine (CKE)

CKE is a research prototype showing how **structured knowledge graphs + sparse retrieval** can answer questions with less context than traditional dense RAG.

## Features

- Rule-based semantic extraction (`extractor`)
- Optional LLM-based semantic extraction with rule fallback (`extractor/llm_extractor.py`)
- Entity resolution using string + embedding similarity (`entity_resolution`)
- Knowledge graph engine powered by NetworkX (`graph_engine`)
- Query-to-entity routing (`router`)
- BFS graph retrieval for minimal context (`retrieval/retriever.py`)
- Template-based reasoning (`reasoning`)
- Baseline RAG retriever with embeddings + FAISS fallback (`retrieval/rag_baseline.py`)
- Experiment runner comparing CKE vs RAG (`experiments/run_experiment.py`)
- Unit tests for core behavior (`tests`)

## Project Layout

```text
cke/
  extractor/
  entity_resolution/
  graph_engine/
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

## Run Demo

```bash
python demo.py
python demo.py --extractor rule
python demo.py --extractor llm
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
