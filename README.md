# Convergent Knowledge Engine (CKE)

Convergent Knowledge Engine (CKE) is a research infrastructure project for graph-based reasoning over knowledge.

## Architecture

CKE is organized with clear separation between core system stages:

- **Ingestion** (`extractor`): semantic extraction pipeline.
- **Memory** (`graph`): contextual knowledge graph storage and memory.
- **Retrieval** (`retrieval`): sparse retrieval and baseline RAG components.
- **Reasoning** (`reasoning`): reasoning engine over retrieved evidence.

Additional modules support datasets, routing, evaluation, experiments, and shared utilities.

## Repository Structure

```text
cke/
  datasets/
  extractor/
  graph/
  retrieval/
  reasoning/
  router/
  evaluation/
  experiments/
  utils/
  __init__.py
  version.py

configs/
  config.yaml

tests/
  __init__.py
```
