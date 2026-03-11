# Convergent Knowledge Engine (CKE)

Convergent Knowledge Engine (CKE) is a research infrastructure project for graph-based reasoning over knowledge.

## Architecture

CKE is organized with clear separation between core system stages:
CKE is a Python 3.11 research framework for graph-based reasoning with a modular architecture.

## Architecture

CKE is organized with a clean separation of responsibilities:

- **Ingestion**: `cke/extractor/` for parsing and extracting structured statements from source text.
- **Memory**: `cke/graph/` for graph-backed knowledge representation and state.
- **Retrieval**: `cke/retrieval/` for selecting relevant graph context for a query.
- **Reasoning**: `cke/reasoning/` for answer synthesis over retrieved evidence.

Supporting modules:

- `cke/router/` query routing and orchestration
- `cke/datasets/` dataset adapters/loaders
- `cke/evaluation/` evaluation logic and metrics
- `cke/experiments/` experiment runners and pipelines
- `cke/utils/` shared utility helpers

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
