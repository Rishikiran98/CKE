# Convergent Knowledge Engine (CKE)

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

## Project Layout

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

tests/
configs/
```

## Configuration

Main configuration file:

- `configs/config.yaml`

## Installation

```bash
pip install -r requirements.txt
```
