# Convergent Knowledge Engine (CKE)

**Status: research prototype — no validated benchmark results.**

CKE is an experimental Python codebase for graph-based reasoning over text. It
extracts subject/relation/object statements from documents, stores them in a
knowledge graph, retrieves paths through that graph, and applies symbolic
operators and template reasoners to produce answers. A conversational layer
stores multi-turn dialogue and resolves references across turns.

Nothing in this repository has been measured against an external benchmark. Any
performance number you may have seen in an earlier revision was produced against
a synthetic corpus that the repository generated for itself, and has been
removed. Treat CKE as an architecture under construction, not as a system with
known accuracy, token cost, or latency characteristics.

## What currently works

- **Extraction** (`cke.extractor`): a rule-based extractor over a small set of
  sentence patterns, and an LLM-backed extractor that requires an API key.
- **Graph storage** (`cke.graph`, `cke.graph_engine`, `cke.storage`): statement
  storage over SQLite or Neo4j, with entity resolution and alias handling.
- **Retrieval** (`cke.retrieval`): graph path generation and scoring, a dense
  RAG baseline, and a hybrid mode combining the two.
- **Reasoning** (`cke.reasoning`): template and path reasoners, plus discrete
  operators such as `count`, `exists`, and date and numeric comparison.
- **Conversation** (`cke.conversation`, `cke.pipeline`): turn storage,
  conversational retrieval, reference resolution, and grounded answering.
- **Trust and observability** (`cke.trust`, `cke.observability`): confidence
  calibration over assertions and drift monitoring.

## Known limitations

These are stated here rather than left for a reader to discover.

- The package does not install. `pyproject.toml` has no `[project]` table, so
  `pip install -e .` does not work and `cke.sdk.client` has no installable
  package to talk to.
- `python demo.py` produces no answer. The demo corpus does not match the
  patterns the rule-based extractor recognises.
- Several components degrade silently when a dependency or API key is missing.
  The embedding model falls back to hashed bag-of-words vectors and the LLM
  extractor falls back to regexes, in both cases without an error.
- `cke.api.server` cannot be imported without FastAPI installed, has no
  authentication, and does not use the main query pipeline.
- Dependencies in `requirements.txt` are unpinned.
- There is no evaluation harness that can produce a trustworthy number. Building
  one is prerequisite work before any result is reported here.

## Repository layout

```text
cke/            library code (extraction, graph, retrieval, reasoning,
                conversation, trust, storage, API, SDK)
configs/        YAML configuration for retrieval ranking and trust
scripts/        dataset download and benchmark drivers
tests/          pytest suite
cke/tests/      a second, older pytest suite organised by sprint number
demo.py         demonstration entry point (currently returns no answer)
```

## Running the tests

The test suite needs `pytest`, `networkx`, `numpy`, `pydantic`, `PyYAML`, and
`rapidfuzz`. With those installed, from the repository root:

```bash
python -m pytest -q
```

198 tests pass. They cover module behaviour in isolation; none of them
establishes an end-to-end quality claim.

## Datasets

`scripts/download_datasets.py` fetches HotpotQA and 2WikiMultiHopQA through the
HuggingFace `datasets` library. It fails with an error if a dataset cannot be
obtained. It does not generate substitute data.

## License

See `LICENSE`.
