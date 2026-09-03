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
- **Diagnostics** (`cke.diagnostics`): the degradation contract described
  below, and `environment_report()`, which reports which optional dependencies
  resolved, which models loaded, and which components are degraded.

## Degradation is never silent

Two things can make a CKE number untrustworthy, and neither happens quietly.

A component can run with reduced capability when an optional dependency, a
model, an API key, or a config file is missing. And a value can be substituted
where nothing was measured: a retrieval result with no score, a statement with
no trust, an evaluated row with no prediction, a stored context that will not
decode. Twenty-six components declare both kinds.
Any such component warns at `WARNING` naming the specific cause, sets an
inspectable `degraded` flag with a `degraded_reason`, and raises
`DegradedComponentError` instead of degrading when constructed with
`strict=True`.

Every benchmark, evaluation and experiment entry point constructs its
components with `strict=True`, prints the environment report before it starts,
and prints a degradation summary when it finishes. In an environment without
`sentence-transformers`, running one exits non-zero and names the missing
component rather than reporting a number produced by a hash function. Pass
`--allow-degraded` to opt out deliberately.

Three entry points are deliberately outside this: `demo.py`, `cke.api.server`,
and the deprecation shim in `cke.router.entity_linker`. None of them reports a
measurement. Where those substitute a value anyway, the constant is named and
documented at its definition rather than written inline.

## Known limitations

These are stated here rather than left for a reader to discover.

- The package does not install. `pyproject.toml` has no `[project]` table, so
  `pip install -e .` does not work and `cke.sdk.client` has no installable
  package to talk to.
- `python demo.py` produces no answer. The demo corpus does not match the
  patterns the rule-based extractor recognises.
- `cke.api.server` has no authentication and does not use the main query
  pipeline. It is importable without FastAPI, but `create_app()` needs it.
- Dependencies in `requirements.txt` are unpinned.
- There is no evaluation harness that can produce a trustworthy number. Building
  one is prerequisite work before any result is reported here. In particular
  several figures the code computes are not measurements: prompt-token counts
  in `scripts/run_cke_benchmark.py` are word counts multiplied by 1.3 rather
  than tokenizer output, and confidence values are in places substituted
  constants rather than scores.
- `cke/experiments/retrieval_eval_pipeline.py` cannot be imported at all: it
  hard-imports `faiss`, `pandas` and `sentence_transformers`.
- There are two test directories, `tests/` and `cke/tests/`.

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

The suite passes; the command above reports the current count. These tests
cover module behaviour in isolation; none of them establishes an end-to-end
quality claim.

## Datasets

`scripts/download_datasets.py` fetches HotpotQA and 2WikiMultiHopQA through the
HuggingFace `datasets` library. It fails with an error if a dataset cannot be
obtained. It does not generate substitute data.

## License

See `LICENSE`.
