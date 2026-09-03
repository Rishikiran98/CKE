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

Two entry points are deliberately outside this: `cke.api.server` and the
deprecation shim in `cke.router.entity_linker`. Neither reports a measurement.
Where those substitute a value anyway, the constant is named and documented at
its definition rather than written inline. `demo.py` constructs its components
strict as well.

## Known limitations

These are stated here rather than left for a reader to discover.

- `cke.sdk.client` is a client for `cke.api.server`, which has no
  authentication and does not use the main query pipeline (next bullet).
- `cke.api.server` has no authentication and does not use the main query
  pipeline. It is importable without FastAPI, but `create_app()` needs it.
- Dependencies in `requirements.txt` are unpinned.
- There is no evaluation harness that can produce a trustworthy number. Building
  one is prerequisite work before any result is reported here. Prompt-token
  counts in `scripts/run_cke_benchmark.py` are now real counts from `tiktoken`
  under a named encoding, and the benchmark refuses to run without it, but
  confidence values are in places substituted constants rather than scores, and
  no language model answers on either arm.
- `cke/experiments/retrieval_eval_pipeline.py` measures Recall@k by matching
  retrieved MS MARCO titles against HotpotQA and LoCoMo relevance hints. That
  is a title-matching proxy, not a judged relevance set, and it needs the MS
  MARCO full-document TSV, HotpotQA and LoCoMo files to run.

## Repository layout

```text
cke/            library code (extraction, graph, retrieval, reasoning,
                conversation, trust, storage, API, SDK)
configs/        YAML configuration for retrieval ranking and trust
scripts/        dataset download and benchmark drivers
tests/          pytest suite
demo.py         extract five sentences, retrieve, and answer one question
```

## Installation

From a clone, with Python 3.11 or later:

```bash
pip install -e .
```

That installs the library and its dependencies from `requirements.txt`, each
bounded below by a verified version and above by the next major release (the
next minor for a library still at 0.x), and four console commands for the
in-package entry points: `cke-eval`,
`cke-experiment`, `cke-reasoning-eval` and `cke-retrieval-eval`. The drivers
under `scripts/` and `demo.py` are run as files from the repository root.
Components that read `configs/*.yaml` look for that directory relative to the
working directory, so run them from the repository root too.

## Running the demo

```bash
python demo.py
```

It extracts five sentences with the rule extractor, builds a graph, retrieves
around the entity the question names, and answers "Which country is Hagia
Sophia located in?" by a two-hop `located_in` chain, printing the reasoning
trace above the answer. Every component runs strict, so it needs
`sentence-transformers` and downloads the MiniLM embedding model on first
use. It is a demonstration of the pipeline's shape, not of general question
answering: the reasoner resolves a target relation only from "located in",
"nationality", or a relation named as the question's last word. CI runs the
demo and fails if the answer or the rule application changes.

## Running the tests

The tests run against the installed package. The `dev` extra adds the
checking tools from `requirements-dev.txt` (pytest, black, flake8, mypy,
bandit) at the versions CI uses:

```bash
pip install -e ".[dev]"
python -m pytest -q
```

CI also runs `black --check`, `flake8` and `mypy`, and measures statement
coverage of the library with `pytest --cov=cke`. mypy checks the modules
listed under `[tool.mypy]` in `pyproject.toml`; that list is a ratchet, added
to when a module checks clean and never shortened. Coverage has a floor under
`[tool.coverage.report]` that a run must clear; it was set at the measured
value when enforcement began and only rises.

The suite passes; the command above reports the current count. These tests
cover module behaviour in isolation; none of them establishes an end-to-end
quality claim.

## Datasets

`scripts/download_datasets.py` fetches HotpotQA and 2WikiMultiHopQA through the
HuggingFace `datasets` library. It fails with an error if a dataset cannot be
obtained. It does not generate substitute data.

## License

See `LICENSE`.
