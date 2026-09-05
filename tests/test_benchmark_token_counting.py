"""One counter, every arm.

A token figure must not differ between arms because of how it was counted.
The three pipelines used to build a counter apiece; they now take one, and
cannot be constructed without it.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

from cke.evaluation.span_qa import SpanExtractiveQA
from cke.evaluation.token_counter import TokenCounter

ROOT = Path(__file__).resolve().parents[1]


def _load_module():
    path = ROOT / "scripts" / "run_cke_benchmark.py"
    spec = importlib.util.spec_from_file_location("run_cke_benchmark", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


bench = _load_module()

PIPELINES = ("RAGPipeline", "CKELitePipeline", "HybridPipeline")


@pytest.mark.parametrize("name", PIPELINES)
def test_a_pipeline_cannot_be_built_without_a_counter(name):
    with pytest.raises(TypeError):
        getattr(bench, name)(strict=False)


@pytest.mark.needs_download
def test_every_arm_counts_with_the_same_object():
    counter = TokenCounter(strict=True)
    built = [
        getattr(bench, name)(
            token_counter=counter, answerer=SpanExtractiveQA(), strict=False
        )
        for name in PIPELINES
    ]

    assert all(pipeline._counter is counter for pipeline in built)


def test_the_script_no_longer_defines_a_word_count_estimator():
    """The multiplier is gone from the driver, not merely unused.

    It existed twice — as a class and as a module-level helper — and a copy
    left behind is a copy that gets called again.

    This is an anti-resurrection check rather than a behavioural one: the
    property is that certain code does not exist, and there is no run that
    can demonstrate the absence of a definition. It reads the parse tree
    rather than the text, so a mention in a comment or a docstring no longer
    fails it and a reformatting no longer passes it.
    """
    import ast

    tree = ast.parse(
        (ROOT / "scripts" / "run_cke_benchmark.py").read_text(encoding="utf-8")
    )

    defined = {
        node.name
        for node in ast.walk(tree)
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))
    }
    assert "_token_count_static" not in defined
    assert "WORDS_TO_TOKENS" not in defined
    assert not any(
        isinstance(node, ast.Name) and node.id == "WORDS_TO_TOKENS"
        for node in ast.walk(tree)
    )

    multiplications = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.BinOp)
        and isinstance(node.op, ast.Mult)
        and any(
            isinstance(side, ast.Constant) and side.value == 1.3
            for side in (node.left, node.right)
        )
    ]
    assert multiplications == [], "a word-count multiplier is back in the driver"


@pytest.mark.needs_download
def test_the_summary_names_the_counter_that_produced_its_figures():
    counter = TokenCounter(strict=True)
    summary = bench.produce_summary(
        {"rag_k10": {"median_tokens": 100.0}, "cke_n12": {"median_tokens": 10.0}},
        counter,
        SpanExtractiveQA(),
    )

    assert summary["prompt_token_counter"] == counter.description
    assert summary["prompt_token_figures_are_estimates"] is False


def test_truncation_is_aggregated_per_arm():
    """One answerer serves every arm; its shared total cannot say which arm
    was cut. The per-item flag each pipeline records can."""
    rows = [
        {
            "rag_k10": {
                "em": 0,
                "f1": 0,
                "prompt_tokens": 1200,
                "latency_ms": 1,
                "answer_truncated": True,
            },
            "cke_n12": {
                "em": 0,
                "f1": 0,
                "prompt_tokens": 40,
                "latency_ms": 1,
                "answer_truncated": False,
            },
        },
        {
            "rag_k10": {
                "em": 0,
                "f1": 0,
                "prompt_tokens": 900,
                "latency_ms": 1,
                "answer_truncated": True,
            },
            "cke_n12": {
                "em": 0,
                "f1": 0,
                "prompt_tokens": 50,
                "latency_ms": 1,
                "answer_truncated": False,
            },
        },
    ]
    agg = bench.aggregate_metrics(rows)

    assert agg["rag_k10"]["truncated_items"] == 2
    assert agg["rag_k10"]["truncation_rate"] == 1.0
    assert agg["cke_n12"]["truncated_items"] == 0


def test_a_scored_row_keeps_the_truncation_the_pipeline_recorded():
    """Six hand-written row literals each dropped this, so per-arm truncation
    read zero against a shared total of 596. The aggregation test above could
    not see it: it fed rows straight to aggregate_metrics, bypassing the row
    builder that was losing the field."""
    raw = {
        "answer": "Elon Musk",
        "prompt_tokens": 1250,
        "latency_ms": 3.0,
        "answer_truncated": True,
        "answer_dropped_tokens": 700,
        "n_statements": 4,
    }
    row = bench._score_row(raw, "Elon Musk", "n_statements")

    assert row["answer_truncated"] is True
    assert row["answer_dropped_tokens"] == 700
    assert row["n_statements"] == 4
    assert row["em"] == 1.0


def test_an_answerer_that_cannot_truncate_records_that_nothing_was_cut():
    """The span baseline reads the whole context, so False there is a fact."""
    truncated, dropped = bench._truncation_of(SpanExtractiveQA())

    assert truncated is False
    assert dropped == 0

    raw = {
        "answer": "x",
        "prompt_tokens": 5,
        "latency_ms": 1.0,
        "answer_truncated": truncated,
        "answer_dropped_tokens": dropped,
    }
    row = bench._score_row(raw, "x")

    assert row["answer_truncated"] is False
    assert row["answer_dropped_tokens"] == 0


def test_an_answerer_that_cannot_measure_records_no_figure_at_all():
    """False here would travel into every aggregate as a measured zero."""
    truncated, dropped = bench._truncation_of(_UnmeasuringAnswerer())

    assert truncated is None
    assert dropped is None


def test_a_row_that_never_recorded_truncation_does_not_claim_none_was_cut():
    raw = {"answer": "x", "prompt_tokens": 5, "latency_ms": 1.0}
    row = bench._score_row(raw, "x")

    assert row["answer_truncated"] is None
    assert row["answer_dropped_tokens"] is None


# ---------------------------------------------------------------------------
# The run's strictness reaches the datasets
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "loader, filename, records",
    [
        (
            "_load_hotpotqa",
            "hotpotqa_dev.json",
            [{"_id": "h", "question": "Q?", "answer": "A", "context": [["only"]]}],
        ),
        (
            "_load_wiki2",
            "wiki2_dev.json",
            [{"_id": "w", "question": "Q?", "answer": "A", "context": [["only"]]}],
        ),
        (
            "_load_musique",
            "musique_dev.json",
            [
                {
                    "id": "2hop__1",
                    "question": "Q?",
                    "answer": "A",
                    "paragraphs": [{"idx": 0, "title": "T", "paragraph_text": "  "}],
                }
            ],
        ),
    ],
)
def test_a_strict_run_refuses_a_dataset_that_drops_entries(
    tmp_path, loader, filename, records
):
    """Every loader built itself non-strict, whatever the run was.

    A strict run therefore loaded its data non-strict: the dataset declared
    that it had dropped malformed entries and the run carried on to report
    metrics computed over fewer documents than the items hold.
    """
    import json

    from cke.diagnostics import DegradedComponentError

    path = tmp_path / filename
    path.write_text(json.dumps(records), encoding="utf-8")

    with pytest.raises(DegradedComponentError):
        getattr(bench, loader)(path, 10, True, bench.SAMPLE_SEED, "sample")

    # The opt-out still loads, with the item thinned and the drop declared.
    items, provenance = getattr(bench, loader)(
        path, 10, False, bench.SAMPLE_SEED, "sample"
    )
    assert items[0]["documents"] == []
    assert provenance["items_evaluated"] == 1


# ---------------------------------------------------------------------------
# The summary says what produced the answers
# ---------------------------------------------------------------------------


class _StubTruncation:
    calls = 12
    truncated = 4
    rate = 4 / 12
    dropped_tokens: list = []


class _MeasuringAnswerer:
    description = "a model that fits the context itself"
    uses_language_model = True
    truncation_measured = True
    truncation = _StubTruncation()


class _UnmeasuringAnswerer(_MeasuringAnswerer):
    description = "a model behind an endpoint"
    truncation_measured = False


def test_the_summary_records_whether_a_model_read_the_context():
    """Without it the accuracy columns could be a lexical baseline's own."""
    from cke.evaluation.span_qa import SpanExtractiveQA

    counter = TokenCounter()
    combined = {"rag_k10": {"em": 0.3}, "cke_n12": {"em": 0.1}}

    with_model = bench.produce_summary(combined, counter, _MeasuringAnswerer())
    without = bench.produce_summary(combined, counter, SpanExtractiveQA())

    assert with_model["generator_in_the_loop"] is True
    assert without["generator_in_the_loop"] is False


def test_an_answerer_that_cannot_measure_truncation_reports_that():
    """A zero here would be a substituted value where a measurement belongs.

    The api backend sends the context whole because it has no tokeniser for
    the model behind the endpoint, so its truncated count stays zero whatever
    the endpoint did with an over-long prompt.
    """
    counter = TokenCounter()
    combined = {"rag_k10": {"truncated_items": 0, "truncation_rate": 0.0}}

    reported = bench.produce_summary(combined, counter, _UnmeasuringAnswerer())
    truncation = reported["answer_truncation"]

    assert truncation["measured"] is False
    assert truncation["calls"] == 12
    assert "rate" not in truncation
    assert "by_arm" not in truncation
    assert "tokeniser" in truncation["reason"]


def test_an_answerer_that_measures_truncation_reports_the_figures():
    counter = TokenCounter()
    combined = {
        "rag_k10": {"truncated_items": 3, "truncation_rate": 0.5},
        "cke_n12": {"truncated_items": 0, "truncation_rate": 0.0},
    }

    truncation = bench.produce_summary(combined, counter, _MeasuringAnswerer())[
        "answer_truncation"
    ]

    assert truncation["measured"] is True
    assert truncation["truncated"] == 4
    assert truncation["by_arm"]["rag_k10"]["truncated_items"] == 3
    assert truncation["by_arm"]["cke_n12"]["rate"] == 0.0


# ---------------------------------------------------------------------------
# What produced the numbers
# ---------------------------------------------------------------------------


def test_the_summary_records_the_identity_of_every_model_loaded():
    """A results file that names no commit cannot be reproduced from itself.

    The environment report printed at the start of a run cannot carry this:
    nothing has been constructed yet, so its model list is empty. The summary
    is written after the work, which is the only point at which the question
    "which weights produced these numbers" has an answer.
    """
    from cke.diagnostics import clear_runtime_state, record_loaded_model

    clear_runtime_state()
    try:
        record_loaded_model("EmbeddingModel", "some/model", "some/model@" + "a" * 40)

        summary = bench.produce_summary({}, TokenCounter(), SpanExtractiveQA())
        environment = summary["environment"]

        assert {
            "component": "EmbeddingModel",
            "requested": "some/model",
            "loaded": "some/model@" + "a" * 40,
        } in environment["loaded_models"]
        assert environment["python_version"]
        assert environment["platform"]
        # The libraries around the model move too, and their versions are as
        # much a part of a number's provenance as the weights are.
        assert any(dep["import_name"] for dep in environment["dependencies"])
    finally:
        clear_runtime_state()


class _HealthyCounter:
    """The two attributes produce_summary reads, from something that cannot
    degrade and so cannot appear in the degradation list under test."""

    description = "tiktoken cl100k_base"
    is_estimate = False

    def count(self, text: str) -> int:
        return len(text.split())


def test_the_summary_carries_what_degraded_beside_the_figures():
    """A degradation printed only to a console is lost the moment it scrolls."""
    from cke.diagnostics import clear_runtime_state, record_degradation

    clear_runtime_state()
    try:
        record_degradation("EmbeddingModel", "it hashed tokens instead")

        # A healthy counter, supplied rather than built: produce_summary reads
        # only these two attributes from it, and a real TokenCounter fetches
        # its encoding, degrades when it cannot, and adds itself to the very
        # list this test asserts the contents of.
        summary = bench.produce_summary({}, _HealthyCounter(), SpanExtractiveQA())

        assert summary["environment"]["degradations"] == [
            {"component": "EmbeddingModel", "reason": "it hashed tokens instead"}
        ]
    finally:
        clear_runtime_state()


# ---------------------------------------------------------------------------
# An unmeasured zero must not reach any output
# ---------------------------------------------------------------------------


def _rows(truncated):
    """Two items on one arm, each carrying *truncated* as its cut flag."""
    return [
        {
            "rag_k10": {
                "em": 1.0,
                "f1": 1.0,
                "prompt_tokens": 100,
                "latency_ms": 1.0,
                "answer_truncated": truncated,
                "answer_dropped_tokens": None if truncated is None else 0,
            }
        }
        for _ in range(2)
    ]


def test_unmeasured_items_produce_no_truncation_aggregate():
    """ablation.json published truncation_rate 0.0 for a backend that had
    measured nothing, contradicting the summary written from the same run."""
    metrics = bench.aggregate_metrics(_rows(None))["rag_k10"]

    assert "truncated_items" not in metrics
    assert "truncation_rate" not in metrics
    assert metrics["n"] == 2, "the arm is still aggregated, only the cut is not"


def test_measured_items_still_produce_the_truncation_aggregate():
    metrics = bench.aggregate_metrics(_rows(True))["rag_k10"]

    assert metrics["truncated_items"] == 2
    assert metrics["truncation_rate"] == 1.0


def test_the_comparison_table_says_not_measured_rather_than_zero():
    """The table printed "Items with context truncated | 0 | 0 | ..." for a
    backend that never looked."""
    counter = TokenCounter()
    per_dataset = {"hotpotqa": bench.aggregate_metrics(_rows(None))}
    combined = bench.aggregate_metrics(_rows(None))

    table = bench.produce_comparison_table(
        per_dataset, combined, counter, _UnmeasuringAnswerer()
    )

    assert "Items with context truncated" in table
    assert "not measured" in table
    truncation_rows = [
        line for line in table.splitlines() if "Items with context truncated" in line
    ]
    assert truncation_rows, "the row must not vanish in silence"
    for line in truncation_rows:
        assert "| 0 |" not in line


def test_the_comparison_table_still_counts_what_was_measured():
    counter = TokenCounter()
    per_dataset = {"hotpotqa": bench.aggregate_metrics(_rows(True))}
    combined = bench.aggregate_metrics(_rows(True))

    table = bench.produce_comparison_table(
        per_dataset, combined, counter, _MeasuringAnswerer()
    )

    assert "not measured" not in table
    assert any(
        "Items with context truncated" in line and "| 2 |" in line
        for line in table.splitlines()
    )


# ---------------------------------------------------------------------------
# What the model read, as against what was assembled for it
# ---------------------------------------------------------------------------


class _WindowedAnswerer:
    """An answerer that cuts the context, and can say what survived."""

    description = "a model with a window"
    uses_language_model = True
    truncation_measured = True
    context_measured = True

    def __init__(self, keeps: str):
        self._keeps = keeps
        self.last_context_used: str | None = None

    def answer(self, question: str, context: str) -> str:
        self.last_context_used = self._keeps
        return "an answer"


class _BlindAnswerer(_WindowedAnswerer):
    """An answerer whose provider truncates out of sight."""

    context_measured = False


def test_the_read_count_is_smaller_than_the_assembled_one_when_the_window_bites():
    """The figure the token claim belongs to.

    A context assembled at some size and cut to another is two numbers, and
    the ratio the benchmark headlines is the one the model actually read.
    """
    counter = TokenCounter()
    answerer = _WindowedAnswerer(keeps="short")

    answerer.answer("q", "a much longer context than the model will be given")
    read = bench._tokens_read(answerer, "q", counter)

    assembled = counter.count("q") + counter.count(
        "a much longer context than the model will be given"
    )
    assert read == counter.count("q") + counter.count("short")
    assert read < assembled


def test_an_answerer_that_cannot_say_reports_no_read_count():
    """None, not the assembled figure and not zero.

    An api provider cuts out of sight and the span baseline has no window;
    reporting either one's assembled size as "read" would state a measurement
    nobody took.
    """
    counter = TokenCounter()
    answerer = _BlindAnswerer(keeps="short")
    answerer.answer("q", "a long context")

    assert bench._tokens_read(answerer, "q", counter) is None
    assert bench._tokens_read(SpanExtractiveQA(), "q", counter) is None


def _read_rows(read):
    """Two items on one arm, each reporting *read* tokens as consumed."""
    return [
        {
            "rag_k10": {
                "em": 1.0,
                "f1": 1.0,
                "prompt_tokens": 1200,
                "prompt_tokens_read": read,
                "latency_ms": 1.0,
                "answer_truncated": None,
                "answer_dropped_tokens": None,
            }
        }
        for _ in range(2)
    ]


def test_an_arm_that_reported_no_read_count_publishes_no_figure():
    """Not a zero, and not the assembled median wearing the read one's name."""
    metrics = bench.aggregate_metrics(_read_rows(None))["rag_k10"]

    assert "median_tokens_read" not in metrics
    assert "tokens_read_measured_items" not in metrics
    assert metrics["median_tokens"] == 1200, "the assembled figure still stands"


def test_a_measured_arm_publishes_the_read_median_beside_the_assembled_one():
    metrics = bench.aggregate_metrics(_read_rows(500))["rag_k10"]

    assert metrics["median_tokens"] == 1200
    assert metrics["median_tokens_read"] == 500
    assert metrics["tokens_read_measured_items"] == 2


def test_the_read_figure_carries_an_interval_like_every_other_headline():
    """A point estimate over two items and one over two thousand print alike."""
    intervals = bench.bootstrap_intervals(_read_rows(500), replicates=64, seed=1)

    assert "median_tokens_read" in intervals["rag_k10"]
    assert "median_tokens" in intervals["rag_k10"]


def test_the_table_names_which_token_figure_each_row_is():
    """Neither can be quoted as the other."""
    metrics = bench.with_intervals(
        bench.aggregate_metrics(_read_rows(500)), _read_rows(500)
    )
    table = bench.produce_comparison_table(
        {"hotpotqa": metrics}, metrics, TokenCounter(), _WindowedAnswerer("x")
    )

    assert "Median prompt tokens (assembled)" in table
    assert "Median prompt tokens (read)" in table


# ---------------------------------------------------------------------------
# Declining is a judgement only an answerer that can decline gets to make
# ---------------------------------------------------------------------------


class _DecliningAnswerer:
    """An answerer whose vocabulary holds a declared abstention."""

    description = "a model that can decline"
    uses_language_model = True
    can_abstain = True


def test_an_answerer_that_cannot_decline_records_no_judgement():
    """Not False. False is a judgement, and this one could not have made it.

    Neither shipped answerer can emit a value in ABSTAIN_ANSWERS: the span
    baseline returns "" when it has nothing and the language model returns
    whatever it generated.
    """
    assert bench._abstention_of(SpanExtractiveQA(), "") is None
    assert bench._abstention_of(_MeasuringAnswerer(), "INSUFFICIENT_EVIDENCE") is None


def test_an_answerer_that_can_decline_has_its_answer_read():
    answerer = _DecliningAnswerer()

    assert bench._abstention_of(answerer, "INSUFFICIENT_EVIDENCE") is True
    assert bench._abstention_of(answerer, "Istanbul") is False


def _abstention_rows(abstained, em=1.0):
    return [
        {
            "rag_k10": {
                "em": em,
                "f1": 1.0,
                "prompt_tokens": 100,
                "latency_ms": 1.0,
                "abstained": abstained,
                "answer_truncated": None,
                "answer_dropped_tokens": None,
            }
        }
        for _ in range(2)
    ]


def test_no_abstention_rate_is_published_when_nothing_could_abstain():
    """ablation.json carried false_abstention_rate 0.0 for every arm on every
    dataset, with nothing beside it saying the figure was never taken."""
    metrics = bench.aggregate_metrics(_abstention_rows(None))["rag_k10"]

    assert "false_abstention_rate" not in metrics
    assert "abstention_rate" not in metrics
    assert metrics["n"] == 2, "the arm is still aggregated, only the rate is not"


def test_a_false_abstention_is_published_when_declining_was_possible():
    metrics = bench.aggregate_metrics(_abstention_rows(True))["rag_k10"]

    assert metrics["false_abstention_rate"] == 1.0
    assert "abstention_rate" not in metrics, "no item here was unanswerable"


def test_the_summary_says_the_abstention_figure_was_never_available():
    """A missing block reads as nothing to say; this says why."""
    block = bench._abstention_summary(SpanExtractiveQA(), {"rag_k10": {"em": 0.3}})

    assert block["measured"] is False
    assert block["answerer_can_abstain"] is False
    assert "zero whatever the arm did" in block["reason"]


def test_the_summary_file_carries_the_abstention_block():
    """Reaching summary.json is the point; a helper nobody calls says nothing.

    Written after a mutation deleting this line from produce_summary survived:
    the tests exercised the helper and never checked that its answer was
    published.
    """
    summary = bench.produce_summary(
        {"rag_k10": {"median_tokens": 100.0}, "cke_n12": {"median_tokens": 10.0}},
        # Not a real TokenCounter: this is about the abstention block reaching
        # summary.json, and a strict counter would make the test depend on a
        # tiktoken download it has no use for.
        _HealthyCounter(),
        SpanExtractiveQA(),
    )

    assert summary["abstention"]["measured"] is False
    assert "zero whatever the arm did" in summary["abstention"]["reason"]


def test_the_summary_reports_the_rates_when_an_arm_measured_them():
    combined = {"rag_k10": {"abstention_rate": 0.5, "unanswerable_items": 4}}

    block = bench._abstention_summary(_DecliningAnswerer(), combined)

    assert block["measured"] is True
    assert block["by_arm"]["rag_k10"]["abstention_rate"] == 0.5
