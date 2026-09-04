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
    """
    source = (ROOT / "scripts" / "run_cke_benchmark.py").read_text(encoding="utf-8")

    assert "WORDS_TO_TOKENS" not in source
    assert "_token_count_static" not in source
    assert "len(text.split()) * 1.3" not in source


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


def test_a_scored_row_defaults_truncation_when_the_answerer_has_none():
    raw = {"answer": "x", "prompt_tokens": 5, "latency_ms": 1.0}
    row = bench._score_row(raw, "x")

    assert row["answer_truncated"] is False
    assert row["answer_dropped_tokens"] == 0


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
        getattr(bench, loader)(path, 10, True)

    # The opt-out still loads, with the item thinned and the drop declared.
    items = getattr(bench, loader)(path, 10, False)
    assert items[0]["documents"] == []


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
