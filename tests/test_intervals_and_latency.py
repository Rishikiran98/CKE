"""What a figure is worth, and what its latency covers.

A point estimate over fifteen items and one over fifteen thousand print
identically, and every table here has shown the first while reading like the
second.
"""

from __future__ import annotations

import importlib.util
import sys
import time
from pathlib import Path

import pytest

from cke.evaluation.span_qa import SpanExtractiveQA
from cke.evaluation.token_counter import TokenCounter

ROOT = Path(__file__).resolve().parents[1]


def _load_module():
    path = ROOT / "scripts" / "run_cke_benchmark.py"
    spec = importlib.util.spec_from_file_location("run_cke_benchmark_ci", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


bench = _load_module()


def _item(em, tokens=100, completion=2, latency=5.0, recall=1.0):
    return {
        "em": em,
        "f1": em,
        "prompt_tokens": tokens,
        "completion_tokens": completion,
        "latency_ms": latency,
        "answer_truncated": False,
        "retrieval_recall": recall,
    }


def _rows(ems, arms=("rag_k10",), **kwargs):
    return [{arm: _item(em, **kwargs) for arm in arms} for em in ems]


# ---------------------------------------------------------------------------
# The interval itself
# ---------------------------------------------------------------------------


def test_an_interval_brackets_the_figure_it_qualifies():
    rows = _rows([1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0])
    point = bench.aggregate_metrics(rows)["rag_k10"]
    bounds = bench.bootstrap_intervals(rows)["rag_k10"]["em"]

    assert bounds["low"] <= point["em"] <= bounds["high"]
    assert bounds["low"] < bounds["high"], "eight coin flips do not pin a mean"


def test_more_of_the_same_evidence_narrows_the_interval():
    """The whole reason to print one: it must respond to how much was seen."""
    few = bench.bootstrap_intervals(_rows([1.0, 0.0] * 5))["rag_k10"]["em"]
    many = bench.bootstrap_intervals(_rows([1.0, 0.0] * 200))["rag_k10"]["em"]

    assert (many["high"] - many["low"]) < (few["high"] - few["low"])


#: Latencies spread finely enough that a percentile moves when the draw does.
#: A coarse sample — five items of 1.0 and 0.0 — lands on the same bounds
#: whatever it draws, and a test built on one cannot see an ignored seed.
_SPREAD = [{"rag_k10": _item(1.0, latency=float(i))} for i in range(40)]


def test_the_same_seed_gives_the_same_interval():
    """Two runs of the same command must produce the same interval."""
    first = bench.bootstrap_intervals(_SPREAD, replicates=200)
    second = bench.bootstrap_intervals(_SPREAD, replicates=200)

    assert first == second
    assert first["rag_k10"]["median_latency_ms"]["low"] < (
        first["rag_k10"]["median_latency_ms"]["high"]
    ), "the fixture must be able to show a difference at all"


def test_a_different_seed_gives_a_different_draw():
    """Otherwise the seed is decoration and the interval is not resampled."""
    first = bench.bootstrap_intervals(_SPREAD, replicates=200, seed=1)
    second = bench.bootstrap_intervals(_SPREAD, replicates=200, seed=2)

    assert first != second


def test_one_resample_is_shared_by_every_arm():
    """The arms answered the same items. Resampling them independently would
    break the only thing that makes their columns comparable."""
    rows = _rows([1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0], arms=("rag_k5", "rag_k10"))

    intervals = bench.bootstrap_intervals(rows, replicates=300)

    assert intervals["rag_k5"] == intervals["rag_k10"], (
        "identical per-item values under one shared resample must produce "
        "identical intervals"
    )


def test_a_single_item_gets_no_interval():
    """A resample of one item is that item, and an interval of zero width
    over one observation reads as certainty."""
    assert bench.bootstrap_intervals(_rows([1.0])) == {}


def test_a_figure_only_one_item_reported_gets_no_interval():
    """Per figure, not per run: an arm can have plenty of items and one
    observation of a particular figure."""
    rows = _rows([1.0, 0.0, 1.0, 0.0])
    for row in rows[1:]:
        row["rag_k10"]["retrieval_recall"] = None

    bounds = bench.bootstrap_intervals(rows, replicates=200)["rag_k10"]

    assert "retrieval_recall" not in bounds
    assert "em" in bounds, "the figures every item reported still get one"


def test_a_figure_only_some_items_reported_is_estimated_on_those():
    rows = _rows([1.0] * 10)
    for row in rows[2:]:
        row["rag_k10"]["retrieval_recall"] = None

    bounds = bench.bootstrap_intervals(rows, replicates=500)["rag_k10"]

    assert bounds["retrieval_recall"]["low"] == 1.0
    assert (
        bounds["retrieval_recall"]["replicates_used"] < 500
    ), "some resamples draw none of the two items that reported it"
    assert bounds["em"]["replicates_used"] == 500


def test_asking_for_no_replicates_is_refused():
    with pytest.raises(ValueError, match="at least 1"):
        bench.bootstrap_intervals(_rows([1.0, 0.0]), replicates=0)


# ---------------------------------------------------------------------------
# Where the interval has to appear
# ---------------------------------------------------------------------------


def test_the_interval_travels_with_the_figure():
    rows = _rows([1.0, 0.0, 1.0, 0.0])
    metrics = bench.with_intervals(bench.aggregate_metrics(rows), rows)

    assert "em" in metrics["rag_k10"]["intervals"]
    assert "median_tokens" in metrics["rag_k10"]["intervals"]


def test_the_table_prints_the_intervals_and_names_the_method():
    rows = _rows([1.0, 0.0, 1.0, 0.0])
    metrics = bench.with_intervals(bench.aggregate_metrics(rows), rows)

    table = bench.produce_comparison_table(
        {"hotpotqa": metrics}, metrics, TokenCounter(), SpanExtractiveQA()
    )

    assert "bootstrap intervals" in table
    assert str(bench.BOOTSTRAP_SEED) in table
    assert str(bench.BOOTSTRAP_REPLICATES) in table
    bounds = metrics["rag_k10"]["intervals"]["em"]
    assert f"{bounds['low']:.4g}–{bounds['high']:.4g}" in table


def test_the_summary_carries_the_intervals_and_the_method():
    rows = _rows([1.0, 0.0, 1.0, 0.0])
    metrics = bench.with_intervals(bench.aggregate_metrics(rows), rows)

    summary = bench.produce_summary(metrics, TokenCounter(), SpanExtractiveQA())

    assert summary["intervals"]["rag_k10"]["em"]["low"] is not None
    assert "bootstrap" in summary["interval_method"]
    assert str(bench.BOOTSTRAP_SEED) in summary["interval_method"]


# ---------------------------------------------------------------------------
# Latency, and what it covers
# ---------------------------------------------------------------------------


def test_the_summary_reports_latency_beside_the_token_counts():
    """It was in the tables and missing from the file they summarise."""
    rows = _rows([1.0, 0.0], latency=12.5)
    metrics = bench.with_intervals(bench.aggregate_metrics(rows), rows)

    summary = bench.produce_summary(metrics, TokenCounter(), SpanExtractiveQA())

    assert summary["rag_k10_median_latency_ms"] == 12.5
    assert "rag_k10_median_tokens" in summary


def test_the_outputs_say_what_a_latency_figure_covers():
    """Read as a query latency it is wrong: each arm builds its index or
    graph over the item's documents inside the timed region."""
    rows = _rows([1.0, 0.0])
    metrics = bench.with_intervals(bench.aggregate_metrics(rows), rows)

    summary = bench.produce_summary(metrics, TokenCounter(), SpanExtractiveQA())
    table = bench.produce_comparison_table(
        {"hotpotqa": metrics}, metrics, TokenCounter(), SpanExtractiveQA()
    )

    for text in (summary["latency_includes"], table):
        assert "builds its index once" in text


class _CountingCounter:
    """A token counter that needs no encoding and so no download."""

    description = "stub counter"
    is_estimate = False

    def count(self, text: str) -> int:
        return len(str(text).split())


def test_the_timed_region_covers_building_the_arms_structure(monkeypatch):
    """The claim above must match the run: the index build is inside the clock.

    This used to slice the driver's source between two class names, find
    "t0 = time.perf_counter()", and assert "build_index" appeared after it.
    That passes on a comment and on a dead branch, fails on a rename that
    changes nothing, and says nothing about the number reported. So make the
    index build take a measurable amount of time and read the latency.
    """
    slow_build_ms = 60

    class SlowIndexRetriever:
        def __init__(self, *args, **kwargs):
            pass

        def build_index(self, docs):
            time.sleep(slow_build_ms / 1000.0)

        def retrieve(self, question, k=5):
            return [{"doc_id": "d1", "text": "Hagia Sophia is in Istanbul."}]

    monkeypatch.setattr(bench, "RAGRetriever", SlowIndexRetriever)
    pipeline = bench.RAGPipeline(
        token_counter=_CountingCounter(),
        answerer=SpanExtractiveQA(),
        strict=False,
    )

    result = pipeline.run_item(
        "Where is Hagia Sophia?",
        [{"doc_id": "d1", "text": "Hagia Sophia is in Istanbul."}],
        k=1,
    )

    assert result["latency_ms"] >= slow_build_ms, (
        "the reported latency does not include the index build, so it is a "
        "query latency and the reports describe it as more than that"
    )


# ---------------------------------------------------------------------------
# Every report, not just the one the change was written against
# ---------------------------------------------------------------------------


def test_the_ablation_table_carries_the_intervals_too():
    """It is read on its own. A figure quoted from it without its interval is
    quoted without the thing that says what it is worth."""
    rows = _rows([1.0, 0.0, 1.0, 0.0])
    metrics = bench.with_intervals(bench.aggregate_metrics(rows), rows)

    table = bench.produce_ablation_table({"hotpotqa": metrics}, metrics)

    bounds = metrics["rag_k10"]["intervals"]["em"]
    assert f"({bounds['low']:.4g}–{bounds['high']:.4g})" in table
    assert str(bench.BOOTSTRAP_SEED) in table


def test_the_ablation_table_says_what_its_latency_covers():
    rows = _rows([1.0, 0.0])
    metrics = bench.with_intervals(bench.aggregate_metrics(rows), rows)

    table = bench.produce_ablation_table({"hotpotqa": metrics}, metrics)

    assert "builds its index once" in table


def test_a_figure_an_arm_never_produced_is_not_printed_as_zero():
    """The ablation renderer defaulted every missing figure to 0, so an arm
    that produced nothing printed 0.0000 beside arms that measured."""
    rows = _rows([1.0, 0.0])
    metrics = bench.with_intervals(bench.aggregate_metrics(rows), rows)

    table = bench.produce_ablation_table({"hotpotqa": metrics}, metrics)

    cke_rows = [line for line in table.splitlines() if line.startswith("| CKE N=8")]
    assert cke_rows, "the row must still be listed"
    for line in cke_rows:
        assert "0.0000" not in line
        assert "not measured" in line


def test_the_figure_cell_states_an_absence_rather_than_a_number():
    assert bench.figure_cell({}, "em") == "not measured"
    assert bench.figure_cell({"em": 0.0}, "em") == "0.0000"
    assert (
        bench.figure_cell(
            {"em": 0.5, "intervals": {"em": {"low": 0.25, "high": 0.75}}}, "em"
        )
        == "0.5000 (0.25–0.75)"
    )


# ---------------------------------------------------------------------------
# What determines a run
# ---------------------------------------------------------------------------


def test_a_capped_run_samples_rather_than_taking_a_prefix():
    """A prefix carries whatever ordering the file has. MuSiQue's dev split
    is ordered by hop count, so every capped run of it was a two-hop run."""
    chosen = bench.select_indices(1000, 10)

    assert chosen != list(range(10))
    assert len(set(chosen)) == 10, "sampled without replacement"
    assert chosen == sorted(chosen), "evaluated in file order whatever was drawn"
    assert all(0 <= index < 1000 for index in chosen)


def test_the_same_seed_selects_the_same_items():
    assert bench.select_indices(1000, 25) == bench.select_indices(1000, 25)


def test_a_different_seed_selects_different_items():
    first = bench.select_indices(1000, 25, seed=1)
    second = bench.select_indices(1000, 25, seed=2)

    assert first != second


def test_a_prefix_is_still_reachable_and_named():
    """Kept so the old behaviour can be reproduced deliberately, and it is
    recorded in the provenance when it is."""
    assert bench.select_indices(1000, 5, method="prefix") == [0, 1, 2, 3, 4]


def test_a_cap_at_or_above_the_file_takes_everything():
    assert bench.select_indices(5, 5) == [0, 1, 2, 3, 4]
    assert bench.select_indices(5, 50) == [0, 1, 2, 3, 4]


def test_an_unknown_selection_method_is_refused():
    with pytest.raises(ValueError, match="unknown selection method"):
        bench.select_indices(10, 2, method="whatever")


def test_a_cap_below_one_is_refused():
    with pytest.raises(ValueError, match="at least 1"):
        bench.select_indices(10, 0)


def test_only_the_evaluated_records_are_normalised(tmp_path):
    """A malformed record this run never evaluates declared a degradation and
    refused a strict run. The driver loaded the whole file and sliced."""
    import json

    from cke.datasets.hotpot_loader import HotpotDataset

    good = {"_id": "good", "question": "Q?", "answer": "A", "context": [["T", ["s"]]]}
    broken = {"_id": "bad", "question": "Q?", "answer": "A", "context": [["only"]]}
    path = tmp_path / "hotpotqa_dev.json"
    path.write_text(json.dumps([good, broken]), encoding="utf-8")

    items, provenance = bench.load_selected(
        HotpotDataset(strict=True), path, 1, bench.SAMPLE_SEED, "prefix"
    )

    assert [item["id"] for item in items] == ["good"]
    assert provenance["records_in_file"] == 2
    assert provenance["items_evaluated"] == 1


def test_the_provenance_names_the_file_by_its_contents(tmp_path):
    """A path says nothing: it can hold different bytes tomorrow."""
    import json

    from cke.datasets.hotpot_loader import HotpotDataset

    record = {"_id": "a", "question": "Q?", "answer": "A", "context": [["T", ["s"]]]}
    path = tmp_path / "hotpotqa_dev.json"
    path.write_text(json.dumps([record]), encoding="utf-8")
    _, first = bench.load_selected(HotpotDataset(), path, 1, 1, "prefix")

    path.write_text(json.dumps([record, record]), encoding="utf-8")
    _, second = bench.load_selected(HotpotDataset(), path, 1, 1, "prefix")

    assert first["sha256"] != second["sha256"]
    assert len(first["sha256"]) == 64
    assert first["item_ids"] == ["a"]


def test_the_deterministic_view_drops_only_what_cannot_repeat():
    payload = {
        "rag_k10_em": 0.5,
        "rag_k10_median_latency_ms": 12.0,
        "provenance": {"started_at": "now", "seeds": {"item_sample": 1}},
        "rows": [{"latency_ms": 3.0, "em": 1.0}],
    }

    view = bench.deterministic_view(payload)

    assert view == {
        "rag_k10_em": 0.5,
        "provenance": {"seeds": {"item_sample": 1}},
        "rows": [{"em": 1.0}],
    }


def test_every_dropped_field_says_why_it_cannot_repeat():
    assert bench.NON_REPRODUCIBLE_FIELDS
    for field, reason in bench.NON_REPRODUCIBLE_FIELDS.items():
        assert reason and not reason.endswith("."), field


def test_two_runs_that_agree_compare_clean(tmp_path):
    import json

    payload = {
        "rag_k10_em": 0.5,
        "provenance": {"seeds": {"item_sample": 1}},
    }
    first = tmp_path / "a.json"
    second = tmp_path / "b.json"
    first.write_text(json.dumps({**payload, "started_at": "one"}), encoding="utf-8")
    second.write_text(json.dumps({**payload, "started_at": "two"}), encoding="utf-8")

    assert (
        bench.compare_runs(first, second) == []
    ), "a timestamp is not a disagreement about a result"


def test_a_figure_that_moved_between_runs_is_reported(tmp_path):
    import json

    first = tmp_path / "a.json"
    second = tmp_path / "b.json"
    first.write_text(
        json.dumps({"rag_k10_em": 0.5, "rag_k10_median_latency_ms": 1.0}),
        encoding="utf-8",
    )
    second.write_text(
        json.dumps({"rag_k10_em": 0.6, "rag_k10_median_latency_ms": 99.0}),
        encoding="utf-8",
    )

    differences = bench.compare_runs(first, second)

    assert differences == [
        "rag_k10_em: 0.5 then 0.6"
    ], "the latency moved too, and that is expected rather than a defect"


def test_a_key_present_in_only_one_run_is_reported(tmp_path):
    import json

    first = tmp_path / "a.json"
    second = tmp_path / "b.json"
    first.write_text(json.dumps({"a": 1, "b": 2}), encoding="utf-8")
    second.write_text(json.dumps({"a": 1}), encoding="utf-8")

    assert bench.compare_runs(first, second) == ["b: only in the first run"]


def test_the_git_state_is_read_before_the_run_writes_anything(
    tmp_path, monkeypatch, capsys
):
    """With --output-dir inside the repository, the run's own files were in
    the tree by the time the state was read, so a run that started clean
    recorded itself dirty because of the artifacts it had just produced.

    This used to compare the character offsets of two lines in the driver.
    Instead, watch the call: record whether the output directory existed at
    the moment the state was read. The run is expected to fail afterwards —
    it has no datasets — and that is fine, because everything under test has
    already happened by then.
    """
    output_dir = tmp_path / "results-that-do-not-exist-yet"
    observed = {}

    def _spy():
        observed["output_dir_existed"] = output_dir.exists()
        return "0000000 (clean)"

    monkeypatch.setattr(bench, "_git_description", _spy)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run_cke_benchmark.py",
            # --allow-degraded so the run gets as far as the line under test
            # on a machine with no tokenizer cached: a strict TokenCounter
            # refuses before main() reads the git state, and this test is
            # about the ordering, not about strictness.
            "--allow-degraded",
            "--skip-download",
            "--output-dir",
            str(output_dir),
            # Dataset paths that do not exist, so the run stops at the same
            # place on every machine. Left to the defaults it reads data/,
            # which is populated on a development container and empty on a
            # fresh checkout — so how far main() got, and therefore the
            # measured coverage, depended on the machine. The floor is a gate
            # on that number; it cannot be allowed to move with the checkout.
            "--hotpot-path",
            str(tmp_path / "absent-hotpot.json"),
            "--wiki2-path",
            str(tmp_path / "absent-wiki2.json"),
            "--musique-path",
            str(tmp_path / "absent-musique.json"),
            "--limit",
            "1",
        ],
    )

    stopped_with = None
    try:
        bench.main()
    except BaseException as exc:  # noqa: BLE001 - the run is meant to stop
        stopped_with = exc

    # The run must stop because a dataset file is missing, not because it
    # found the repository's data/ and got further. Asserting where it
    # stopped is what keeps the executed path — and so the coverage figure
    # the floor gates — the same on a populated container and a fresh clone.
    assert stopped_with is not None, "the run was expected to stop and did not"
    printed = capsys.readouterr().out
    assert "absent-hotpot.json" in printed, (
        "the run did not look for the dataset this test named, so it found "
        "whatever the machine had and got further than it does elsewhere"
    )
    assert observed, "the run never read the git state at all"
    assert observed["output_dir_existed"] is False, (
        "the state was read after the output directory existed, so a run "
        "writing into the repository records itself dirty for its own files"
    )


def test_the_provenance_says_whether_the_file_was_a_whole_split(tmp_path):
    """A sample of a capped prefix is still a sample of a prefix."""
    import json

    from cke.datasets.hotpot_loader import HotpotDataset

    record = {"_id": "a", "question": "Q?", "answer": "A", "context": [["T", ["s"]]]}
    path = tmp_path / "hotpotqa_dev.json"
    path.write_text(json.dumps([record]), encoding="utf-8")

    _, without = bench.load_selected(HotpotDataset(), path, 1, 1, "prefix")
    assert without["complete_split"] is None, "no note is not a good note"

    path.with_name(path.name + ".source.json").write_text(
        json.dumps({"complete_split": True, "records": 1}), encoding="utf-8"
    )
    _, with_note = bench.load_selected(HotpotDataset(), path, 1, 1, "prefix")
    assert with_note["complete_split"] is True


def test_a_note_that_outlived_its_file_does_not_vouch_for_it(tmp_path):
    """A note survives the file being truncated or replaced, and under
    --skip-download the note is the only check there is."""
    import json

    from cke.datasets.hotpot_loader import HotpotDataset

    record = {"_id": "a", "question": "Q?", "answer": "A", "context": [["T", ["s"]]]}
    path = tmp_path / "hotpotqa_dev.json"
    path.write_text(json.dumps([record, record]), encoding="utf-8")
    path.with_name(path.name + ".source.json").write_text(
        json.dumps({"complete_split": True, "records": 2417}), encoding="utf-8"
    )

    _, provenance = bench.load_selected(HotpotDataset(), path, 1, 1, "prefix")

    assert (
        provenance["complete_split"] is False
    ), "a note claiming 2417 records beside a file of 2 vouches for nothing"
    assert provenance["records_in_file"] == 2
