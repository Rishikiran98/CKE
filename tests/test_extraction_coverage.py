"""The coverage measurement reports what it counted, and refuses what it cannot.

This script exists to answer one question before the benchmark is run for real:
does ``RuleExtractor`` read enough of published prose for the CKE arm's triple
count to be a measurement rather than an artefact of five regexes. A figure
that decides that must itself be checkable, so these cover the parts that could
quietly report the wrong thing — the frame attribution, the distinction between
"no observations" and "a measured zero", and the refusal to run on a dataset
that is not there.

The records here are minimal fixtures for the plumbing. They test wiring, not
extraction quality: nothing in this file asserts a coverage figure, because a
coverage figure over text written in this file would be a fact about this file.
"""

from __future__ import annotations

import importlib.util
import json
import pathlib

import pytest

from cke.extractor.rule_extractor import RuleExtractor

ROOT = pathlib.Path(__file__).resolve().parents[1]


def _load_module():
    path = ROOT / "scripts" / "measure_extraction_coverage.py"
    spec = importlib.util.spec_from_file_location("measure_extraction_coverage", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


coverage = _load_module()


# --- frame attribution -------------------------------------------------------


def test_every_frame_the_extractor_has_can_be_attributed():
    """Derived from PATTERNS, so a frame added upstream cannot go uncounted.

    A statement attributed to "unattributed" is one this script cannot report
    against any frame, which would make the breakdown silently incomplete.
    """
    for _, template in RuleExtractor.PATTERNS:
        # A relation the frame would actually produce: placeholders stand for
        # whatever it captured, so any word does.
        relation = template.format(rel="directed", prep="in")

        assert coverage.frame_of(relation) == template, (
            f"the frame {template!r} produces {relation!r}, which this script "
            f"attributes to {coverage.frame_of(relation)!r}"
        )


def test_the_passive_frame_wins_over_the_prepositional_one():
    """Both templates fit "directed_by"; PATTERNS order decides, as in the
    extractor, which tries them in that order and takes the first match."""
    assert coverage.frame_of("directed_by") == "{rel}_by"
    assert coverage.frame_of("released_in") == "{rel}_{prep}"


def test_a_relation_no_frame_produces_is_named_as_such():
    assert coverage.frame_of("borders") == "unattributed"


# --- reading the published shapes -------------------------------------------


def test_supporting_facts_are_read_as_title_and_sentence_index():
    record = {"supporting_facts": [["Ed Wood", 0], ["Scott Derrickson", 2]]}

    assert coverage._supporting_sentences(record) == {
        ("Ed Wood", 0),
        ("Scott Derrickson", 2),
    }


@pytest.mark.parametrize(
    "fact",
    [["only-one-element"], ["title", "not-an-index"], "not-a-pair", []],
    ids=["short", "unparseable-index", "scalar", "empty"],
)
def test_a_malformed_supporting_fact_is_dropped_not_guessed_at(fact):
    assert coverage._supporting_sentences({"supporting_facts": [fact]}) == set()


def test_a_context_entry_carrying_one_string_is_read_as_one_sentence():
    """Both published shapes appear: a list of sentences, or a single body."""
    record = {"context": [["A", ["one.", "two."]], ["B", "just one body"]]}

    assert coverage._sentences_by_title(record) == {
        "A": ["one.", "two."],
        "B": ["just one body"],
    }


# --- the two figures that must not look alike -------------------------------


def test_a_share_over_nothing_is_not_reported_as_zero():
    """The distinction this whole programme keeps reinstating: nothing
    observed and nothing found are different answers."""
    assert coverage._yield_block(0, 0)["share"] is None
    assert coverage._yield_block(0, 10)["share"] == 0.0


def test_the_distribution_says_how_often_twelve_were_available():
    """The 'CKE N=12' arm retrieves twelve; this is how often it had twelve."""
    block = coverage._distribution([0, 5, 12, 20])

    assert block["items"] == 4
    assert block["median"] == 8.5
    assert block["zero"] == 1
    assert block["at_least_12"] == 2
    assert block["share_at_least_12"] == 0.5


def test_a_distribution_over_no_items_reports_none_rather_than_zeroes():
    assert coverage._distribution([]) == {"items": 0}


# --- end to end, on the plumbing --------------------------------------------


def _record(identifier: str) -> dict:
    """One record in the published shape. Wiring only — see the module docstring."""
    return {
        "_id": identifier,
        "question": "A question?",
        "answer": "An answer",
        "context": [
            ["Titled Doc", ["Titled Doc is a museum.", "It has no frame here"]],
        ],
        "supporting_facts": [["Titled Doc", 0]],
    }


def _write(tmp_path, name, records) -> pathlib.Path:
    path = tmp_path / name
    path.write_text(json.dumps(records), encoding="utf-8")
    return path


def test_the_measurement_counts_sentences_and_supporting_sentences_apart(tmp_path):
    hotpot = _write(tmp_path, "h.json", [_record("a"), _record("b")])
    block = coverage.measure("hotpotqa", hotpot, None, 42, "prefix", coverage._driver())

    assert block["provenance"]["records_in_file"] == 2
    assert block["provenance"]["items_evaluated"] == 2
    # Two sentences per record, one of them named as supporting.
    assert block["sentences"]["of"] == 4
    assert block["supporting_sentences"]["of"] == 2
    assert len(block["provenance"]["sha256"]) == 64


def test_the_same_input_measures_the_same_twice(tmp_path):
    """A coverage figure that moves between runs is not a measurement."""
    hotpot = _write(tmp_path, "h.json", [_record("a"), _record("b")])
    driver = coverage._driver()

    first = coverage.measure("hotpotqa", hotpot, None, 42, "prefix", driver)
    second = coverage.measure("hotpotqa", hotpot, None, 42, "prefix", driver)

    assert first == second


def test_a_missing_dataset_is_named_and_not_substituted(tmp_path, capsys):
    """R1. A coverage figure over whichever dataset happened to be present is
    not the figure this reports, and generated text is never a stand-in."""
    present = _write(tmp_path, "h.json", [_record("a")])

    code = coverage.main(
        [
            "--hotpot-path",
            str(present),
            "--wiki2-path",
            str(tmp_path / "absent.json"),
        ]
    )
    err = capsys.readouterr().err

    assert code == 2
    assert "wiki2" in err and "absent.json" in err
    assert "download_datasets" in err, "it must say how to get the dataset"


def test_a_run_over_both_datasets_writes_what_it_counted(tmp_path):
    hotpot = _write(tmp_path, "h.json", [_record("a")])
    wiki2 = _write(tmp_path, "w.json", [_record("b")])
    out = tmp_path / "coverage.json"

    code = coverage.main(
        [
            "--hotpot-path",
            str(hotpot),
            "--wiki2-path",
            str(wiki2),
            "--output",
            str(out),
        ]
    )

    assert code == 0
    written = json.loads(out.read_text(encoding="utf-8"))
    assert set(written["datasets"]) == set(coverage.LOADERS)
    assert "started_at" in written


# --- strict, like every other entry point that produces a figure ----------


def test_the_loader_is_strict_unless_the_run_asks_otherwise(tmp_path, monkeypatch):
    """A loader that drops malformed records changes the denominator.

    Every share this script reports is over the records that normalised. A
    coverage figure taken while a loader was quietly discarding entries
    describes a corpus the report cannot name. This script shipped with
    strict=False and no way to ask for anything else, which made it a third
    exception to a rule the README states as universal.
    """
    seen = {}

    class _Recording(coverage.LOADERS["hotpotqa"]):
        def __init__(self, strict=False, **kwargs):
            seen["strict"] = strict
            super().__init__(strict=strict, **kwargs)

    monkeypatch.setitem(coverage.LOADERS, "hotpotqa", _Recording)
    hotpot = _write(tmp_path, "h.json", [_record("a")])

    coverage.measure("hotpotqa", hotpot, None, 42, "prefix", coverage._driver())
    assert seen["strict"] is True, "the default must be strict"

    coverage.measure(
        "hotpotqa", hotpot, None, 42, "prefix", coverage._driver(), strict=False
    )
    assert seen["strict"] is False, "and it must still be askable"


def test_a_run_says_what_it_ran_on_and_whether_anything_degraded(tmp_path, capsys):
    """The environment report and the degradation summary, as the driver does.

    Without them a saved figure cannot say whether a component was degraded
    when it was taken.
    """
    hotpot = _write(tmp_path, "h.json", [_record("a")])
    wiki2 = _write(tmp_path, "w.json", [_record("b")])
    out = tmp_path / "coverage.json"

    coverage.main(
        [
            "--hotpot-path",
            str(hotpot),
            "--wiki2-path",
            str(wiki2),
            "--output",
            str(out),
        ]
    )
    printed = capsys.readouterr().out

    assert "CKE environment report" in printed
    assert "degraded" in printed
    written = json.loads(out.read_text(encoding="utf-8"))
    assert "environment" in written, "the saved report must carry it too"


def test_the_rendered_report_names_what_each_figure_decides(tmp_path, capsys):
    """A number nobody can interpret is the kind this programme keeps removing."""
    hotpot = _write(tmp_path, "h.json", [_record("a")])
    wiki2 = _write(tmp_path, "w.json", [_record("b")])

    coverage.main(["--hotpot-path", str(hotpot), "--wiki2-path", str(wiki2)])
    out = capsys.readouterr().out

    assert "SUPPORTING sentences producing one" in out
    assert "N=12" in out, "the report must connect the figure to the arm it judges"
