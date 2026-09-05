"""Gate 4 is a command now, and both ways of spelling it are one function.

The programme calls this Gate 4: run the benchmark twice and diff. The
comparison existed and was tested, but nothing could invoke it — re-running the
gate meant importing a two-thousand-line driver from a Python prompt. So its
evidence was prose in a pull request description, written by the one person who
could produce it, on a container nobody else can reproduce.

These cover the two surfaces (``cke-compare-runs`` and the driver's
``--compare-runs``), the exit codes a caller scripting the gate needs to tell
apart, and the ``--twice`` orchestration. They do **not** run a benchmark: the
runs ``--twice`` performs are stubbed, so what is checked here is that two
distinct directories are used and both are compared. Running the instrument for
real is PR O's, and an elaborate stub pretending otherwise is how three of this
session's failures happened.
"""

from __future__ import annotations

import importlib.util
import json
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

from cke.evaluation import run_comparison

ROOT = Path(__file__).resolve().parents[1]


def _load_module():
    path = ROOT / "scripts" / "run_cke_benchmark.py"
    spec = importlib.util.spec_from_file_location("run_cke_benchmark_compare", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


bench = _load_module()


def _summaries(tmp_path, first_payload, second_payload):
    first = tmp_path / "a.json"
    second = tmp_path / "b.json"
    first.write_text(json.dumps(first_payload), encoding="utf-8")
    second.write_text(json.dumps(second_payload), encoding="utf-8")
    return first, second


AGREEING = (
    {"rag_k10_em": 0.5, "started_at": "one", "median_latency_ms": 1.0},
    {"rag_k10_em": 0.5, "started_at": "two", "median_latency_ms": 99.0},
)

DISAGREEING = (
    {"rag_k10_em": 0.5, "started_at": "one"},
    {"rag_k10_em": 0.6, "started_at": "two"},
)


def test_runs_that_agree_exit_zero_and_say_so(tmp_path, capsys):
    first, second = _summaries(tmp_path, *AGREEING)

    assert run_comparison.report_comparison(first, second) == 0
    assert "agree" in capsys.readouterr().out


def test_a_moved_figure_exits_non_zero_and_names_the_field(tmp_path, capsys):
    """Naming it is the point.

    "The runs differ" without saying where is the kind of check this
    programme has spent its time removing.
    """
    first, second = _summaries(tmp_path, *DISAGREEING)

    code = run_comparison.report_comparison(first, second)
    out = capsys.readouterr().out

    assert code == run_comparison.DIFFERENCES_FOUND
    assert "rag_k10_em: 0.5 then 0.6" in out


def test_the_excused_fields_are_printed_with_their_reasons(tmp_path, capsys):
    """A reader must not have to open this package to learn what was excused."""
    first, second = _summaries(tmp_path, *AGREEING)

    run_comparison.report_comparison(first, second)
    out = capsys.readouterr().out

    for field, reason in run_comparison.NON_REPRODUCIBLE_FIELDS.items():
        assert f"[excused] {field}: {reason}" in out


def test_a_missing_file_is_a_message_and_a_distinct_code(tmp_path, capsys):
    first, _ = _summaries(tmp_path, *AGREEING)

    code = run_comparison.report_comparison(first, tmp_path / "absent.json")
    out = capsys.readouterr().out

    assert code == run_comparison.COULD_NOT_COMPARE
    assert code != run_comparison.DIFFERENCES_FOUND, (
        "'these runs disagree' and 'I could not tell' are different answers, "
        "and a caller scripting this gate has to tell them apart"
    )
    assert "absent.json" in out and "does not exist" in out
    assert "Traceback" not in out


def test_a_malformed_file_is_a_message_and_not_a_traceback(tmp_path, capsys):
    first, second = _summaries(tmp_path, *AGREEING)
    second.write_text("{not json", encoding="utf-8")

    code = run_comparison.report_comparison(first, second)
    out = capsys.readouterr().out

    assert code == run_comparison.COULD_NOT_COMPARE
    assert "not valid JSON" in out


def test_the_view_strips_by_bare_name_at_any_depth(tmp_path):
    """A limitation, recorded rather than papered over.

    ``deterministic_view`` drops an excused key wherever it appears, not only
    at the depth the summary writes it. Every place the driver writes one of
    these names today is a timing or a timestamp, so nothing real is hidden;
    but a future payload that used ``latency_ms`` for a configured value
    rather than a measured one would have a genuine difference silently
    excused. Whoever adds such a field should read this test and make
    ``NON_REPRODUCIBLE_FIELDS`` a set of paths instead.
    """
    first, second = _summaries(
        tmp_path,
        {"configuration": {"latency_ms": 10}},
        {"configuration": {"latency_ms": 20}},
    )

    assert run_comparison.compare_runs(first, second) == []


def test_a_rewritten_literal_is_not_a_difference(tmp_path):
    """``--twice`` needs this: the two commands differ in one argument."""
    first, second = _summaries(
        tmp_path,
        {"provenance": {"command": ["--output-dir", "/x/run-1"]}},
        {"provenance": {"command": ["--output-dir", "/x/run-2"]}},
    )

    assert run_comparison.compare_runs(first, second) != []
    assert (
        run_comparison.compare_runs(
            first, second, {"/x/run-1": "<dir>", "/x/run-2": "<dir>"}
        )
        == []
    )


def test_every_rewrite_is_printed(tmp_path, capsys):
    """A substitution nobody can see is what this programme keeps removing."""
    first, second = _summaries(tmp_path, *AGREEING)

    run_comparison.report_comparison(first, second, substitutions={"/x": "<dir>"})

    assert "[rewritten] /x -> <dir>" in capsys.readouterr().out


def test_a_rewrite_does_not_hide_a_difference_elsewhere(tmp_path):
    """Rewriting the directory must not excuse the rest of the command."""
    first, second = _summaries(
        tmp_path,
        {"provenance": {"command": ["--limit", "2", "--output-dir", "/x/run-1"]}},
        {"provenance": {"command": ["--limit", "5", "--output-dir", "/x/run-2"]}},
    )

    differences = run_comparison.compare_runs(
        first, second, {"/x/run-1": "<dir>", "/x/run-2": "<dir>"}
    )

    assert differences == ["provenance.command[1]: '2' then '5'"]


def test_the_console_script_and_the_flag_are_the_same_function():
    """Two surfaces were chosen; they are acceptable only while they cannot
    drift, and they cannot drift because there is one implementation."""
    assert bench.report_comparison is run_comparison.report_comparison
    assert bench.compare_runs is run_comparison.compare_runs
    assert bench.deterministic_view is run_comparison.deterministic_view
    assert bench.NON_REPRODUCIBLE_FIELDS is run_comparison.NON_REPRODUCIBLE_FIELDS


@pytest.mark.parametrize("payloads", [AGREEING, DISAGREEING], ids=["agree", "differ"])
def test_the_console_script_and_the_flag_report_the_same_thing(
    payloads, tmp_path, monkeypatch, capsys
):
    first, second = _summaries(tmp_path, *payloads)

    from_script = run_comparison.main([str(first), str(second)])
    script_output = capsys.readouterr().out

    monkeypatch.setattr(
        sys, "argv", ["run_cke_benchmark.py", "--compare-runs", str(first), str(second)]
    )
    with pytest.raises(SystemExit) as exited:
        bench.main()
    flag_output = capsys.readouterr().out

    assert exited.value.code == from_script
    assert flag_output == script_output


def test_the_flag_compares_before_the_benchmark_machinery_starts(
    tmp_path, monkeypatch, capsys
):
    """Diffing two files that already exist must not need a model.

    Past the flag lie the environment report and a strict ``TokenCounter``,
    which refuses on a machine with no cached tokenizer. If the branch moved
    below them, this test would reach the network and conftest's guard would
    refuse it — which is the failure being prevented, not an accident.
    """
    first, second = _summaries(tmp_path, *AGREEING)
    monkeypatch.setattr(
        sys, "argv", ["run_cke_benchmark.py", "--compare-runs", str(first), str(second)]
    )

    with pytest.raises(SystemExit) as exited:
        bench.main()

    out = capsys.readouterr().out
    assert exited.value.code == 0
    assert "CKE environment report" not in out
    assert (
        out.splitlines()[0] == f"[compare] {first}"
    ), "something ran before the comparison did"


def test_a_file_written_by_only_one_run_is_a_difference(tmp_path):
    first, second = tmp_path / "one", tmp_path / "two"
    for directory in (first, second):
        directory.mkdir()
        (directory / "summary.json").write_text("{}", encoding="utf-8")
    (first / "ablation.json").write_text("{}", encoding="utf-8")

    assert run_comparison.compare_runs(first, second) == [
        "ablation.json: written by the first run only"
    ]


def test_two_directories_holding_nothing_do_not_agree(tmp_path, capsys):
    """Agreeing on nothing is not a pass.

    Two empty directories have no difference to find, so a comparison that
    only looked for differences would call this the gate passing.
    """
    first, second = tmp_path / "one", tmp_path / "two"
    first.mkdir()
    second.mkdir()

    code = run_comparison.report_comparison(first, second)
    out = capsys.readouterr().out

    assert code == run_comparison.COULD_NOT_COMPARE
    assert "nothing to compare" in out


def test_a_file_that_is_not_utf8_is_a_message_and_names_the_file(tmp_path, capsys):
    """json.load raises UnicodeDecodeError here, which is neither
    JSONDecodeError nor OSError, so it escaped both handlers."""
    first, second = _summaries(tmp_path, *AGREEING)
    second.write_bytes(b'{"rag_k10_em": "\xff\xfe"}')

    code = run_comparison.report_comparison(first, second)
    out = capsys.readouterr().out

    assert code == run_comparison.COULD_NOT_COMPARE
    assert "not valid UTF-8" in out
    assert second.name in out
    assert "Traceback" not in out


def test_every_unreadable_file_is_named(tmp_path, capsys):
    """The missing-file message names its file; so must the others."""
    first, second = _summaries(tmp_path, *AGREEING)
    second.write_text("{not json", encoding="utf-8")

    run_comparison.report_comparison(first, second)

    assert second.name in capsys.readouterr().out


class _StubRuns:
    """Stands in for ``subprocess`` so ``--twice`` performs no benchmark.

    It records the directory each pass was given and writes the summary that
    pass would have written, which is all the orchestration needs to be
    checked against.

    The summary it writes carries ``provenance.command``, as a real one does.
    That detail is not decoration: the first version of these tests wrote
    payloads with no path in them, every test passed, and the first real
    ``--twice`` run reported a difference on that field and could never have
    reported agreement. A stub that omits the payload's awkward part tests
    the stub.
    """

    def __init__(self, summaries, rows=None, returncode=0):
        self.summaries = list(summaries)
        # A real run writes per-item results beside the summary, and the gate
        # has to compare those too.
        self.rows = list(rows) if rows is not None else [[{"em": 1.0}]] * 2
        self.returncode = returncode
        self.commands: list[list[str]] = []

    def __getattr__(self, name):
        """Everything but `run` is the real module.

        The driver calls `subprocess.run` for the two passes and also reads
        `subprocess.SubprocessError` while deciding whether the output
        directory is usable. A stub that replaced the module wholesale broke
        the second, which is a stub shaped like the test rather than like the
        thing it stands in for.
        """
        return getattr(subprocess, name)

    def run(self, command, **kwargs):
        if command[0] != sys.executable:
            # git, asked whether the output directory is ignored. This stub
            # stands in for the benchmark run, not for every subprocess the
            # driver makes; answering that one itself would be testing the
            # stub's idea of .gitignore.
            return subprocess.run(command, **kwargs)

        self.commands.append(list(command))
        directory = Path(command[command.index("--output-dir") + 1])
        directory.mkdir(parents=True, exist_ok=True)
        index = len(self.commands) - 1
        payload = {
            **self.summaries[index],
            "provenance": {"command": list(command)},
        }
        (directory / "summary.json").write_text(json.dumps(payload), encoding="utf-8")
        (directory / "full_results_hotpotqa.json").write_text(
            json.dumps(self.rows[index]), encoding="utf-8"
        )

        class _Completed:
            returncode = self.returncode

        return _Completed()

    @property
    def output_dirs(self):
        return [command[command.index("--output-dir") + 1] for command in self.commands]


def _twice(monkeypatch, stub, tmp_path, extra=()):
    monkeypatch.setattr(bench, "subprocess", stub)
    monkeypatch.setattr(
        sys,
        "argv",
        ["run_cke_benchmark.py", "--twice", "--output-dir", str(tmp_path), *extra],
    )
    with pytest.raises(SystemExit) as exited:
        bench.main()
    return exited.value.code


def test_twice_runs_into_two_distinct_directories(tmp_path, monkeypatch, capsys):
    stub = _StubRuns(AGREEING)

    code = _twice(monkeypatch, stub, tmp_path)
    out = capsys.readouterr().out

    assert code == 0
    assert len(stub.output_dirs) == 2
    assert len(set(stub.output_dirs)) == 2, (
        "both passes wrote to one directory, so the second overwrote the "
        "first and the comparison compared a run with itself"
    )
    for directory in stub.output_dirs:
        assert Path(directory).parent == tmp_path
        assert f"[compare] {directory}" in out


def test_twice_compares_the_per_item_results_not_only_the_summary(
    tmp_path, monkeypatch, capsys
):
    """Aggregates can agree while the rows behind them do not.

    A different predicted answer or a different retrieved document that
    leaves EM, F1 and the medians unchanged is invisible in summary.json and
    plain in full_results_*.json. The first version of --twice compared only
    the summary and would have passed this.
    """
    stub = _StubRuns(
        AGREEING,
        rows=[[{"em": 1.0, "predicted": "Rome"}], [{"em": 1.0, "predicted": "Paris"}]],
    )

    code = _twice(monkeypatch, stub, tmp_path)
    out = capsys.readouterr().out

    assert code == run_comparison.DIFFERENCES_FOUND
    assert "full_results_hotpotqa.json" in out
    assert "'Rome' then 'Paris'" in out


def test_twice_reports_a_difference_between_its_own_two_runs(
    tmp_path, monkeypatch, capsys
):
    stub = _StubRuns(DISAGREEING)

    code = _twice(monkeypatch, stub, tmp_path)

    assert code == run_comparison.DIFFERENCES_FOUND
    assert "rag_k10_em: 0.5 then 0.6" in capsys.readouterr().out


def test_twice_passes_the_run_arguments_to_both_passes(tmp_path, monkeypatch):
    stub = _StubRuns(AGREEING)

    _twice(monkeypatch, stub, tmp_path, extra=["--limit", "3", "--skip-download"])

    for command in stub.commands:
        assert "--limit" in command and "3" in command
        assert "--skip-download" in command
        assert "--twice" not in command, "the passes would recurse"
        assert command.count("--output-dir") == 1, (
            "the parent directory was passed through as well as the pass's "
            "own, so which one argparse honoured decided the gate"
        )


def test_twice_accepts_the_joined_form_of_the_parent_directory(tmp_path, monkeypatch):
    """``--output-dir=x`` is one argv entry, and stripping only the split
    form leaves the parent in the passthrough."""
    stub = _StubRuns(AGREEING)
    monkeypatch.setattr(bench, "subprocess", stub)
    monkeypatch.setattr(
        sys,
        "argv",
        ["run_cke_benchmark.py", "--twice", f"--output-dir={tmp_path}"],
    )

    with pytest.raises(SystemExit) as exited:
        bench.main()

    assert exited.value.code == 0
    for command in stub.commands:
        assert command.count("--output-dir") == 1
        assert not any(entry.startswith("--output-dir=") for entry in command)


def test_twice_refuses_an_output_dir_that_would_dirty_the_tree(monkeypatch, capsys):
    """The first pass's files are on disk when the second reads git status.

    An unignored directory inside the repository therefore makes the second
    run record a dirtier tree than the first, and the comparison reports a
    provenance difference that is this command's own doing. Found by review.
    """
    stub = _StubRuns(AGREEING)
    # git answers for a path whether or not it exists, and the guard refuses
    # before anything is created — but a version of this that did not would
    # leave the directory behind, so it is removed either way.
    inside = ROOT / "gate-output-under-test"

    monkeypatch.setattr(bench, "subprocess", stub)
    monkeypatch.setattr(
        sys, "argv", ["run_cke_benchmark.py", "--twice", "--output-dir", str(inside)]
    )
    try:
        with pytest.raises(SystemExit) as exited:
            bench.main()
        out = capsys.readouterr().out

        assert exited.value.code == run_comparison.COULD_NOT_COMPARE
        assert stub.commands == [], "it ran the benchmark before refusing"
        assert "git does not ignore it" in out
        assert not inside.exists(), "it created the directory before refusing"
    finally:
        shutil.rmtree(inside, ignore_errors=True)


def test_twice_accepts_a_directory_git_ignores(monkeypatch, capsys):
    """results/ is the default and is ignored, so the guard must let it pass.

    A guard that refused every path inside the repository would refuse the
    documented invocation.
    """
    stub = _StubRuns(AGREEING)
    ignored = ROOT / "results" / "gate-check"

    monkeypatch.setattr(bench, "subprocess", stub)
    monkeypatch.setattr(
        sys, "argv", ["run_cke_benchmark.py", "--twice", "--output-dir", str(ignored)]
    )
    try:
        with pytest.raises(SystemExit) as exited:
            bench.main()

        assert exited.value.code == 0
        assert len(stub.commands) == 2
        assert "git does not ignore it" not in capsys.readouterr().out
    finally:
        shutil.rmtree(ignored, ignore_errors=True)


def test_twice_stops_when_a_run_fails_rather_than_comparing_nothing(
    tmp_path, monkeypatch, capsys
):
    stub = _StubRuns(AGREEING, returncode=3)

    code = _twice(monkeypatch, stub, tmp_path)
    out = capsys.readouterr().out

    assert code == 3, "the failing run's code, not a comparison verdict"
    assert len(stub.commands) == 1, "the second pass ran after the first failed"
    assert "nothing to compare" in out
