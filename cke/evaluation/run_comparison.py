"""Whether two runs of one command produced the same result.

This is the gate the production-readiness programme calls Gate 4: run the
benchmark twice and diff. Diffing the files themselves always reports a
difference, because a results file carries timings and a timestamp, so the
useful comparison is of everything :data:`NON_REPRODUCIBLE_FIELDS` does not
excuse. An empty list of differences is the pass.

The code lived in ``scripts/run_cke_benchmark.py`` and was reachable only by
importing a two-thousand-line driver from a Python prompt. It is here because
``pyproject.toml`` says console scripts are for entry points inside the
package, and a check nobody can run is a check that gets re-run by nobody: the
gate's evidence existed as prose in a pull request description, written by the
one person who could produce it.

Two surfaces call one function. ``cke-compare-runs a.json b.json`` is this
module's ``main``; ``run_cke_benchmark.py --compare-runs a.json b.json`` is the
same call from the driver. They cannot drift, because there is only one of
them.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from collections.abc import Mapping
from typing import Any, TextIO

__all__ = [
    "NON_REPRODUCIBLE_FIELDS",
    "ResultsUnreadable",
    "compare_runs",
    "deterministic_view",
    "main",
    "report_comparison",
    "substituted",
]

#: Fields of the results that cannot repeat, and why. A run is reproducible
#: everywhere else; naming the exceptions is what lets "run it twice and diff"
#: be a check rather than a hope.
NON_REPRODUCIBLE_FIELDS = {
    "started_at": "the wall clock reads differently on the second run",
    "median_latency_ms": "a timing, which no seed fixes",
    "rag_k10_median_latency_ms": "a timing, which no seed fixes",
    "cke_n12_median_latency_ms": "a timing, which no seed fixes",
    "latency_ms": "a timing, which no seed fixes",
}

#: Exit code when the two runs disagree about something a seed should fix.
DIFFERENCES_FOUND = 1

#: Exit code when a file could not be read or parsed. Distinct from the above:
#: "these runs disagree" and "I could not tell" are different answers, and a
#: caller scripting this gate needs to tell them apart.
COULD_NOT_COMPARE = 2


class ResultsUnreadable(Exception):
    """A results file could not be read, and the message says which one.

    Raised instead of letting the decoder's own exception out: json.load on a
    file that is not valid UTF-8 raises UnicodeDecodeError, which is neither
    JSONDecodeError nor OSError, so it escaped both handlers and produced a
    traceback where the exit code should have been. Found by review.
    """


def _read_json(path: Path) -> Any:
    """Load a results file, naming it in every way it can fail."""
    try:
        with open(path, encoding="utf-8") as handle:
            return json.load(handle)
    except UnicodeDecodeError as broken:
        raise ResultsUnreadable(f"{path}: not valid UTF-8 ({broken})") from broken
    except json.JSONDecodeError as broken:
        raise ResultsUnreadable(f"{path}: not valid JSON ({broken})") from broken


def deterministic_view(payload: Any) -> Any:
    """The part of a results payload that two runs must agree on exactly.

    Drops the fields NON_REPRODUCIBLE_FIELDS names, at any depth. Two runs of
    one command whose deterministic views differ have a defect in this
    harness, not noise.
    """
    if isinstance(payload, dict):
        return {
            key: deterministic_view(value)
            for key, value in payload.items()
            if key not in NON_REPRODUCIBLE_FIELDS
        }
    if isinstance(payload, list):
        return [deterministic_view(value) for value in payload]
    return payload


def substituted(payload: Any, substitutions: Mapping[str, str]) -> Any:
    """Rewrite literal strings anywhere in a payload before comparing.

    One caller needs this: ``--twice`` runs the same command into two output
    directories, and the recorded command line therefore carries a different
    directory in each run. That is a real difference, not noise, so it is not
    excused by name — it is rewritten, and ``report_comparison`` prints every
    rewrite it was given. A substitution nobody can see is the thing this
    programme keeps removing.
    """
    if isinstance(payload, dict):
        return {
            key: substituted(value, substitutions) for key, value in payload.items()
        }
    if isinstance(payload, list):
        return [substituted(value, substitutions) for value in payload]
    if isinstance(payload, str):
        for literal, placeholder in substitutions.items():
            payload = payload.replace(literal, placeholder)
    return payload


def _differences(left: Any, right: Any, path: str) -> list[str]:
    """Every place two deterministic views disagree, named by where it is."""
    if isinstance(left, dict) and isinstance(right, dict):
        differences: list[str] = []
        for key in sorted(set(left) | set(right)):
            where = f"{path}.{key}" if path else key
            if key not in left:
                differences.append(f"{where}: only in the second run")
            elif key not in right:
                differences.append(f"{where}: only in the first run")
            else:
                differences += _differences(left[key], right[key], where)
        return differences
    if isinstance(left, list) and isinstance(right, list):
        if len(left) != len(right):
            return [f"{path}: {len(left)} entries then {len(right)}"]
        return [
            difference
            for index, (a, b) in enumerate(zip(left, right))
            for difference in _differences(a, b, f"{path}[{index}]")
        ]
    if left != right:
        return [f"{path}: {left!r} then {right!r}"]
    return []


def compare_runs(
    first: Path,
    second: Path,
    substitutions: Mapping[str, str] | None = None,
) -> list[str]:
    """Where two runs' results disagree on anything a seed should have fixed.

    The gate for this work is "run it twice and diff". Diffing the files
    themselves always reports a difference, because they carry timings and a
    timestamp, so the useful comparison is of what NON_REPRODUCIBLE_FIELDS
    does not exclude. An empty list is the pass.
    """

    rewrites = substitutions or {}
    if first.is_dir() or second.is_dir():
        return _compare_directories(first, second, rewrites)

    left = substituted(deterministic_view(_read_json(first)), rewrites)
    right = substituted(deterministic_view(_read_json(second)), rewrites)
    return _differences(left, right, "")


def _compare_directories(
    first: Path,
    second: Path,
    substitutions: Mapping[str, str],
) -> list[str]:
    """Every results file the two runs wrote, not just the summary.

    Comparing only summary.json passed a gate that nondeterminism in the
    per-item rows could walk straight through: a different predicted answer
    or a different retrieved document that leaves the aggregate EM and the
    medians unchanged is invisible in the summary and plain in
    full_results_*.json. Found by review, on the first version of this.

    Only the .json files. The .md tables are rendered from them and the .png
    is a plot of them, so comparing the JSON compares what they are made of;
    comparing rendered text as text would report formatting as a defect.
    """
    names = sorted(
        {path.name for path in first.glob("*.json")}
        | {path.name for path in second.glob("*.json")}
    )
    if not names:
        raise ResultsUnreadable(
            f"neither {first} nor {second} holds a .json results file, so "
            f"there is nothing to compare — agreeing on nothing is not a pass"
        )

    differences: list[str] = []
    for name in names:
        left_path, right_path = first / name, second / name
        if not left_path.exists():
            differences.append(f"{name}: written by the second run only")
        elif not right_path.exists():
            differences.append(f"{name}: written by the first run only")
        else:
            left = substituted(deterministic_view(_read_json(left_path)), substitutions)
            right = substituted(
                deterministic_view(_read_json(right_path)), substitutions
            )
            differences += [
                f"{name} {entry}" for entry in _differences(left, right, "")
            ]
    return differences


def report_comparison(
    first: Path,
    second: Path,
    stream: TextIO | None = None,
    substitutions: Mapping[str, str] | None = None,
) -> int:
    """Compare two results files and say what differs. Returns an exit code.

    The one implementation both surfaces call, so that the console script and
    the driver's flag cannot come to disagree about what the gate means.

    Every difference is named. "The runs differ" without saying where is the
    kind of check this programme has spent its time removing.
    """
    out = stream if stream is not None else sys.stdout

    try:
        differences = compare_runs(first, second, substitutions)
    except FileNotFoundError as missing:
        print(f"[error] cannot compare: {missing.filename} does not exist", file=out)
        return COULD_NOT_COMPARE
    except ResultsUnreadable as unreadable:
        print(f"[error] cannot compare: {unreadable}", file=out)
        return COULD_NOT_COMPARE
    except OSError as unreadable:
        print(f"[error] cannot compare: {unreadable}", file=out)
        return COULD_NOT_COMPARE

    print(f"[compare] {first}", file=out)
    print(f"[compare] {second}", file=out)
    # Named, so a reader knows what was excused rather than having to find
    # this file to learn why a timing did not count as a difference.
    for field, reason in sorted(NON_REPRODUCIBLE_FIELDS.items()):
        print(f"[excused] {field}: {reason}", file=out)
    for literal, placeholder in sorted((substitutions or {}).items()):
        print(f"[rewritten] {literal} -> {placeholder}", file=out)

    if not differences:
        print(
            "\nThe two runs agree on everything a seed should have fixed.",
            file=out,
        )
        return 0

    print(f"\n{len(differences)} difference(s) a seed should have fixed:", file=out)
    for difference in differences:
        print(f"  {difference}", file=out)
    print(
        "\nTwo runs of one command that differ here have a defect in the "
        "harness, not noise.",
        file=out,
    )
    return DIFFERENCES_FOUND


def main(argv: list[str] | None = None) -> int:
    """``cke-compare-runs a/summary.json b/summary.json``."""
    parser = argparse.ArgumentParser(
        prog="cke-compare-runs",
        description=(
            "Check that two runs of one benchmark command agree on everything "
            "a seed should have fixed. Exits 0 when they agree, "
            f"{DIFFERENCES_FOUND} when they differ, and {COULD_NOT_COMPARE} "
            "when a file could not be read."
        ),
    )
    parser.add_argument(
        "first", help="a results file, or a whole output directory, from one run"
    )
    parser.add_argument("second", help="the same file or directory from the other run")
    args = parser.parse_args(argv)

    return report_comparison(Path(args.first), Path(args.second))


if __name__ == "__main__":  # pragma: no cover - exercised through main()
    raise SystemExit(main())
