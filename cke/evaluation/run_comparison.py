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

    def _walk(left: Any, right: Any, path: str) -> list[str]:
        if isinstance(left, dict) and isinstance(right, dict):
            differences: list[str] = []
            for key in sorted(set(left) | set(right)):
                where = f"{path}.{key}" if path else key
                if key not in left:
                    differences.append(f"{where}: only in the second run")
                elif key not in right:
                    differences.append(f"{where}: only in the first run")
                else:
                    differences += _walk(left[key], right[key], where)
            return differences
        if isinstance(left, list) and isinstance(right, list):
            if len(left) != len(right):
                return [f"{path}: {len(left)} entries then {len(right)}"]
            return [
                difference
                for index, (a, b) in enumerate(zip(left, right))
                for difference in _walk(a, b, f"{path}[{index}]")
            ]
        if left != right:
            return [f"{path}: {left!r} then {right!r}"]
        return []

    rewrites = substitutions or {}
    with open(first, encoding="utf-8") as handle:
        left = substituted(deterministic_view(json.load(handle)), rewrites)
    with open(second, encoding="utf-8") as handle:
        right = substituted(deterministic_view(json.load(handle)), rewrites)
    return _walk(left, right, "")


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
    except json.JSONDecodeError as broken:
        print(f"[error] cannot compare: not valid JSON ({broken})", file=out)
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
    parser.add_argument("first", help="a results file from the first run")
    parser.add_argument("second", help="the same file from the second run")
    args = parser.parse_args(argv)

    return report_comparison(Path(args.first), Path(args.second))


if __name__ == "__main__":  # pragma: no cover - exercised through main()
    raise SystemExit(main())
