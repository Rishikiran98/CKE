#!/usr/bin/env python3
"""Apply one source mutation at a time and require a test to notice.

A passing suite says the tests did not fail; it does not say they could. Each
mutation here is a specific way a guarantee could break. If the suite still
passes with the mutation applied, the guarantee is asserted and not checked,
and this script reports it as a survivor.

Every mutation is restored from the pristine text captured before it was
applied, in a ``finally``, so an interrupted run does not leave the tree
modified. A mutation whose target text is not found is refused rather than
skipped quietly: it means the code moved and the mutation no longer describes
a real way to break it.

Usage:
    python scripts/mutation/mutate.py                # every suite
    python scripts/mutation/mutate.py reasoning      # one suite by name
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess  # nosec B404 - fixed argv, no shell
import sys
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SUITES_PATH = Path(__file__).with_name("mutations.json")


@dataclass(frozen=True)
class Mutation:
    """One way to break a guarantee, and the tests that should notice."""

    label: str
    path: str
    old: str
    new: str
    tests: str


def load_suites() -> dict[str, list[Mutation]]:
    with open(SUITES_PATH, encoding="utf-8") as handle:
        raw = json.load(handle)
    return {
        name: [Mutation(**entry) for entry in entries] for name, entries in raw.items()
    }


def _clear_bytecode() -> None:
    for cache in ROOT.rglob("__pycache__"):
        shutil.rmtree(cache, ignore_errors=True)


def run_mutation(mutation: Mutation) -> tuple[bool, str]:
    """Return whether a test caught it, and the last line of the run."""
    target = ROOT / mutation.path
    pristine = target.read_text(encoding="utf-8")
    if mutation.old not in pristine:
        return False, f"target text not found in {mutation.path}"

    try:
        target.write_text(
            pristine.replace(mutation.old, mutation.new, 1), encoding="utf-8"
        )
        _clear_bytecode()
        completed = subprocess.run(  # nosec B603 - fixed argv, no shell
            [
                sys.executable,
                "-m",
                "pytest",
                "-q",
                "-p",
                "no:cacheprovider",
                "--no-cov",
                *mutation.tests.split(),
            ],
            cwd=str(ROOT),
            capture_output=True,
            text=True,
            env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1"},
        )
    finally:
        target.write_text(pristine, encoding="utf-8")
        _clear_bytecode()

    lines = completed.stdout.strip().splitlines()
    return completed.returncode != 0, lines[-1] if lines else ""


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("suite", nargs="?", help="run only this suite")
    args = parser.parse_args()

    suites = load_suites()
    if args.suite:
        if args.suite not in suites:
            parser.error(f"unknown suite {args.suite!r}; have {sorted(suites)}")
        suites = {args.suite: suites[args.suite]}

    survivors: list[str] = []
    for name, mutations in suites.items():
        print(f"\n=== {name} ({len(mutations)} mutations)")
        for mutation in mutations:
            killed, detail = run_mutation(mutation)
            if not killed:
                survivors.append(f"{name}: {mutation.label}")
            print(f"{'KILLED  ' if killed else 'SURVIVED'}  {mutation.label}")
            print(f"          {detail}")

    print()
    if survivors:
        print(f"{len(survivors)} mutation(s) survived:")
        for survivor in survivors:
            print(f"  - {survivor}")
        return 1
    print("Every mutation was caught.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
