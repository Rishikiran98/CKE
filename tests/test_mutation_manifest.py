"""Every mutation must still describe a way to break the code.

A mutation whose target text has moved is not a weaker check, it is no check
at all: the runner refuses it rather than skipping quietly, which is right,
but the refusal only surfaces when somebody runs that suite.

Two had been stale for some time when the mutation workflow first ran in CI
and found them:

* "the coverage floor is lowered" still named ``fail_under = 78.0`` after the
  floor was raised to 78.5.
* "the orchestrator's alias overwrite is fixed without the xfail being
  retired" named a loop that had since been deleted, along with the xfail it
  referred to.

Neither was noticed because each suite is only run when somebody touches it,
and the change that invalidated each was in a different suite's territory.
These checks run in the ordinary test suite in milliseconds, so a mutation
goes stale for exactly as long as it takes to run the tests once.
"""

from __future__ import annotations

import json
import pathlib

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "scripts" / "mutation" / "mutations.json"


def _mutations():
    """(suite, index, mutation) for every mutation in the manifest."""
    suites = json.loads(MANIFEST.read_text(encoding="utf-8"))
    return [
        (suite, index, mutation)
        for suite, mutations in sorted(suites.items())
        for index, mutation in enumerate(mutations)
    ]


def _ids():
    return [f"{suite}[{index}]" for suite, index, _ in _mutations()]


@pytest.mark.parametrize("case", _mutations(), ids=_ids())
def test_every_mutation_still_applies(case):
    """Its target text must appear exactly once in the file it names.

    Zero means the code moved and the mutation describes nothing. More than
    one means the runner would patch an arbitrary occurrence, so what it
    tested would depend on where the text happened to appear.
    """
    suite, _, mutation = case
    target = ROOT / mutation["path"]

    assert target.exists(), (
        f"{suite}: {mutation['label']!r} names {mutation['path']}, "
        f"which does not exist"
    )

    occurrences = target.read_text(encoding="utf-8").count(mutation["old"])
    assert occurrences == 1, (
        f"{suite}: {mutation['label']!r} matches its target text "
        f"{occurrences} times in {mutation['path']}. The mutation no longer "
        f"describes a way to break the code — update its target, or delete "
        f"it if what it guarded is gone."
    )


@pytest.mark.parametrize("case", _mutations(), ids=_ids())
def test_every_mutation_changes_something(case):
    """A mutation whose replacement equals its target breaks nothing."""
    suite, _, mutation = case

    assert (
        mutation["old"] != mutation["new"]
    ), f"{suite}: {mutation['label']!r} replaces its target with itself"


@pytest.mark.parametrize("case", _mutations(), ids=_ids())
def test_every_mutation_names_the_tests_that_should_catch_it(case):
    """The runner passes these straight to pytest, so they must exist."""
    suite, _, mutation = case

    for path in str(mutation["tests"]).split():
        assert (ROOT / path).exists(), (
            f"{suite}: {mutation['label']!r} names {path}, which does not "
            f"exist, so the runner would invoke pytest on nothing"
        )


def test_the_manifest_holds_no_duplicate_labels():
    """A survivor is reported by label, so two of a name is one nobody finds."""
    labels = [f"{suite}: {mutation['label']}" for suite, _, mutation in _mutations()]

    assert len(labels) == len(set(labels)), sorted(
        label for label in labels if labels.count(label) > 1
    )
