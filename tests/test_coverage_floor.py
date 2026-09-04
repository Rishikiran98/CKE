"""Coverage has a floor and a scope, and neither may shrink.

CI measured nothing; the first measurement of the library alone was 82.56% of
its statements, and the floor started there. It guarded cke/ while scripts/ —
the benchmark driver and the downloaders, the code that produces every number
this project reports — went unmeasured, and while one arm of every branch
counted as covered.

The scope now includes scripts/ and counts branches, so the figure describes
more of the repository and reads lower for it. The two numbers measure
different things and are not comparable; what is comparable is that neither
the floor nor the scope may narrow from here.
"""

from __future__ import annotations

import pathlib
import tomllib

ROOT = pathlib.Path(__file__).resolve().parents[1]

#: The floor as it stands, for the scope below. Raise it with the
#: configuration; never lower it, and never narrow the scope to raise it.
FLOOR = 78.5

#: What is measured. Widening this is what re-baselined the floor once, and
#: recording it here is what makes a narrowing visible.
SCOPE = ["cke", "scripts"]


def _coverage_config() -> dict:
    with (ROOT / "pyproject.toml").open("rb") as handle:
        return tomllib.load(handle)["tool"]["coverage"]


def test_the_floor_is_exactly_the_recorded_one():
    configured = _coverage_config()["report"]["fail_under"]

    assert configured >= FLOOR, f"the coverage floor was lowered to {configured}"
    assert configured == FLOOR, (
        f"the coverage floor rose to {configured}; record it as FLOOR in this "
        "test so it cannot be lowered again"
    )


def test_coverage_measures_the_drivers_as_well_as_the_library():
    """The floor guarded cke/ while every number the project reports came out
    of scripts/, which nothing measured."""
    assert _coverage_config()["run"]["source"] == SCOPE


def test_coverage_counts_branches():
    """Without this one arm of every if counts as covered, and the figure is
    an upper bound on a weaker metric than it appears to be."""
    assert _coverage_config()["run"].get("branch") is True


def test_coverage_measures_the_library_not_the_tests():
    """Test files count as covered lines, and inflated the figure by two
    points while a suite lived inside the package. The suite lives in tests/
    now; a second one inside cke/ would count itself again."""
    assert not (ROOT / "cke" / "tests").exists(), "a test directory is back in cke/"


def test_ci_runs_the_suite_under_coverage():
    """A floor nobody measures against holds nothing."""
    workflow = (ROOT / ".github" / "workflows" / "ci.yml").read_text(encoding="utf-8")
    run_lines = [
        line.strip()[len("run:") :].strip()
        for line in workflow.splitlines()
        if line.strip().startswith("run:")
    ]

    # "--cov" in any form, not "--cov=cke": what is measured is pinned by the
    # source assertion above, and asserting it twice meant widening coverage
    # to the drivers required editing this test as well as the config.
    assert any(
        line.startswith("pytest") and "--cov" in line for line in run_lines
    ), run_lines
