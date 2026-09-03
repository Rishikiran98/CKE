"""Coverage has a floor, it measures the library, and it only rises.

CI measured nothing; the first measurement, with test files excluded, was
82.56% of the package's statements. The floor under [tool.coverage.report]
started there. This pins it, so lowering it fails a test a reviewer sees; a
PR that raises it raises the recorded value here alongside.
"""

from __future__ import annotations

import pathlib
import tomllib

ROOT = pathlib.Path(__file__).resolve().parents[1]

#: The floor as it stands. Raise it with the configuration; never lower it.
FLOOR = 82.5


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


def test_coverage_measures_the_library_not_the_tests():
    """Test files count as covered lines and inflate the figure by two points."""
    run = _coverage_config()["run"]

    assert run["source"] == ["cke"]
    assert "cke/tests/*" in run["omit"]


def test_ci_runs_the_suite_under_coverage():
    """A floor nobody measures against holds nothing."""
    workflow = (ROOT / ".github" / "workflows" / "ci.yml").read_text(encoding="utf-8")
    run_lines = [
        line.strip()[len("run:") :].strip()
        for line in workflow.splitlines()
        if line.strip().startswith("run:")
    ]

    assert any(
        line.startswith("pytest") and "--cov=cke" in line for line in run_lines
    ), run_lines
