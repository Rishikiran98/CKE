"""The type-checked module list only grows.

mypy was never run despite most functions carrying return annotations. It now
runs in CI on the modules listed under ``[tool.mypy]`` in pyproject.toml. The
list is a ratchet: a module is added when it checks clean and never removed.
This pins the floor, so shortening the list fails a test a reviewer sees; a
PR that lengthens it raises the floor here alongside.
"""

from __future__ import annotations

import pathlib
import tomllib

ROOT = pathlib.Path(__file__).resolve().parents[1]

#: The list as it stands. A PR that adds a module adds it here in the same
#: change, so the floor moves with the list and a later removal is caught;
#: a subset check would let an addition slip in unrecorded and out again
#: unnoticed. Never remove an entry.
FLOOR = {
    "cke/models.py",
    "cke/pipeline/types.py",
    "cke/retrieval/path_types.py",
    "cke/schema",
    "cke/diagnostics.py",
    "cke/datasets/hotpot_loader.py",
    "cke/datasets/locomo_loader.py",
    "cke/datasets/musique_loader.py",
    "cke/datasets/wiki2_loader.py",
    "cke/evaluation/span_qa.py",
    "cke/evaluation/token_counter.py",
    "cke/observability/token_tracker.py",
    "cke/reasoning/reasoner_adapter.py",
    "cke/reasoning/verifier.py",
    "cke/trust/confidence_calibrator.py",
}

#: Modules that decide something, as opposed to modules that only declare a
#: shape. The list held four entries and every one was a dataclass or enum
#: module, so mypy reported success over code that could not be wrong. A
#: ratchet made only of those can be satisfied forever without checking
#: anything, so at least one logic module must stay in it.
_DECLARATION_ONLY = {
    "cke/models.py",
    "cke/pipeline/types.py",
    "cke/retrieval/path_types.py",
    "cke/schema",
}


def _mypy_config() -> dict:
    with (ROOT / "pyproject.toml").open("rb") as handle:
        return tomllib.load(handle)["tool"]["mypy"]


def test_the_list_checks_code_that_decides_something():
    """Six files passed and all six were declaration-only, so `Success: no
    issues found` was reported over nothing that could be wrong."""
    listed = set(_mypy_config()["files"])

    assert listed - _DECLARATION_ONLY, (
        "every checked module only declares a shape; mypy is reporting "
        "success over code that cannot be wrong"
    )


def test_the_checked_modules_are_exactly_the_recorded_floor():
    listed = set(_mypy_config()["files"])

    dropped = FLOOR - listed
    assert not dropped, f"the mypy list was shortened; dropped: {sorted(dropped)}"
    added = listed - FLOOR
    assert not added, (
        f"the mypy list grew by {sorted(added)}; add them to FLOOR in this "
        "test so the ratchet records them"
    )


def test_every_listed_module_exists():
    for entry in _mypy_config()["files"]:
        assert (ROOT / entry).exists(), entry


def test_the_lint_workflow_runs_mypy():
    """A ratchet nobody runs holds nothing."""
    workflow = (ROOT / ".github" / "workflows" / "lint.yml").read_text(encoding="utf-8")
    run_lines = [
        line.strip()[len("run:") :].strip()
        for line in workflow.splitlines()
        if line.strip().startswith("run:")
    ]
    assert "mypy" in run_lines, run_lines


def test_the_ratchet_reports_errors_only_where_it_looks():
    """follow_imports must stay silent, or the list cannot start small.

    With imports followed, the four seed modules report the errors of every
    module they import, which is most of the package; with them silent, a
    listed module's imports supply types but their own errors wait until they
    are listed.
    """
    config = _mypy_config()
    assert config["follow_imports"] == "silent"
    assert config["ignore_missing_imports"] is True
