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

#: Every module the list has ever held. Raise it when the list grows; never
#: lower it.
FLOOR = {
    "cke/models.py",
    "cke/pipeline/types.py",
    "cke/retrieval/path_types.py",
    "cke/schema",
}


def _mypy_config() -> dict:
    with (ROOT / "pyproject.toml").open("rb") as handle:
        return tomllib.load(handle)["tool"]["mypy"]


def test_the_checked_modules_include_everything_ever_checked():
    listed = set(_mypy_config()["files"])

    missing = FLOOR - listed
    assert not missing, f"the mypy list was shortened; dropped: {sorted(missing)}"


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
