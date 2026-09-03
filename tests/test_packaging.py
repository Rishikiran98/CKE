"""The package installs and says what it is.

``pyproject.toml`` had no ``[project]`` table, so ``pip install -e .`` failed
and no console entry point existed. These run against the installed
distribution, so they need ``pip install -e .`` first, which is what CI does.
"""

from __future__ import annotations

import importlib
import importlib.metadata
import pathlib
import re
import tomllib

import cke

ROOT = pathlib.Path(__file__).resolve().parents[1]

_REQUIREMENT_NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*")


def _project() -> dict:
    with (ROOT / "pyproject.toml").open("rb") as handle:
        return tomllib.load(handle)["project"]


def _names(requirements) -> set[str]:
    names = set()
    for line in requirements:
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        match = _REQUIREMENT_NAME.match(line)
        assert match, f"unparseable requirement {line!r}"
        names.add(match.group(0).lower().replace("_", "-"))
    return names


def test_the_installed_distribution_reports_the_package_version():
    """One version, read from cke/version.py, not typed twice."""
    assert importlib.metadata.version("cke") == cke.__version__


def test_every_console_script_resolves_to_a_callable():
    scripts = _project()["scripts"]
    assert scripts, "no console scripts declared"
    for name, target in scripts.items():
        module_name, _, attribute = target.partition(":")
        module = importlib.import_module(module_name)
        assert callable(getattr(module, attribute)), f"{name} -> {target}"


def test_the_declared_console_scripts_are_the_installed_ones():
    """The table and the installation agree, and the set is the known one.

    Pinning the names means a command cannot disappear from the table
    without a reviewer seeing a test change alongside it.
    """
    installed = {
        entry.name: entry.value
        for entry in importlib.metadata.entry_points(group="console_scripts")
        if entry.name.startswith("cke-")
    }
    assert installed == _project()["scripts"]
    assert set(installed) == {
        "cke-eval",
        "cke-experiment",
        "cke-reasoning-eval",
        "cke-retrieval-eval",
    }


def test_the_declared_dependencies_are_requirements_txt():
    """The dependency list is read from requirements.txt, not typed twice."""
    declared = importlib.metadata.requires("cke") or []
    wanted = (ROOT / "requirements.txt").read_text(encoding="utf-8").splitlines()

    assert _names(declared) == _names(wanted)
    assert _names(declared), "no dependencies declared"


def test_the_python_requirement_matches_ci():
    workflow = (ROOT / ".github" / "workflows" / "ci.yml").read_text(encoding="utf-8")
    ci_version = re.search(r'python-version:\s*"([\d.]+)"', workflow).group(1)

    assert _project()["requires-python"] == f">={ci_version}"
