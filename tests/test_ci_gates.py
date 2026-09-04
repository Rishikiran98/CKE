"""The gates can fail.

A green check is worth what it could have caught. Each property here is one a
workflow lost or never had: a demo step whose crash was discarded, a test job
that did not run on half the repository's pull requests, and a security scan
that could not block a merge and did not read the code that produces every
number the project reports.
"""

from __future__ import annotations

from pathlib import Path

import pytest

try:
    import yaml
except ImportError:  # pragma: no cover - PyYAML is a runtime dependency
    yaml = None

ROOT = Path(__file__).resolve().parents[1]
WORKFLOWS = ROOT / ".github" / "workflows"

pytestmark = pytest.mark.skipif(yaml is None, reason="PyYAML is not installed")


def _workflow(name: str) -> dict:
    with open(WORKFLOWS / name, encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def _triggers(workflow: dict) -> dict:
    # PyYAML reads a bare `on:` key as the boolean True.
    return workflow.get("on", workflow.get(True, {})) or {}


@pytest.mark.parametrize("name", ["ci.yml", "lint.yml", "security.yml"])
def test_every_workflow_runs_on_pull_requests(name):
    """security.yml ran only on push to main, so it could never block a
    merge: a finding appeared afterwards, on a branch nobody was gating."""
    assert "pull_request" in _triggers(_workflow(name))


@pytest.mark.parametrize("name", ["ci.yml", "lint.yml", "security.yml"])
def test_no_workflow_restricts_pull_requests_to_one_branch(name):
    """ci.yml restricted pull_request to [main] while push covered
    [main, dev], so a pull request into dev ran only the linters."""
    pull_request = _triggers(_workflow(name)).get("pull_request")

    assert pull_request is None or "branches" not in (pull_request or {})


@pytest.mark.parametrize("name", ["ci.yml", "lint.yml", "security.yml"])
def test_every_workflow_is_read_only_and_bounded(name):
    workflow = _workflow(name)

    assert workflow.get("permissions", {}).get("contents") == "read"
    for job in workflow["jobs"].values():
        assert job.get("timeout-minutes"), "a hung job should end, not hold a runner"


def test_a_piped_step_sets_pipefail():
    """`python demo.py | tee demo.out` under `bash -e` reports tee's status,
    so the demo could raise after printing its answer and the greps below
    would still match."""
    for path in sorted(WORKFLOWS.glob("*.yml")):
        workflow = _workflow(path.name)
        for job in workflow["jobs"].values():
            for step in job.get("steps", []):
                script = step.get("run", "")
                if "|" not in script or "||" in script:
                    continue
                assert "pipefail" in script, (
                    f"{path.name}: a piped step discards the exit status of "
                    f"everything but the last command"
                )


def test_the_security_scan_reads_the_code_that_produces_the_numbers():
    """It scanned cke only, so scripts/ — the benchmark drivers, the
    downloaders, 2,700 lines — was never scanned at all."""
    security = _workflow("security.yml")
    steps = security["jobs"]["security"]["steps"]
    scan = next(step["run"] for step in steps if "bandit" in step.get("run", ""))

    assert "cke" in scan and "scripts" in scan
