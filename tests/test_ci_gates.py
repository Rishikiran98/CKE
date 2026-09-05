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


@pytest.mark.parametrize("name", ["ci.yml", "lint.yml", "security.yml", "mutation.yml"])
def test_every_workflow_runs_on_pull_requests(name):
    """security.yml ran only on push to main, so it could never block a
    merge: a finding appeared afterwards, on a branch nobody was gating."""
    assert "pull_request" in _triggers(_workflow(name))


@pytest.mark.parametrize("name", ["ci.yml", "lint.yml", "security.yml", "mutation.yml"])
def test_no_workflow_restricts_pull_requests_to_one_branch(name):
    """ci.yml restricted pull_request to [main] while push covered
    [main, dev], so a pull request into dev ran only the linters."""
    pull_request = _triggers(_workflow(name)).get("pull_request")

    assert pull_request is None or "branches" not in (pull_request or {})


@pytest.mark.parametrize("name", ["ci.yml", "lint.yml", "security.yml", "mutation.yml"])
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


# ---------------------------------------------------------------------------
# The mutation suite is run by something other than its author
# ---------------------------------------------------------------------------


def _mutation_steps() -> list[dict]:
    workflow = _workflow("mutation.yml")
    return [step for job in workflow["jobs"].values() for step in job.get("steps", [])]


def test_the_mutation_suite_is_run_by_ci():
    """Sixty-seven mutations existed and nothing ran them but their author.

    Every pull request in this programme ended with "every mutation was
    caught", produced by a command one person ran on one machine. A survivor
    was found whenever somebody next happened to run the suite.
    """
    # Not "mentions mutate.py": the matrix is built by a step that runs
    # `mutate.py --list`, so looking for the filename alone stays true when
    # the step that actually runs the mutations is gutted. A mutation proved
    # exactly that. What matters is a step that runs a suite.
    runs = [
        run
        for step in _mutation_steps()
        if "mutate.py" in (run := str(step.get("run", ""))) and "--list" not in run
    ]

    assert runs, (
        "no step in mutation.yml runs a suite; the only mention of the "
        "script builds the job matrix, which runs nothing"
    )
    assert any("matrix.suite" in run for run in runs), (
        "the suite each job runs does not come from the matrix, so the jobs "
        "either all run the same thing or all run everything"
    )


def _listed_suites(capsys) -> list:
    """What `mutate.py --list` prints, run in process.

    In process rather than through subprocess.run: the workflow shells out,
    but a subprocess is invisible to coverage, and a branch the suite never
    executes is a branch the floor does not guard.
    """
    import json
    import sys

    import scripts.mutation.mutate as runner

    argv = sys.argv
    sys.argv = ["mutate.py", "--list"]
    try:
        assert runner.main() == 0
    finally:
        sys.argv = argv
    return json.loads(capsys.readouterr().out)


def test_the_job_matrix_is_read_from_the_suite_file(capsys):
    """A hardcoded list is a list that goes stale.

    The matrix comes from `mutate.py --list`, so a suite added to
    mutations.json cannot be left unrun by forgetting to name it here. This
    asserts the workflow holds no copy of the names.
    """
    listed = _listed_suites(capsys)
    text = (WORKFLOWS / "mutation.yml").read_text(encoding="utf-8")

    assert listed, "the mutation runner reports no suites at all"
    for suite in listed:
        assert f'"{suite}"' not in text and f"'{suite}'" not in text, (
            f"mutation.yml names the {suite!r} suite. Build the matrix from "
            f"`mutate.py --list` instead, so a new suite cannot go unrun."
        )


def test_every_suite_in_the_file_is_reachable_by_name(capsys):
    """--list and the runner's own argument parser must agree.

    The matrix passes each listed name straight back to the script, which
    errors on an unknown suite; if the two disagreed, CI would fail on a name
    the file contains.
    """
    import scripts.mutation.mutate as runner

    assert set(_listed_suites(capsys)) == set(runner.load_suites())
