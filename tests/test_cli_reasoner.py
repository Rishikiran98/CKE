import sys

import pytest

from cke.diagnostics import DegradedComponentError, clear_runtime_state
from cke.experiments import run_experiment
import demo


@pytest.mark.needs_model
def test_demo_cli_runs_strict_and_answers(monkeypatch, capsys):
    """The demo has one path and every component on it is strict.

    This replaced a check that ``--reasoner template`` printed its own name:
    that flag routed to a template reasoner whose heuristics were written for
    the old demo's question, and the demo no longer offers a choice.
    """
    # The degradation registry is process-wide; earlier tests in the same
    # run leave entries in it, and the demo's environment report would
    # list them as this run's.
    clear_runtime_state()
    monkeypatch.setattr(sys, "argv", ["demo.py"])
    demo.main()
    out = capsys.readouterr().out
    # The closing lines, not the opening report: that one is taken before
    # any component exists and says "Models loaded: none yet" on every run.
    assert "EmbeddingModel: sentence-transformers/all-MiniLM-L6-v2" in out
    assert "No component degraded during this run." in out
    assert "Rule applied located_in_transitivity" in out
    assert "Answer: Turkey" in out.splitlines()


def test_experiment_cli_supports_reasoner_flag(monkeypatch, capsys):
    """The flag is honoured on a smoke run, which must be opted into."""
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run_experiment.py",
            "--extractor",
            "rule",
            "--reasoner",
            "template",
            "--allow-degraded",
        ],
    )
    run_experiment.main()
    out = capsys.readouterr().out
    assert "Experiment results:" in out


def test_experiment_cli_refuses_to_report_on_the_built_in_sample(monkeypatch):
    """Without --dataset there is nothing to measure, so it must not report."""
    monkeypatch.setattr(
        sys,
        "argv",
        ["run_experiment.py", "--extractor", "rule", "--reasoner", "template"],
    )
    with pytest.raises(DegradedComponentError, match="no --dataset"):
        run_experiment.main()
