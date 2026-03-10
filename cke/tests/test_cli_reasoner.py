import sys

from cke.experiments import run_experiment
import demo


def test_demo_cli_supports_reasoner_flag(monkeypatch, capsys):
    monkeypatch.setattr(sys, "argv", ["demo.py", "--reasoner", "template"])
    demo.main()
    out = capsys.readouterr().out
    assert "Reasoner: template" in out


def test_experiment_cli_supports_reasoner_flag(monkeypatch, capsys):
    monkeypatch.setattr(
        sys,
        "argv",
        ["run_experiment.py", "--extractor", "rule", "--reasoner", "template"],
    )
    run_experiment.main()
    out = capsys.readouterr().out
    assert "Experiment results:" in out
