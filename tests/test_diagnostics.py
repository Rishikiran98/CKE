"""Tests for the degradation contract and the environment report.

The rule under test: no component may lose capability silently. Each
degradation must warn, mark the object, and refuse outright under strict.
"""

from __future__ import annotations

import logging

import pytest

from cke import diagnostics
from cke.diagnostics import (
    DegradationMixin,
    DegradedComponentError,
    clear_runtime_state,
    declare_degradation,
    environment_report,
    record_loaded_model,
)


@pytest.fixture(autouse=True)
def _clean_runtime_state():
    clear_runtime_state()
    yield
    clear_runtime_state()


class Component(DegradationMixin):
    """Minimal component that degrades on construction."""

    def __init__(self, strict: bool = False, reason: str = "a model is missing"):
        self._init_degradation(strict)
        self._degrade(reason)


def test_degradation_warns_flags_and_records(caplog):
    """All three obligations are met on a single degradation."""
    with caplog.at_level(logging.WARNING):
        component = Component()

    assert component.degraded is True
    assert component.degraded_reason == "a model is missing"
    assert any(record.levelno == logging.WARNING for record in caplog.records)
    assert "a model is missing" in caplog.text

    report = environment_report()
    assert report.is_degraded
    assert report.degradations[0].component == "Component"


def test_strict_raises_instead_of_degrading():
    with pytest.raises(DegradedComponentError) as excinfo:
        Component(strict=True)

    message = str(excinfo.value)
    assert "Component" in message
    assert "a model is missing" in message
    assert "strict=True" in message


def test_strict_failure_is_not_recorded_as_a_degradation():
    """A refused run has not degraded; it has stopped."""
    with pytest.raises(DegradedComponentError):
        Component(strict=True)

    assert environment_report().is_degraded is False


def test_a_healthy_component_is_not_degraded():
    class Healthy(DegradationMixin):
        def __init__(self):
            self._init_degradation(False)

    component = Healthy()
    assert component.degraded is False
    assert component.degraded_reason == ""


def test_multiple_reasons_are_all_kept():
    class TwiceDegraded(DegradationMixin):
        def __init__(self):
            self._init_degradation(False)
            self._degrade("first cause")
            self._degrade("second cause")

    component = TwiceDegraded()
    assert "first cause" in component.degraded_reason
    assert "second cause" in component.degraded_reason


def test_declare_degradation_works_without_an_object(caplog):
    with caplog.at_level(logging.WARNING):
        declare_degradation("some_loader", "config file is missing")

    assert "config file is missing" in caplog.text
    assert environment_report().degradations[0].component == "some_loader"

    with pytest.raises(DegradedComponentError):
        declare_degradation("some_loader", "config file is missing", strict=True)


def test_report_lists_dependency_status():
    report = environment_report(
        dependencies=[
            ("json", "json-stdlib", "a module that always imports"),
            ("no_such_module_xyz", "nonexistent-dist", "a module that never does"),
        ]
    )

    statuses = {dep.import_name: dep for dep in report.dependencies}
    assert statuses["json"].available is True
    assert statuses["no_such_module_xyz"].available is False
    assert statuses["no_such_module_xyz"].error
    assert report.missing == [statuses["no_such_module_xyz"]]


def test_report_distinguishes_a_broken_install_from_an_absent_one(monkeypatch):
    """A package that raises on import is not the same as one that is missing."""

    def _explode(name):
        raise RuntimeError("torch blew up")

    monkeypatch.setattr(diagnostics.importlib, "import_module", _explode)
    report = environment_report(dependencies=[("anything", "anything", "test")])

    assert report.dependencies[0].available is False
    assert "RuntimeError during import" in report.dependencies[0].error


def test_report_records_loaded_models():
    record_loaded_model("EmbeddingModel", "model-a", "model-a")
    report = environment_report()

    assert report.loaded_models[0].loaded == "model-a"
    assert "model-a" in report.render()


def test_render_states_plainly_when_a_run_is_invalid():
    Component(reason="the embedder is hashing")
    rendered = environment_report().render()

    assert "DEGRADED COMPONENTS" in rendered
    assert "not valid" in rendered
    assert "the embedder is hashing" in rendered


def test_report_is_json_serialisable():
    import json

    Component()
    record_loaded_model("EmbeddingModel", "m", "m")
    payload = environment_report().as_dict()

    assert (
        json.loads(json.dumps(payload))["degradations"][0]["component"] == "Component"
    )
