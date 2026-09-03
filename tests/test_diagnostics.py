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
    degradation_summary,
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
    """A package that raises on import is not the same as one that is missing.

    The probe locates the package first, so a failure after that point is a
    broken installation, including a transitive ModuleNotFoundError raised by
    one of the package's own dependencies.
    """

    def _explode(name):
        raise ModuleNotFoundError("No module named 'torch'")

    monkeypatch.setattr(diagnostics.importlib, "import_module", _explode)
    # "json" exists, so find_spec succeeds and the import failure is reached.
    report = environment_report(dependencies=[("json", "json-stdlib", "test")])

    assert report.dependencies[0].available is False
    assert "ModuleNotFoundError during import" in report.dependencies[0].error


def test_report_calls_an_absent_package_absent(monkeypatch):
    report = environment_report(
        dependencies=[("no_such_module_xyz", "nonexistent", "test")]
    )

    assert report.dependencies[0].available is False
    assert "No module named" in report.dependencies[0].error
    assert "during import" not in report.dependencies[0].error


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


def test_a_repeated_reason_is_recorded_once(caplog):
    """A degradation reached from inside a loop must not repeat.

    _fuzzy_score ran once per candidate entity, so an identical reason was
    logged per iteration and appended to degraded_reason each time, growing it
    quadratically.
    """

    class Looping(DegradationMixin):
        def __init__(self):
            self._init_degradation(False)

        def work(self, n):
            for _ in range(n):
                self._degrade("the same cause every time")

    with caplog.at_level(logging.WARNING):
        component = Looping()
        component.work(50)

    assert component.degraded_reason == "the same cause every time"
    assert len(caplog.records) == 1
    assert len(environment_report().degradations) == 1


def test_distinct_reasons_are_still_all_kept():
    class Varied(DegradationMixin):
        def __init__(self):
            self._init_degradation(False)
            self._degrade("first")
            self._degrade("second")
            self._degrade("first")

    assert Varied().degraded_reason == "first; second"


def test_degradation_summary_reports_what_degraded():
    """The report printed before a run cannot list degradations.

    Nothing is constructed yet at that point, so its degradation section
    always read "none". This is the other half, printed once work is done.
    """
    assert "No component degraded" in degradation_summary()

    Component(reason="the embedder is hashing")
    summary = degradation_summary()

    assert "DEGRADED COMPONENTS" in summary
    assert "not valid" in summary
    assert "the embedder is hashing" in summary


def test_a_reason_containing_the_separator_is_still_deduplicated():
    """Reasons were recovered by splitting degraded_reason on "; ".

    CoreferenceResolver joins its model failures with that separator, so an
    identical compound reason failed the membership check and was appended
    every time it recurred.
    """

    class Compound(DegradationMixin):
        def __init__(self):
            self._init_degradation(False)

    component = Compound()
    reason = "models failed: a: boom; b: bang"
    for _ in range(3):
        component._degrade(reason)

    assert component.degraded_reason == reason


def test_a_distinct_reason_matching_a_fragment_is_not_suppressed():
    class Compound(DegradationMixin):
        def __init__(self):
            self._init_degradation(False)

    component = Compound()
    component._degrade("alpha; beta")
    component._degrade("beta")

    assert component.degraded_reason == "alpha; beta; beta"


def test_require_strict_component_refuses_a_degraded_injection():
    from cke.diagnostics import require_strict_component

    class Degraded:
        degraded = True
        degraded_reason = "its model is missing"
        strict = False

    class Strict:
        degraded = False
        strict = True

    with pytest.raises(DegradedComponentError, match="already degraded"):
        require_strict_component("Pipeline", Degraded(), "reasoner", True)

    with pytest.raises(DegradedComponentError, match="does not declare"):
        require_strict_component("Pipeline", object(), "reasoner", True)

    # Healthy and strict is accepted, and a non-strict caller accepts anything.
    require_strict_component("Pipeline", Strict(), "reasoner", True)
    require_strict_component("Pipeline", Degraded(), "reasoner", False)
    require_strict_component("Pipeline", None, "reasoner", True)
