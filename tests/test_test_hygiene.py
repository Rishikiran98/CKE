"""The suite's own guarantees, held by tests rather than by good intentions.

`tests/conftest.py` blocks the network, clears the process-global degradation
registry, and clears the embedding-model caches. None of that survives being
quietly reverted unless something checks it, and this repository has already
had one gate — the coverage floor — that lived in a single string nobody
tested. These are the checks.
"""

from __future__ import annotations

import ast
import pathlib
import socket

import pytest

TESTS = pathlib.Path(__file__).resolve().parent
REPO_ROOT = TESTS.parent

#: Tests allowed to reach the network, each with the reason it must.
#: Adding to this list is a deliberate act: every entry is a test that is red
#: whenever the Hub is, and green only where the cache is already warm.
MAY_USE_THE_NETWORK = {
    "test_cli_reasoner.py::test_demo_cli_runs_strict_and_answers": (
        "The demo is the one end-to-end assertion in CI. It asserts that no "
        "component degraded, which is only true with the real encoder loaded, "
        "and it names the model in its output. Stubbing it would leave "
        "nothing checking that the documented entry point works."
    ),
    "test_storage.py::test_demo_supports_db_path_end_to_end": (
        "Runs demo.py in a subprocess, so the socket guard cannot reach it "
        "anyway. Marked so that it is counted rather than overlooked."
    ),
}

#: tiktoken fetches an encoding's BPE file the first time it is asked for it,
#: so a strict TokenCounter needs the network on a cold machine. These tests
#: are about what a real tokenizer does — that its count is not a function of
#: the whitespace word count, that it sees punctuation, that a literal control
#: token in a document is counted rather than rejected — and a stub would only
#: measure the stub. CI found them: they passed here on a warm tiktoken cache
#: and were refused on the runner, which is the fragility the guard exists for.
_TOKENIZER = (
    "Needs the real cl100k_base encoding. The property under test is that a "
    "tokenizer is not a word-count multiplier, which no stub can demonstrate."
)
MAY_USE_THE_NETWORK.update(
    {
        f"test_token_counter.py::{name}": _TOKENIZER
        for name in (
            "test_a_healthy_counter_is_not_degraded",
            "test_the_count_is_not_a_function_of_the_word_count",
            "test_punctuation_and_whitespace_are_counted",
            "test_an_empty_string_is_zero_tokens",
            "test_the_loaded_encoding_is_recorded_in_the_environment_report",
            "test_a_special_token_literal_in_the_text_is_counted_not_rejected",
        )
    }
)
MAY_USE_THE_NETWORK.update(
    {
        f"test_benchmark_token_counting.py::{name}": _TOKENIZER
        for name in (
            "test_every_arm_counts_with_the_same_object",
            "test_the_summary_names_the_counter_that_produced_its_figures",
        )
    }
)

#: Kept because it supplies an input the real router cannot be asked for.
#: Every other router stub in this suite answered the routing question the
#: orchestrator existed to ask, and has been replaced by the real QueryRouter.
ROUTER_STUBS_THAT_MAY_STAY = {
    "test_calibrated_confidence_and_abstention.py::FixedConfidenceRouter",
}


def _test_files():
    return sorted(p for p in TESTS.glob("test_*.py"))


def _marked_needs_download():
    found = set()
    for path in _test_files():
        for node in ast.walk(ast.parse(path.read_text())):
            if not isinstance(node, ast.FunctionDef):
                continue
            for decorator in node.decorator_list:
                if "needs_download" in ast.unparse(decorator):
                    found.add(f"{path.name}::{node.name}")
    return found


def test_the_network_marker_list_is_the_marker_list_in_the_files():
    """A test cannot grant itself network access without appearing here."""
    assert _marked_needs_download() == set(MAY_USE_THE_NETWORK)


def test_every_exemption_states_why():
    for name, reason in MAY_USE_THE_NETWORK.items():
        assert len(reason) > 40, f"{name} is exempt with no stated reason"


def test_an_unmarked_test_cannot_open_a_socket():
    """The guard is autouse, so it is already in force inside this test."""
    from tests.conftest import NetworkAccessInTest

    with pytest.raises(NetworkAccessInTest):
        socket.socket()

    with pytest.raises(NetworkAccessInTest):
        socket.create_connection(("example.invalid", 80))


def test_the_registry_is_cleared_around_every_test(pytester):
    """Run two tests for real and check the second cannot see the first's.

    Written with pytester rather than as a pair of tests in this file: a
    pair only demonstrates clearing if the dirtying one runs first, so it
    would make this file's order load-bearing — the very thing the fixture
    exists to remove. A mutation that deleted the clearing entirely passed a
    pair, because the "starts empty" half happened to run first.

    Fourteen files used to call clear_runtime_state themselves, and a test
    that forgot left its entries for whoever ran next.
    """
    # The real conftest, verbatim: the point is to exercise the fixture this
    # repository actually installs, not a re-statement of it.
    pytester.makeconftest((TESTS / "conftest.py").read_text())
    pytester.makepyfile(
        """
        from cke.diagnostics import environment_report, record_degradation

        def test_one_dirties_the_registry():
            record_degradation("LeakedFromTestOne", "a reason recorded here")
            assert environment_report().degradations

        def test_two_cannot_see_it():
            components = [r.component for r in environment_report().degradations]
            assert "LeakedFromTestOne" not in components
        """
    )
    # -p no:cacheprovider so the inner run does not write into the outer
    # run's cache directory.
    result = pytester.runpytest_subprocess("-p", "no:cacheprovider", "--no-cov")

    result.assert_outcomes(passed=2)


def test_the_model_caches_start_empty():
    from cke.retrieval import embedding_model

    assert embedding_model._GLOBAL_MODEL_CACHE == {}
    assert embedding_model._FAILED_MODEL_LOADS == {}


def _router_stubs():
    """Every class in tests/ that looks like a stand-in for the router."""
    found = set()
    for path in _test_files():
        for node in ast.walk(ast.parse(path.read_text())):
            if not isinstance(node, ast.ClassDef):
                continue
            methods = {
                child.name
                for child in node.body
                if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef))
            }
            if "route" in methods:
                found.add(f"{path.name}::{node.name}")
    return found


def test_no_test_hands_the_orchestrator_its_routing_answer():
    """Eight stub routers keyword-matched the fixtures they were tested with.

    One of them returned operator_hint="equality" for "same nationality" —
    the orchestrator's job was to work that out, and the test handed it over.
    The tests now build the real QueryRouter, which disagreed with the stubs
    in two places; both disagreements are recorded in the tests concerned
    rather than smoothed over.
    """
    assert _router_stubs() == ROUTER_STUBS_THAT_MAY_STAY


def test_pytest_is_configured_in_the_repository_and_not_only_in_ci():
    """A contributor running plain `pytest` got no coverage and no floor.

    The floor lived in one string in ci.yml, so the gate existed on the
    runner and nowhere anybody would meet it before pushing.
    """
    import tomllib

    with (REPO_ROOT / "pyproject.toml").open("rb") as handle:
        config = tomllib.load(handle)["tool"]["pytest"]["ini_options"]

    assert "--cov" in config["addopts"]
    assert "--strict-markers" in config["addopts"]
    assert "error" in config["filterwarnings"]
