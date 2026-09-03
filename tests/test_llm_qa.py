"""Tests for the LLM answerer.

Nothing here downloads a model or calls a network. The local backend is
exercised through a stub tokenizer and model set directly on the instance,
which is the only way to test truncation accounting and prompt use offline;
the contract paths — no key, an unloadable model — are what a strict run
must refuse on, and are tested as such.
"""

from __future__ import annotations

import logging

import pytest

from cke.diagnostics import DegradedComponentError, clear_runtime_state
from cke.evaluation.llm_qa import PROMPT, LLMAnswerer


@pytest.fixture(autouse=True)
def _clean(monkeypatch):
    clear_runtime_state()
    monkeypatch.delenv("CKE_LLM_API_KEY", raising=False)
    yield
    clear_runtime_state()


class _StubTokenizer:
    """Counts whitespace tokens; a 'window' is enforced by the answerer."""

    model_max_length = 8

    def __call__(self, text, return_tensors=None, truncation=False, max_length=None):
        tokens = text.split()
        if truncation and max_length is not None:
            tokens = tokens[:max_length]
        self.last_prompt = text
        self.last_length = len(tokens)
        return {"input_ids": _Shape(len(tokens))}

    def decode(self, ids, skip_special_tokens=True):
        return "  stub answer  "


class _Shape:
    def __init__(self, n):
        self.shape = (1, n)


class _StubModel:
    def generate(self, input_ids=None, max_new_tokens=None, do_sample=None, **_):
        self.max_new_tokens = max_new_tokens
        self.do_sample = do_sample
        return [[0]]


def _local_with_stubs(window=None) -> LLMAnswerer:
    """A local answerer whose model is a stub, bypassing any download."""
    answerer = LLMAnswerer.__new__(LLMAnswerer)
    answerer._init_degradation(False)
    answerer.backend = "local"
    answerer.model_name = "stub"
    answerer.truncation = type(answerer).__init__.__globals__["TruncationLog"]()
    answerer._tokenizer = _StubTokenizer()
    answerer._model = _StubModel()
    answerer._requested_window = window
    answerer._window = window or _StubTokenizer.model_max_length
    return answerer


# ----------------------------------------------------------------------
# The contract: no model, no number
# ----------------------------------------------------------------------


def test_api_backend_without_a_key_declares_the_degradation(caplog):
    with caplog.at_level(logging.WARNING):
        answerer = LLMAnswerer(backend="api")

    assert answerer.degraded is True
    assert answerer.available is False
    assert "CKE_LLM_API_KEY" in answerer.degraded_reason
    assert "CKE_LLM_API_KEY" in caplog.text
    assert answerer.description.startswith("NO MODEL")


def test_api_backend_without_a_key_stops_a_strict_run():
    with pytest.raises(DegradedComponentError, match="CKE_LLM_API_KEY"):
        LLMAnswerer(backend="api", strict=True)


def test_an_unavailable_answerer_refuses_to_answer():
    """It must not fall back to anything, span baseline included."""
    answerer = LLMAnswerer(backend="api")

    with pytest.raises(RuntimeError, match="no model"):
        answerer.answer("Who founded SpaceX?", "SpaceX was founded by Elon Musk.")


def test_an_unknown_backend_is_rejected():
    with pytest.raises(ValueError):
        LLMAnswerer(backend="telepathy")


# ----------------------------------------------------------------------
# Symmetry: one prompt, one decoding, whatever the context
# ----------------------------------------------------------------------


def test_the_prompt_is_used_verbatim_with_the_context_as_given():
    answerer = _local_with_stubs()
    answerer.answer("Who founded SpaceX?", "SpaceX uses RESP\nRedis uses RESP")

    expected = PROMPT.format(
        context="SpaceX uses RESP\nRedis uses RESP", question="Who founded SpaceX?"
    )
    assert answerer._tokenizer.last_prompt == expected


def test_decoding_is_greedy_and_short():
    """Greedy so two arms with the same context get the same answer; short so
    a generation cannot become the sentence prefix the span answerer replaced."""
    answerer = _local_with_stubs()
    answerer.answer("q", "c")

    assert answerer._model.do_sample is False
    assert answerer._model.max_new_tokens == 16


def test_the_answer_is_stripped():
    assert _local_with_stubs().answer("q", "c") == "stub answer"


# ----------------------------------------------------------------------
# Truncation is measured
# ----------------------------------------------------------------------


def test_a_prompt_within_the_window_is_not_counted_as_truncated():
    answerer = _local_with_stubs(window=64)
    answerer.answer("q", "short context")

    assert answerer.truncation.calls == 1
    assert answerer.truncation.truncated == 0


def test_a_prompt_beyond_the_window_is_truncated_and_counted():
    answerer = _local_with_stubs(window=8)
    answerer.answer("q", " ".join(["word"] * 40))

    assert answerer.truncation.calls == 1
    assert answerer.truncation.truncated == 1
    assert answerer.truncation.dropped_tokens[0] > 0
    assert answerer._tokenizer.last_length == 8
    assert answerer.truncation.rate == 1.0


def _patch_loaders(monkeypatch):
    """Route construction through the real _load_local with stub loaders.

    The stub helper above sets the window directly and so cannot see whether
    the loader honours a requested one; a mutation ignoring it survived that
    test. These go through the loader.
    """
    import transformers

    monkeypatch.setattr(
        transformers.AutoTokenizer, "from_pretrained", lambda name: _StubTokenizer()
    )
    monkeypatch.setattr(
        transformers.AutoModelForSeq2SeqLM,
        "from_pretrained",
        lambda name: type(
            "M", (), {"eval": lambda self: None, "generate": _StubModel().generate}
        )(),
    )


def test_the_native_window_is_used_when_none_is_requested(monkeypatch):
    _patch_loaders(monkeypatch)
    answerer = LLMAnswerer(backend="local", model="stub")

    assert answerer._window == _StubTokenizer.model_max_length
    assert f"{_StubTokenizer.model_max_length}-token window" in answerer.description


def test_the_requested_window_overrides_the_native_one(monkeypatch):
    _patch_loaders(monkeypatch)
    answerer = LLMAnswerer(backend="local", model="stub", max_input_tokens=200)
    answerer.answer("q", " ".join(["word"] * 40))

    assert answerer._window == 200
    assert answerer.truncation.truncated == 0
    assert "200-token window" in answerer.description
