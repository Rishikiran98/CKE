"""Tests for the token counter.

Two things are guarded. That the counts come from a tokenizer rather than a
multiplier — the difference is observable, because a multiplier is a function
of the whitespace word count alone and a tokenizer is not. And that a missing
tokenizer is declared, never silently restored to the multiplier it replaced.
"""

from __future__ import annotations

import logging
import sys

import pytest

from cke.diagnostics import DegradedComponentError
from cke.evaluation.token_counter import TokenCounter


@pytest.fixture()
def counter() -> TokenCounter:
    """A working counter.

    Constructed strict on purpose: if the tokenizer cannot be loaded, these
    tests fail loudly rather than passing against the estimate they exist to
    rule out.

    tiktoken fetches the BPE file for an encoding the first time it is asked
    for it, so every test taking this fixture is marked needs_download. They
    are about what a real tokenizer does — that its count is not a function
    of the word count, that it sees punctuation — and a stub would only
    measure the stub.
    """
    return TokenCounter(strict=True)


@pytest.mark.needs_download
def test_a_healthy_counter_is_not_degraded(counter):
    assert counter.degraded is False
    assert counter.is_estimate is False
    assert counter.description == "tiktoken cl100k_base"


@pytest.mark.needs_download
def test_the_count_is_not_a_function_of_the_word_count(counter):
    """The property that separates a tokenizer from a multiplier.

    Both strings hold three whitespace-delimited words, so the multiplier this
    class replaced returned 3 for each. A tokenizer splits the rare words into
    several tokens and the common ones into one apiece.
    """
    common = "the cat sat"
    rare = (
        "pneumonoultramicroscopic antidisestablishmentarian "
        "floccinaucinihilipilification"
    )

    assert len(common.split()) == len(rare.split()) == 3
    assert counter.count(rare) > counter.count(common)


@pytest.mark.needs_download
def test_punctuation_and_whitespace_are_counted(counter):
    """A word count cannot see either."""
    assert counter.count("a,b,c,d,e,f") > 1
    assert counter.count("hello") >= 1


@pytest.mark.needs_download
def test_an_empty_string_is_zero_tokens(counter):
    assert counter.count("") == 0


def test_a_missing_tokenizer_is_declared(monkeypatch, caplog):
    """All three obligations on the degraded path."""
    monkeypatch.setitem(sys.modules, "tiktoken", None)

    with caplog.at_level(logging.WARNING):
        degraded = TokenCounter()

    assert degraded.degraded is True
    assert degraded.is_estimate is True
    assert "tiktoken" in degraded.degraded_reason
    assert "tiktoken" in caplog.text
    assert "ESTIMATE" in degraded.description


def test_a_missing_tokenizer_stops_a_strict_run(monkeypatch):
    monkeypatch.setitem(sys.modules, "tiktoken", None)

    with pytest.raises(DegradedComponentError, match="tiktoken"):
        TokenCounter(strict=True)


def test_an_unloadable_encoding_is_declared(caplog):
    """The encoding is fetched, so it can fail for reasons beyond the import."""
    with caplog.at_level(logging.WARNING):
        degraded = TokenCounter(encoding="no_such_encoding_xyz")

    assert degraded.degraded is True
    assert degraded.is_estimate is True
    assert "no_such_encoding_xyz" in degraded.degraded_reason


def test_an_unloadable_encoding_stops_a_strict_run():
    with pytest.raises(DegradedComponentError, match="no_such_encoding_xyz"):
        TokenCounter(encoding="no_such_encoding_xyz", strict=True)


def test_a_degraded_counter_still_returns_a_number(monkeypatch):
    """It has to; what it must not do is hide which number it is.

    The description carries the estimate marker so that a figure produced on
    this path cannot be read as a measurement.
    """
    monkeypatch.setitem(sys.modules, "tiktoken", None)
    degraded = TokenCounter()

    assert degraded.count("one two three four five") > 0
    assert "word count" in degraded.description


@pytest.mark.needs_download
def test_the_loaded_encoding_is_recorded_in_the_environment_report(counter):
    from cke.diagnostics import environment_report

    loaded = environment_report().loaded_models
    assert any(model.component == "TokenCounter" for model in loaded)


@pytest.mark.needs_download
def test_a_special_token_literal_in_the_text_is_counted_not_rejected(counter):
    """Dataset text is text, including text that looks like a control token.

    tiktoken's encode() rejects a literal such as "<|endoftext|>" by default.
    This counter is only ever asked to measure text, never to build a prompt,
    so such a literal is a string a document happened to contain — and one
    document containing it must not abort a whole benchmark run.
    """
    plain = "A document mentioning nothing odd in its prose."
    special = "A document mentioning <|endoftext|> in its prose."

    assert counter.count(special) > counter.count(plain)
