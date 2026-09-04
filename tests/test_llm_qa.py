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


#: A revision with the shape of a pin. The answerer refuses anything that is
#: not a full commit hash, so a placeholder has to look like one.
_A_COMMIT = "0123456789abcdef0123456789abcdef01234567"


@pytest.fixture(autouse=True)
def _clean(monkeypatch):
    clear_runtime_state()
    monkeypatch.delenv("CKE_LLM_API_KEY", raising=False)
    yield
    clear_runtime_state()


class _Ids:
    """A tokenised text: shape like a tensor, indexable like one."""

    def __init__(self, tokens):
        self.tokens = tokens
        self.shape = (1, len(tokens))

    def __getitem__(self, index):
        return self.tokens


class _StubTokenizer:
    """Counts whitespace tokens; a 'window' is enforced by the answerer.

    Decoding a token list gives the words back, so a truncated context comes
    back as its kept words; decoding the model's output gives a fixed answer.
    """

    model_max_length = 8

    def __call__(self, text, return_tensors=None, truncation=False, max_length=None):
        tokens = text.split()
        if truncation and max_length is not None:
            tokens = tokens[:max_length]
        self.last_prompt = text
        self.last_length = len(tokens)
        return {"input_ids": _Ids(tokens)}

    def decode(self, ids, skip_special_tokens=True):
        if isinstance(ids, list) and ids and isinstance(ids[0], str):
            return " ".join(ids)
        return "  stub answer  "


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
    # A window the whole prompt fits in: truncation is tested separately.
    answerer = _local_with_stubs(window=_scaffold_words("Who founded SpaceX?") + 100)
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


def _scaffold_words(question="q"):
    return len(PROMPT.format(context="", question=question).split())


def test_a_context_beyond_the_window_is_truncated_and_counted():
    window = _scaffold_words() + 10
    answerer = _local_with_stubs(window=window)
    answerer.answer("q", " ".join(["word"] * 40))

    assert answerer.truncation.calls == 1
    assert answerer.truncation.truncated == 1
    assert answerer.truncation.dropped_tokens == [30]
    assert answerer.truncation.rate == 1.0
    assert answerer.last_truncated is True
    assert answerer.last_dropped_tokens == 30


def test_truncation_cuts_the_context_and_never_the_question():
    """The defect: the prompt puts the question after the context, and
    right-side truncation of the assembled prompt threw the question away.
    The model then answered "Who founded SpaceX?" with a sentence about Redis,
    because it had been shown no question at all.
    """
    window = _scaffold_words("Who founded SpaceX?") + 10
    answerer = _local_with_stubs(window=window)
    context = " ".join(f"w{i}" for i in range(40))
    answerer.answer("Who founded SpaceX?", context)

    prompt = answerer._tokenizer.last_prompt
    assert "Question: Who founded SpaceX?" in prompt
    assert prompt.rstrip().endswith("Answer:")
    # Only the context's tail is gone: the first ten words survive, none after.
    assert "w9" in prompt and "w10" not in prompt
    assert len(prompt.split()) <= window


def test_last_truncation_is_reset_on_a_call_that_fits():
    window = _scaffold_words() + 10
    answerer = _local_with_stubs(window=window)
    answerer.answer("q", " ".join(["word"] * 40))
    answerer.answer("q", "short")

    assert answerer.last_truncated is False
    assert answerer.last_dropped_tokens == 0


def _patch_loaders(monkeypatch, seen=None):
    """Route construction through the real _load_local with stub loaders.

    The stub helper above sets the window directly and so cannot see whether
    the loader honours a requested one; a mutation ignoring it survived that
    test. These go through the loader. ``seen`` collects the keyword
    arguments each loader was called with.
    """
    import transformers

    def tok(name, **kw):
        if seen is not None:
            seen.append(("tokenizer", name, kw))
        return _StubTokenizer()

    def mdl(name, **kw):
        if seen is not None:
            seen.append(("model", name, kw))
        return type(
            "M", (), {"eval": lambda self: None, "generate": _StubModel().generate}
        )()

    monkeypatch.setattr(transformers.AutoTokenizer, "from_pretrained", tok)
    monkeypatch.setattr(transformers.AutoModelForSeq2SeqLM, "from_pretrained", mdl)


def test_the_native_window_is_used_when_none_is_requested(monkeypatch):
    _patch_loaders(monkeypatch)
    answerer = LLMAnswerer(backend="local", model="stub", model_revision=_A_COMMIT)

    assert answerer._window == _StubTokenizer.model_max_length
    assert f"{_StubTokenizer.model_max_length}-token window" in answerer.description


def test_the_requested_window_overrides_the_native_one(monkeypatch):
    _patch_loaders(monkeypatch)
    answerer = LLMAnswerer(
        backend="local", model="stub", max_input_tokens=200, model_revision=_A_COMMIT
    )
    answerer.answer("q", " ".join(["word"] * 40))

    assert answerer._window == 200
    assert answerer.truncation.truncated == 0
    assert "200-token window" in answerer.description


# ----------------------------------------------------------------------
# The weights are pinned
# ----------------------------------------------------------------------


def test_the_default_model_loads_at_a_pinned_revision(monkeypatch):
    """A model name is not a model. Both loaders must be told the commit."""
    from cke.evaluation.llm_qa import _DEFAULT_LOCAL_REVISION

    seen = []
    _patch_loaders(monkeypatch, seen)
    answerer = LLMAnswerer(backend="local")

    assert [kw.get("revision") for _, _, kw in seen] == [_DEFAULT_LOCAL_REVISION] * 2
    assert _DEFAULT_LOCAL_REVISION[:12] in answerer.description


def test_a_model_without_a_pinned_revision_is_declared(monkeypatch, caplog):
    _patch_loaders(monkeypatch)
    with caplog.at_level(logging.WARNING):
        answerer = LLMAnswerer(backend="local", model="someone/other-model")

    assert answerer.degraded is True
    assert answerer.available is False
    assert "revision" in answerer.degraded_reason


def test_a_model_without_a_pinned_revision_stops_a_strict_run(monkeypatch):
    _patch_loaders(monkeypatch)
    with pytest.raises(DegradedComponentError, match="revision"):
        LLMAnswerer(backend="local", model="someone/other-model", strict=True)


def test_an_explicit_revision_reaches_both_loaders(monkeypatch):
    seen = []
    _patch_loaders(monkeypatch, seen)
    LLMAnswerer(backend="local", model="someone/other-model", model_revision=_A_COMMIT)

    assert [kw.get("revision") for _, _, kw in seen] == [_A_COMMIT, _A_COMMIT]


@pytest.mark.parametrize(
    "revision",
    ["main", "v1.0", "deadbeef", _A_COMMIT[:12], _A_COMMIT + "0"],
    ids=["branch", "tag", "word", "short-hash", "too-long"],
)
def test_a_revision_that_is_not_a_full_commit_is_declared(
    monkeypatch, caplog, revision
):
    """A branch or tag moves and a short hash is a prefix; none of them pins.

    Truthiness let ``main`` through, and a run given it would have loaded
    whatever the Hub served that day while reporting itself pinned. Nothing
    may be loaded through such a name, so the loaders must not be called.
    """
    seen = []
    _patch_loaders(monkeypatch, seen)
    with caplog.at_level(logging.WARNING):
        answerer = LLMAnswerer(
            backend="local", model="someone/other-model", model_revision=revision
        )

    assert answerer.degraded is True
    assert answerer.available is False
    assert revision in answerer.degraded_reason
    assert "commit" in answerer.degraded_reason
    assert seen == []
    assert any("commit" in record.message for record in caplog.records)


def test_a_branch_name_stops_a_strict_run(monkeypatch):
    _patch_loaders(monkeypatch)
    with pytest.raises(DegradedComponentError, match="main"):
        LLMAnswerer(
            backend="local",
            model="someone/other-model",
            model_revision="main",
            strict=True,
        )


# ----------------------------------------------------------------------
# The api backend cannot measure what it cut
# ----------------------------------------------------------------------


def test_a_window_requested_for_the_api_backend_is_declared(caplog):
    """It was accepted and ignored: the context went whole either way."""
    with caplog.at_level(logging.WARNING):
        answerer = LLMAnswerer(backend="api", api_key="k", max_input_tokens=50)

    assert answerer.degraded is True
    assert "window" in answerer.degraded_reason
    assert "tokeniser" in answerer.degraded_reason
    assert any("window" in record.message for record in caplog.records)


def test_a_window_requested_for_the_api_backend_stops_a_strict_run():
    with pytest.raises(DegradedComponentError, match="window"):
        LLMAnswerer(backend="api", api_key="k", max_input_tokens=50, strict=True)


def test_the_api_backend_does_not_claim_to_measure_truncation():
    """Its truncated count stays zero whatever the endpoint did with the
    prompt, so reporting a rate would state a measurement never taken."""
    answerer = LLMAnswerer(backend="api", api_key="k")

    assert answerer.truncation_measured is False


def test_the_local_backend_measures_truncation(monkeypatch):
    _patch_loaders(monkeypatch)

    answerer = LLMAnswerer(backend="local", model="stub", model_revision=_A_COMMIT)

    assert answerer.truncation_measured is True


def test_both_answerers_say_whether_a_model_read_the_context():
    from cke.evaluation.span_qa import SpanExtractiveQA

    assert LLMAnswerer(backend="api", api_key="k").uses_language_model is True
    assert SpanExtractiveQA().uses_language_model is False
