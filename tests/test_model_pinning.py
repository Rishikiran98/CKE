"""A model name is not a model identity.

The Hub serves whatever ``main`` points at on the day of the run, so two runs
of the same command could embed with different weights and report the same
model name. Every loader here asks for a commit, refuses anything that is not
one, and records what it loaded so a results file can state it.
"""

from __future__ import annotations

import pathlib

import logging

import pytest

from cke.diagnostics import (
    DegradedComponentError,
    environment_report,
    revision_pin_problem,
)

#: A revision with the shape of a pin. Stubbed models have no Hub page.
_A_COMMIT = "a" * 40
_ANOTHER_COMMIT = "b" * 40


class _StubModel:
    """A loaded model that reports a dimension and embeds to zeros."""

    def __init__(self, name: str, revision: str | None = None) -> None:
        self.name = name
        self.revision = revision

    def get_sentence_embedding_dimension(self):
        return 4

    def encode(self, texts, **kwargs):
        import numpy as np

        return np.zeros((len(texts), 4), dtype="float32")


def _stub_loader(seen: list[tuple[str, str | None]]):
    def _load(name, revision=None):
        seen.append((name, revision))
        return _StubModel(name, revision)

    return _load


# ---------------------------------------------------------------------------
# What counts as a pin
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "revision",
    [
        None,
        "",
        "main",
        "v2.0",
        "1110a24",  # a prefix resolves to whatever the Hub matches it against
        "z" * 40,  # right length, not hexadecimal
    ],
)
def test_a_revision_that_is_not_a_full_commit_is_a_problem(revision):
    problem = revision_pin_problem("some/model", revision)

    assert problem is not None
    assert "some/model" in problem


def test_a_full_commit_hash_pins():
    assert revision_pin_problem("some/model", _A_COMMIT) is None
    assert revision_pin_problem("some/model", _A_COMMIT.upper()) is None


def test_the_two_pinned_loaders_share_one_check():
    """One statement of what a pin is, not one per loader that drifts."""
    from cke.evaluation import llm_qa
    from cke.retrieval import embedding_model

    for module in (llm_qa, embedding_model):
        source = (module.__file__ or "").replace(".pyc", ".py")
        text = pathlib.Path(source).read_text(encoding="utf-8")
        assert "[0-9a-fA-F]{40}" not in text, f"{module.__name__} re-states the rule"
        assert "revision_pin_problem" in text


# ---------------------------------------------------------------------------
# The embedder
# ---------------------------------------------------------------------------


def test_the_default_embedder_loads_at_a_pinned_revision(monkeypatch):
    from cke.retrieval import embedding_model as module

    seen: list[tuple[str, str | None]] = []
    monkeypatch.setattr(module, "SentenceTransformer", _stub_loader(seen))
    monkeypatch.setattr(module, "_GLOBAL_MODEL_CACHE", {})
    monkeypatch.setattr(module, "_FAILED_MODEL_LOADS", {})

    model = module.EmbeddingModel(strict=True)

    assert seen == [(module.DEFAULT_EMBEDDING_MODEL, module.DEFAULT_EMBEDDING_REVISION)]
    assert model.degraded is False


def test_an_unpinned_embedder_is_declared(monkeypatch, caplog):
    from cke.retrieval import embedding_model as module

    seen: list[tuple[str, str | None]] = []
    monkeypatch.setattr(module, "SentenceTransformer", _stub_loader(seen))
    monkeypatch.setattr(module, "_GLOBAL_MODEL_CACHE", {})
    monkeypatch.setattr(module, "_FAILED_MODEL_LOADS", {})

    with caplog.at_level(logging.WARNING):
        model = module.EmbeddingModel(model_name="someone/other-embedder")

    assert model.degraded is True
    assert "reproducible" in model.degraded_reason
    assert seen == [], "an unpinned model must not be loaded anyway"
    assert "someone/other-embedder" in caplog.text


def test_an_unpinned_embedder_stops_a_strict_run(monkeypatch):
    from cke.retrieval import embedding_model as module

    monkeypatch.setattr(module, "SentenceTransformer", _stub_loader([]))
    monkeypatch.setattr(module, "_GLOBAL_MODEL_CACHE", {})
    monkeypatch.setattr(module, "_FAILED_MODEL_LOADS", {})

    with pytest.raises(DegradedComponentError, match="revision"):
        module.EmbeddingModel(model_name="someone/other-embedder", strict=True)


def test_a_moving_revision_is_declared(monkeypatch):
    """ "main" names whatever was pushed last, so it is not a pin."""
    from cke.retrieval import embedding_model as module

    monkeypatch.setattr(module, "SentenceTransformer", _stub_loader([]))
    monkeypatch.setattr(module, "_GLOBAL_MODEL_CACHE", {})
    monkeypatch.setattr(module, "_FAILED_MODEL_LOADS", {})

    with pytest.raises(DegradedComponentError, match="commit"):
        module.EmbeddingModel(
            model_name="someone/other-embedder",
            model_revision="main",
            strict=True,
        )


def test_the_cache_does_not_serve_one_revision_for_another(monkeypatch):
    """Keyed on the name alone, the second caller silently got the first's
    weights while reporting the commit it had asked for."""
    from cke.retrieval import embedding_model as module

    seen: list[tuple[str, str | None]] = []
    monkeypatch.setattr(module, "SentenceTransformer", _stub_loader(seen))
    monkeypatch.setattr(module, "_GLOBAL_MODEL_CACHE", {})
    monkeypatch.setattr(module, "_FAILED_MODEL_LOADS", {})

    first = module.EmbeddingModel("some/model", _A_COMMIT, strict=True)
    second = module.EmbeddingModel("some/model", _ANOTHER_COMMIT, strict=True)
    again = module.EmbeddingModel("some/model", _A_COMMIT, strict=True)

    assert first.model.revision == _A_COMMIT
    assert second.model.revision == _ANOTHER_COMMIT
    assert again.model is first.model, "the cache must still spare a reload"
    assert seen == [("some/model", _A_COMMIT), ("some/model", _ANOTHER_COMMIT)]


def test_the_embedder_records_the_commit_it_loaded(monkeypatch):
    from cke.retrieval import embedding_model as module

    monkeypatch.setattr(module, "SentenceTransformer", _stub_loader([]))
    monkeypatch.setattr(module, "_GLOBAL_MODEL_CACHE", {})
    monkeypatch.setattr(module, "_FAILED_MODEL_LOADS", {})

    module.EmbeddingModel("some/model", _A_COMMIT, strict=True)

    loaded = {
        model.component: model.loaded for model in environment_report().loaded_models
    }
    assert loaded["EmbeddingModel"] == f"some/model@{_A_COMMIT}"


# ---------------------------------------------------------------------------
# The second loader: entity resolution embeds too
# ---------------------------------------------------------------------------


def test_the_resolver_loads_the_same_model_at_the_same_commit(monkeypatch):
    """It named the model without its organisation and without a revision, so
    a run embedded through two separately-resolved copies of the weights."""
    from cke.entity_resolution import entity_resolver as module
    from cke.retrieval import embedding_model as embed_module

    seen: list[tuple[str, str | None]] = []
    monkeypatch.setattr(module, "SentenceTransformer", _stub_loader(seen))
    monkeypatch.setattr(module, "_MODEL_CACHE", {})
    monkeypatch.setattr(module, "_FAILED_MODEL_LOADS", {})

    resolver = module.EntityResolver(strict=True)
    resolver._load_embedding_model()

    assert seen == [
        (embed_module.DEFAULT_EMBEDDING_MODEL, embed_module.DEFAULT_EMBEDDING_REVISION)
    ]
    loaded = {
        model.component: model.loaded for model in environment_report().loaded_models
    }
    assert loaded["EntityResolver"] == (
        f"{embed_module.DEFAULT_EMBEDDING_MODEL}@"
        f"{embed_module.DEFAULT_EMBEDDING_REVISION}"
    )


def test_the_pinned_revision_is_stated_once(monkeypatch):
    """Two copies of a commit hash drift, and a drifted pin is not a pin."""
    from cke.entity_resolution import entity_resolver as module
    from cke.retrieval import embedding_model as embed_module

    assert module._EMBEDDING_MODEL_REVISION is embed_module.DEFAULT_EMBEDDING_REVISION
    assert module._EMBEDDING_MODEL_NAME is embed_module.DEFAULT_EMBEDDING_MODEL
