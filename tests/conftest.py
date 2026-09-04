"""Guards that apply to every test. There was no conftest.py before this one.

Three things were unguarded, and the absence of this file is why.

**The network.** Nothing prevented a test reaching out. Two of them load a
real sentence-transformer, and they pass locally only because
``~/.cache/huggingface`` is warm; on a cold runner they fetch about 90 MB, and
an upstream outage turns the suite red for reasons that have nothing to do
with the code. A test that legitimately needs a model must now say so with
``@pytest.mark.needs_model``, and ``tests/test_test_hygiene.py`` holds that
list to a named few. Everything else is refused at the socket.

**The degradation registry.** ``cke.diagnostics`` keeps a process-wide record
of what degraded and which models loaded. Tests that read it were cleaning up
by hand — fourteen files called ``clear_runtime_state`` at various points —
and a test that forgot left its entries for whoever ran next, which made the
order of the suite load-bearing. It is cleared around every test now.

**The embedding model caches.** ``cke.retrieval.embedding_model`` keeps two
module-level dicts, one of loaded models and one of failed loads, both keyed
by name and revision. A test that provokes a failure leaves the reason in the
second, and a later test asking for the same model gets the recorded failure
rather than attempting its own load. Cleared around every test too, except
for the tests that want a real model — for those, the cache is what stops a
second download.
"""

from __future__ import annotations

import socket
import zlib

import pytest

from cke.diagnostics import clear_runtime_state

#: Enables the `pytester` fixture, which runs a real pytest in a
#: subprocess. tests/test_test_hygiene.py uses it to show that the
#: clearing below happens between tests, without making the order of
#: this suite the evidence.
pytest_plugins = ["pytester"]


class NetworkAccessInTest(RuntimeError):
    """Raised when an unmarked test tries to open a socket."""


def pytest_configure(config):
    config.addinivalue_line(
        "markers",
        "needs_model: this test loads a real model and may use the network. "
        "Marked tests are listed in tests/test_test_hygiene.py.",
    )


@pytest.fixture(autouse=True)
def _clean_runtime_state(request):
    """Clear process-global state around every test.

    Around, not merely after: a test that reads the registry must not see
    what an earlier one recorded, and the next test must not see this one's.
    """
    clear_runtime_state()
    yield
    clear_runtime_state()


@pytest.fixture(autouse=True)
def _clean_model_caches(request):
    from cke.retrieval import embedding_model

    if request.node.get_closest_marker("needs_model"):
        # Emptying the cache here would make each marked test download again.
        yield
        return

    embedding_model._GLOBAL_MODEL_CACHE.clear()
    embedding_model._FAILED_MODEL_LOADS.clear()
    yield
    embedding_model._GLOBAL_MODEL_CACHE.clear()
    embedding_model._FAILED_MODEL_LOADS.clear()


@pytest.fixture(autouse=True)
def _no_network(request, monkeypatch):
    """Refuse to open a socket unless the test declared that it needs one.

    This cannot reach a subprocess, so a test that shells out is on its
    honour; those are marked too, and named in the hygiene test.
    """
    if request.node.get_closest_marker("needs_model"):
        return

    def _refuse(*args, **kwargs):
        raise NetworkAccessInTest(
            "this test opened a network connection. A test that depends on a "
            "download is red whenever the other end is, and green only on a "
            "machine whose cache is already warm. If it genuinely needs a "
            "real model, mark it @pytest.mark.needs_model and add it to "
            "tests/test_test_hygiene.py; otherwise stub the component."
        )

    monkeypatch.setattr(socket, "socket", _refuse)
    monkeypatch.setattr(socket, "create_connection", _refuse)


class StubSentenceTransformer:
    """A loadable stand-in for the real encoder.

    It hashes, so its vectors are no more semantic than the fallback's. The
    one property that matters is that constructing it succeeds, which is what
    lets a strict component exist without a download: ``EmbeddingModel``
    refuses to degrade under ``strict=True``, and a model that will not load
    is a degradation, so before this every strict construction in the suite
    reached the network.

    Nothing that uses it asserts on retrieval quality. A test that did would
    be measuring this class.
    """

    #: Narrow, and not the real model's width. A test that happened to depend
    #: on 384 would be depending on the real encoder while claiming not to.
    DIMENSION = 16

    def __init__(self, model_name, revision=None, **kwargs):
        self.model_name = model_name
        self.revision = revision

    def get_sentence_embedding_dimension(self):
        return self.DIMENSION

    def encode(self, texts, **kwargs):
        import numpy as np

        if isinstance(texts, str):
            texts = [texts]
        vectors = []
        for text in texts:
            vector = [0.0] * self.DIMENSION
            for token in str(text).lower().split():
                # crc32, not hash(): str.__hash__ is salted per process, so a
                # stub built on it embeds the same corpus differently on every
                # run and any test that compared two runs would flap.
                bucket = zlib.crc32(token.encode("utf-8")) % self.DIMENSION
                vector[bucket] += 1.0
            vectors.append(vector)
        return np.asarray(vectors, dtype=np.float32)


@pytest.fixture
def offline_embedder(monkeypatch):
    """Make a strict component constructible without reaching the Hub.

    Request this rather than marking a test ``needs_model`` whenever the test
    is about strictness, plumbing or graph reasoning and the encoder is only
    in the way.
    """
    from cke.entity_resolution import entity_resolver
    from cke.retrieval import embedding_model

    monkeypatch.setattr(embedding_model, "SentenceTransformer", StubSentenceTransformer)
    monkeypatch.setattr(entity_resolver, "SentenceTransformer", StubSentenceTransformer)
    embedding_model._GLOBAL_MODEL_CACHE.clear()
    embedding_model._FAILED_MODEL_LOADS.clear()
    yield StubSentenceTransformer
    embedding_model._GLOBAL_MODEL_CACHE.clear()
    embedding_model._FAILED_MODEL_LOADS.clear()
