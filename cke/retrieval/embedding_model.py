"""Embedding model wrapper with global model caching.

Without sentence-transformers this class can only hash tokens into a sparse
vector, which is not a semantic embedding. A benchmark that ran on that
fallback measured a hash function, so the fallback is never silent: it warns,
sets ``degraded``, and refuses to engage at all under ``strict=True``.
"""

from __future__ import annotations

import hashlib
import logging
from typing import Iterable

import numpy as np

from cke.diagnostics import (
    DegradationMixin,
    record_loaded_model,
    revision_pin_problem,
)

try:
    from sentence_transformers import SentenceTransformer
except ImportError:  # pragma: no cover - optional runtime dependency
    SentenceTransformer = None


logger = logging.getLogger(__name__)

DEFAULT_EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

#: The Hub commit the default embedder is pinned to. Without it the Hub
#: serves whatever "main" points at on the day of the run, and two runs of
#: the same command could embed with different weights while reporting the
#: same model name. This is the commit the vectors behind every figure in
#: this repository were produced with.
DEFAULT_EMBEDDING_REVISION = "1110a243fdf4706b3f48f1d95db1a4f5529b4d41"

#: Loaded models, keyed by name *and* revision. Keying on the name alone
#: would hand a caller that asked for one commit the weights of another.
_GLOBAL_MODEL_CACHE: dict[tuple[str, str], object] = {}

#: Why a model failed to load, by name and revision. A failed load is never
#: cached as a usable model, but it is remembered so that constructing many
#: instances does not re-attempt a download per instance. The recorded cause is
#: reused, so a later strict construction still raises for the original reason.
_FAILED_MODEL_LOADS: dict[tuple[str, str], str] = {}

#: Width of the hashed fallback vector. Not a semantic embedding dimension.
FALLBACK_DIM = 128


class EmbeddingModel(DegradationMixin):
    """Sentence-transformer embedding wrapper with batch support.

    Args:
        model_name: sentence-transformers model identifier.
        model_revision: the 40-character Hub commit to load. Defaults to
            :data:`DEFAULT_EMBEDDING_REVISION` for the default model; for any
            other model it must be given, because a name on its own does not
            say which weights will arrive.
        strict: when True, raise :class:`~cke.diagnostics.DegradedComponentError`
            rather than falling back to hashed vectors. Every evaluation and
            benchmark path must pass ``strict=True``.
    """

    def __init__(
        self,
        model_name: str = DEFAULT_EMBEDDING_MODEL,
        model_revision: str | None = None,
        strict: bool = False,
    ) -> None:
        self._init_degradation(strict)
        self.model_name = model_name
        if model_revision is None and model_name == DEFAULT_EMBEDDING_MODEL:
            model_revision = DEFAULT_EMBEDDING_REVISION
        self.model_revision = model_revision
        self._measured_dimension: int | None = None
        self.model = self._load_model(model_name, model_revision)

    @property
    def dimension(self) -> int:
        """Width of the vectors this instance produces."""
        if self.model is None:
            return FALLBACK_DIM

        try:
            dim = self.model.get_sentence_embedding_dimension()
        except AttributeError:  # pragma: no cover - non-standard model object
            dim = None

        if dim:
            return int(dim)

        # The model is real but does not report a dimension. Measure it rather
        # than reporting the hashed fallback's width for a healthy model.
        if self._measured_dimension is None:
            probe = self.model.encode(
                [""], convert_to_numpy=True, show_progress_bar=False
            )
            self._measured_dimension = int(np.asarray(probe).shape[-1])
        return self._measured_dimension

    def embed_text(self, text: str) -> np.ndarray:
        """Embed a single text and return a float32 vector."""
        return self.embed_texts([text])[0]

    def embed_texts(self, texts: Iterable[str], batch_size: int = 32) -> np.ndarray:
        """Embed many texts efficiently in batches."""
        text_list = list(texts)
        if not text_list:
            return np.zeros((0, self.dimension), dtype=np.float32)

        if self.model is not None:
            vectors = self.model.encode(
                text_list,
                batch_size=batch_size,
                convert_to_numpy=True,
                show_progress_bar=False,
                normalize_embeddings=False,
            )
            return np.asarray(vectors, dtype=np.float32)

        return np.asarray(
            [self._fallback_embed(t) for t in text_list], dtype=np.float32
        )

    def _load_model(self, model_name: str, model_revision: str | None):
        """Load the sentence transformer, or declare the degradation."""
        pin_problem = revision_pin_problem(model_name, model_revision)
        if pin_problem is not None:
            # Asked before the cache, so an unpinned request is refused rather
            # than served whatever commit a previous caller happened to load.
            self._degrade(pin_problem)
            return None

        key = (model_name, model_revision)
        cached = _GLOBAL_MODEL_CACHE.get(key)
        if cached is not None:
            return cached

        previous_failure = _FAILED_MODEL_LOADS.get(key)
        if previous_failure is not None:
            self._degrade(previous_failure)
            return None

        if SentenceTransformer is None:
            self._degrade(
                "sentence-transformers is not installed, so text is embedded by "
                f"hashing tokens into a {FALLBACK_DIM}-dimension sparse vector. "
                "That is a hash function, not a semantic embedding, and any "
                "retrieval quality measured against it is meaningless. "
                "Install it with `pip install sentence-transformers`"
            )
            return None

        try:
            model = SentenceTransformer(model_name, revision=model_revision)
        except Exception as exc:  # noqa: BLE001 - download/runtime failures vary
            # A failed load is never cached as a usable model. Its cause is
            # remembered instead, so a caller that builds one instance per
            # item does not re-attempt the load every time, while a later
            # strict construction still raises naming the original failure.
            reason = (
                f"sentence-transformers could not load {model_name!r} "
                f"({type(exc).__name__}: {exc}), so text is embedded by hashing "
                f"tokens into a {FALLBACK_DIM}-dimension sparse vector instead "
                "of being embedded semantically"
            )
            _FAILED_MODEL_LOADS[key] = reason
            self._degrade(reason)
            return None

        loaded = f"{model_name}@{model_revision}"
        _GLOBAL_MODEL_CACHE[key] = model
        record_loaded_model("EmbeddingModel", model_name, loaded)
        logger.info("EmbeddingModel loaded %s", loaded)
        return model

    @staticmethod
    def _fallback_embed(text: str, dim: int = FALLBACK_DIM) -> np.ndarray:
        """Hash tokens into a sparse vector. NOT a semantic embedding."""
        vector = np.zeros((dim,), dtype=np.float32)
        for token in text.lower().split():
            digest = hashlib.sha256(token.encode("utf-8")).hexdigest()
            idx = int(digest, 16) % dim
            vector[idx] += 1.0
        norm = float(np.linalg.norm(vector))
        if norm > 0:
            vector /= norm
        return vector
