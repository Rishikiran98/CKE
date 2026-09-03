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

from cke.diagnostics import DegradationMixin, record_loaded_model

try:
    from sentence_transformers import SentenceTransformer
except ImportError:  # pragma: no cover - optional runtime dependency
    SentenceTransformer = None


logger = logging.getLogger(__name__)

_GLOBAL_MODEL_CACHE: dict[str, object] = {}

#: Width of the hashed fallback vector. Not a semantic embedding dimension.
FALLBACK_DIM = 128


class EmbeddingModel(DegradationMixin):
    """Sentence-transformer embedding wrapper with batch support.

    Args:
        model_name: sentence-transformers model identifier.
        strict: when True, raise :class:`~cke.diagnostics.DegradedComponentError`
            rather than falling back to hashed vectors. Every evaluation and
            benchmark path must pass ``strict=True``.
    """

    def __init__(
        self,
        model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        strict: bool = False,
    ) -> None:
        self._init_degradation(strict)
        self.model_name = model_name
        self.model = self._load_model(model_name)

    @property
    def dimension(self) -> int:
        """Width of the vectors this instance produces."""
        if self.model is None:
            return FALLBACK_DIM
        try:
            dim = self.model.get_sentence_embedding_dimension()
        except AttributeError:  # pragma: no cover - non-standard model object
            return FALLBACK_DIM
        return int(dim) if dim else FALLBACK_DIM

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

    def _load_model(self, model_name: str):
        """Load the sentence transformer, or declare the degradation."""
        cached = _GLOBAL_MODEL_CACHE.get(model_name)
        if cached is not None:
            return cached

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
            model = SentenceTransformer(model_name)
        except Exception as exc:  # noqa: BLE001 - download/runtime failures vary
            # Only cache successes: a transient load failure must not pin this
            # process to the fallback, and must not hide the cause from a later
            # strict construction.
            self._degrade(
                f"sentence-transformers could not load {model_name!r} "
                f"({type(exc).__name__}: {exc}), so text is embedded by hashing "
                f"tokens into a {FALLBACK_DIM}-dimension sparse vector instead "
                "of being embedded semantically"
            )
            return None

        _GLOBAL_MODEL_CACHE[model_name] = model
        record_loaded_model("EmbeddingModel", model_name, model_name)
        logger.info("EmbeddingModel loaded %s", model_name)
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
