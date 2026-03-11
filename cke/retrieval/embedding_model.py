"""Embedding model wrapper with global model caching."""

from __future__ import annotations

import hashlib
from typing import Iterable

import numpy as np

try:
    from sentence_transformers import SentenceTransformer
except Exception:  # pragma: no cover - optional runtime dependency
    SentenceTransformer = None


_GLOBAL_MODEL_CACHE: dict[str, object] = {}


class EmbeddingModel:
    """Sentence-transformer embedding wrapper with batch support."""

    def __init__(
        self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2"
    ) -> None:
        self.model_name = model_name
        self.model = self._get_or_create_model(model_name)

    def embed_text(self, text: str) -> np.ndarray:
        """Embed a single text and return a float32 vector."""
        return self.embed_texts([text])[0]

    def embed_texts(self, texts: Iterable[str], batch_size: int = 32) -> np.ndarray:
        """Embed many texts efficiently in batches."""
        text_list = list(texts)
        if not text_list:
            return np.zeros((0, 128), dtype=np.float32)

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

    @staticmethod
    def _get_or_create_model(model_name: str):
        if model_name in _GLOBAL_MODEL_CACHE:
            return _GLOBAL_MODEL_CACHE[model_name]
        if SentenceTransformer is None:
            _GLOBAL_MODEL_CACHE[model_name] = None
            return None
        try:
            model = SentenceTransformer(model_name)
        except Exception:  # pragma: no cover - runtime model download issue
            model = None
        _GLOBAL_MODEL_CACHE[model_name] = model
        return model

    @staticmethod
    def _fallback_embed(text: str, dim: int = 128) -> np.ndarray:
        vector = np.zeros((dim,), dtype=np.float32)
        for token in text.lower().split():
            digest = hashlib.sha256(token.encode("utf-8")).hexdigest()
            idx = int(digest, 16) % dim
            vector[idx] += 1.0
        norm = float(np.linalg.norm(vector))
        if norm > 0:
            vector /= norm
        return vector
