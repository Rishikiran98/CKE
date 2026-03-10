"""Baseline dense retrieval using embeddings + FAISS."""

from __future__ import annotations

import math
import time
from dataclasses import dataclass
from typing import List

try:
    import numpy as np
except Exception:  # pragma: no cover
    np = None

try:
    import faiss
except Exception:  # pragma: no cover
    faiss = None

try:
    from sentence_transformers import SentenceTransformer
except Exception:  # pragma: no cover
    SentenceTransformer = None


@dataclass
class RetrievalResult:
    chunk: str
    score: float


class RAGBaseline:
    """FAISS-backed retriever with graceful offline fallback."""

    def __init__(self) -> None:
        self.model = self._load_model()
        self.chunks: list[str] = []
        self.vectors = None
        self.index = None

    def build_index(self, documents: List[str]) -> None:
        self.chunks = documents
        self.vectors = self._encode(documents)
        if faiss is not None and np is not None:
            dim = self.vectors.shape[1]
            self.index = faiss.IndexFlatIP(dim)
            self.index.add(self.vectors.astype("float32"))

    def retrieve(
        self, query: str, top_k: int = 3
    ) -> tuple[List[RetrievalResult], float]:
        if not self.chunks:
            return [], 0.0

        start = time.perf_counter()
        qvec = self._encode([query])
        if self.index is not None and np is not None:
            scores, idx = self.index.search(qvec.astype("float32"), top_k)
            results = [
                RetrievalResult(self.chunks[i], float(scores[0][rank]))
                for rank, i in enumerate(idx[0])
            ]
        else:
            q = qvec[0] if isinstance(qvec[0], list) else qvec[0].tolist()
            sims = [
                self._dot(v if isinstance(v, list) else v.tolist(), q)
                for v in self.vectors
            ]
            best_idx = sorted(range(len(sims)), key=lambda i: sims[i], reverse=True)[
                :top_k
            ]
            results = [
                RetrievalResult(self.chunks[i], float(sims[i])) for i in best_idx
            ]

        return results, (time.perf_counter() - start) * 1000

    def _load_model(self):
        if SentenceTransformer is None:
            return None
        try:
            return SentenceTransformer("all-MiniLM-L6-v2")
        except Exception:
            return None

    def _encode(self, texts: List[str]):
        if self.model is not None and np is not None:
            return np.array(
                self.model.encode(texts, normalize_embeddings=True), dtype=np.float32
            )

        vectors: list[list[float]] = []
        for text in texts:
            vec = [0.0] * 128
            for token in text.lower().split():
                vec[hash(token) % 128] += 1.0
            norm = math.sqrt(sum(v * v for v in vec)) or 1.0
            vectors.append([v / norm for v in vec])

        if np is not None:
            return np.array(vectors, dtype=np.float32)
        return vectors

    @staticmethod
    def _dot(a: list[float], b: list[float]) -> float:
        return sum(x * y for x, y in zip(a, b))
