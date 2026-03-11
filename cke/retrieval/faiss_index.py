"""FAISS-backed document index with fallback search."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

try:
    import faiss
except Exception:  # pragma: no cover
    faiss = None


class FaissIndex:
    """Build/search a dense vector index mapped to source documents."""

    def __init__(self, dimension: int | None = None) -> None:
        self.dimension = dimension
        self.index = None
        self.documents: list[dict[str, Any]] = []
        self._vectors: np.ndarray | None = None

    def build_index(self, docs: list[dict[str, Any]]) -> None:
        """Create a new index from embedded docs."""
        self.documents = []
        self._vectors = None
        self.index = None
        self.add_documents(docs)

    def add_documents(self, docs: list[dict[str, Any]]) -> None:
        """Append embedded documents to the index."""
        if not docs:
            return

        vectors = np.asarray([doc["embedding"] for doc in docs], dtype=np.float32)
        if vectors.ndim != 2:
            raise ValueError("Embeddings must be 2D.")

        if self.dimension is None:
            self.dimension = int(vectors.shape[1])
        if int(vectors.shape[1]) != self.dimension:
            raise ValueError("Embedding dimension mismatch.")

        for doc in docs:
            self.documents.append(
                {"doc_id": str(doc["doc_id"]), "text": str(doc["text"])}
            )

        if faiss is not None:
            if self.index is None:
                self.index = faiss.IndexFlatL2(self.dimension)
            self.index.add(vectors)
        else:
            self._vectors = (
                vectors
                if self._vectors is None
                else np.vstack([self._vectors, vectors])
            )

    def search(self, query_embedding: np.ndarray, k: int) -> list[dict[str, Any]]:
        """Search nearest docs and return list of mappings with scores."""
        if not self.documents:
            return []
        query = np.asarray(query_embedding, dtype=np.float32).reshape(1, -1)
        k = max(1, min(k, len(self.documents)))

        if self.index is not None:
            distances, indices = self.index.search(query, k)
            return [
                {
                    "doc_id": self.documents[idx]["doc_id"],
                    "text": self.documents[idx]["text"],
                    "score": float(dist),
                }
                for dist, idx in zip(distances[0], indices[0])
                if idx >= 0
            ]

        if self._vectors is None:
            return []
        dists = np.sum((self._vectors - query) ** 2, axis=1)
        best = np.argsort(dists)[:k]
        return [
            {
                "doc_id": self.documents[idx]["doc_id"],
                "text": self.documents[idx]["text"],
                "score": float(dists[idx]),
            }
            for idx in best
        ]

    def save(self, path: str | Path) -> None:
        """Persist index and id->document mapping as JSON."""
        if self._vectors is None and self.index is not None and faiss is not None:
            self._vectors = np.asarray(self.index.reconstruct_n(0, self.index.ntotal))

        payload = {
            "dimension": self.dimension,
            "documents": self.documents,
            "vectors": self._vectors.tolist() if self._vectors is not None else None,
        }
        output_path = Path(path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(payload), encoding="utf-8")

    def load(self, path: str | Path) -> None:
        """Restore index and mapping from a JSON file."""
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        self.dimension = payload.get("dimension")
        self.documents = payload.get("documents", [])

        vectors = payload.get("vectors")
        self._vectors = (
            None if vectors is None else np.asarray(vectors, dtype=np.float32)
        )

        self.index = None
        if (
            faiss is not None
            and self._vectors is not None
            and self.dimension is not None
        ):
            self.index = faiss.IndexFlatL2(int(self.dimension))
            self.index.add(np.asarray(self._vectors, dtype=np.float32))
