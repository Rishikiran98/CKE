"""Baseline RAG retriever built on sentence-transformers and FAISS."""

from __future__ import annotations

import time
from dataclasses import dataclass

from cke.retrieval.embedding_model import EmbeddingModel
from cke.retrieval.faiss_index import FaissIndex


class RAGRetriever:
    """Embed query and retrieve top-k documents from a FAISS index."""

    def __init__(self, embedding_model: EmbeddingModel | None = None) -> None:
        self.embedding_model = embedding_model or EmbeddingModel()
        self.index = FaissIndex()

    def build_index(self, docs: list[dict[str, str]] | list[str]) -> None:
        prepared: list[dict[str, str]] = []
        for i, doc in enumerate(docs):
            if isinstance(doc, str):
                prepared.append({"doc_id": str(i), "text": doc})
            else:
                prepared.append(
                    {"doc_id": str(doc["doc_id"]), "text": str(doc["text"])}
                )

        embeddings = self.embedding_model.embed_texts([d["text"] for d in prepared])
        indexed = [
            {"doc_id": d["doc_id"], "text": d["text"], "embedding": emb}
            for d, emb in zip(prepared, embeddings)
        ]
        self.index.build_index(indexed)

    def retrieve(self, query: str, k: int = 5) -> list[dict[str, str | float]]:
        query_embedding = self.embedding_model.embed_text(query)
        return self.index.search(query_embedding, k)


@dataclass
class RetrievalResult:
    chunk: str
    score: float


class RAGBaseline(RAGRetriever):
    """Backward-compatible baseline wrapper used by existing experiment code."""

    def retrieve(
        self, query: str, top_k: int = 3
    ) -> tuple[list[RetrievalResult], float]:
        start = time.perf_counter()
        docs = super().retrieve(query, k=top_k)
        results = [
            RetrievalResult(chunk=str(item["text"]), score=float(item["score"]))
            for item in docs
        ]
        latency_ms = (time.perf_counter() - start) * 1000
        return results, latency_ms
