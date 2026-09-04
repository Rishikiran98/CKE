"""Baseline RAG retriever built on sentence-transformers and FAISS.

This is the dense baseline CKE is measured against. If its embedder or its
index has degraded, the comparison is not against dense retrieval at all, so
``strict=True`` propagates to both and the retriever reports their state.
"""

from __future__ import annotations

import time
from dataclasses import dataclass

from cke.diagnostics import DegradationMixin, require_strict_component
from cke.retrieval.embedding_model import EmbeddingModel
from cke.retrieval.faiss_index import FaissIndex


class RAGRetriever(DegradationMixin):
    """Embed query and retrieve top-k documents from a FAISS index.

    Args:
        embedding_model: an embedder to reuse. When supplied under
            ``strict=True`` it must not already be degraded.
        strict: when True, this retriever and the components it builds raise
            rather than degrade. Every evaluation and benchmark path must pass
            ``strict=True``.
    """

    def __init__(
        self,
        embedding_model: EmbeddingModel | None = None,
        strict: bool = False,
    ) -> None:
        self._init_degradation(strict)
        # The docstring above already promised this and nothing enforced it:
        # a prebuilt degraded embedder was accepted in silence, and the
        # baseline CKE is measured against reported itself strict while
        # embedding by hashing.
        require_strict_component(
            type(self).__name__, embedding_model, "embedding model", self.strict
        )
        self.embedding_model = embedding_model or EmbeddingModel(strict=strict)
        self.index = FaissIndex(strict=strict)

        # An embedder passed in by the caller may already have degraded before
        # it reached us; inherit that rather than reporting a healthy baseline.
        # A caller-supplied embedder need not implement the degradation
        # contract, in which case there is nothing to inherit.
        if getattr(self.embedding_model, "degraded", False):
            self._degrade(
                "its embedding model is degraded, so this is not a dense "
                "retrieval baseline: "
                f"{getattr(self.embedding_model, 'degraded_reason', 'unknown')}"
            )
        if self.index.degraded:
            self._degrade(f"its vector index is degraded: {self.index.degraded_reason}")

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
