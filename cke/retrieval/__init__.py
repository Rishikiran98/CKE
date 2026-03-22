"""Retrieval module exports.

Avoid eager heavy imports so lightweight submodules can be imported without
initializing embedding backends.
"""

from __future__ import annotations

from importlib import import_module

__all__ = [
    "ChunkFactStore",
    "DenseEvidenceRetriever",
    "EmbeddingModel",
    "EvidenceGraph",
    "EvidencePack",
    "EvidenceRetriever",
    "FaissIndex",
    "GraphRetriever",
    "HybridEvidenceRetriever",
    "HybridRetrievalMerger",
    "PathFeatures",
    "PathRankingModel",
    "RAGBaseline",
    "RAGRetriever",
    "RetrievalMode",
    "RetrievalRouter",
]

_LAZY_IMPORTS = {
    "EmbeddingModel": ("cke.retrieval.embedding_model", "EmbeddingModel"),
    "FaissIndex": ("cke.retrieval.faiss_index", "FaissIndex"),
    "EvidencePack": ("cke.retrieval.hybrid_retrieval", "EvidencePack"),
    "HybridRetrievalMerger": (
        "cke.retrieval.hybrid_retrieval",
        "HybridRetrievalMerger",
    ),
    "RetrievalRouter": ("cke.retrieval.retrieval_router", "RetrievalRouter"),
    "RAGBaseline": ("cke.retrieval.rag_baseline", "RAGBaseline"),
    "RAGRetriever": ("cke.retrieval.rag_baseline", "RAGRetriever"),
    "EvidenceGraph": ("cke.retrieval.evidence_graph", "EvidenceGraph"),
    "GraphRetriever": ("cke.retrieval.graph_retriever", "GraphRetriever"),
    "ChunkFactStore": ("cke.retrieval.chunk_fact_store", "ChunkFactStore"),
    "EvidenceRetriever": (
        "cke.retrieval.evidence_retriever",
        "EvidenceRetriever",
    ),
    "PathFeatures": ("cke.retrieval.path_ranking", "PathFeatures"),
    "PathRankingModel": ("cke.retrieval.path_ranking", "PathRankingModel"),
    "HybridEvidenceRetriever": (
        "cke.retrieval.hybrid_evidence_retriever",
        "HybridEvidenceRetriever",
    ),
    "DenseEvidenceRetriever": (
        "cke.retrieval.dense_evidence_retriever",
        "DenseEvidenceRetriever",
    ),
    "RetrievalMode": ("cke.retrieval.retrieval_mode", "RetrievalMode"),
}


def __getattr__(name: str):
    if name not in _LAZY_IMPORTS:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module_name, attr_name = _LAZY_IMPORTS[name]
    module = import_module(module_name)
    value = getattr(module, attr_name)
    globals()[name] = value
    return value
