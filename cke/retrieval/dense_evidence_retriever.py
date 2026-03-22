"""Minimal adapter that wraps a RAGRetriever for dense-only orchestrator usage."""

from __future__ import annotations

from cke.models import Statement
from cke.pipeline.types import EvidenceFact, ResolvedEntity, RetrievedChunk


class DenseEvidenceRetriever:
    """Retrieve evidence from a dense retriever without requiring a ChunkFactStore."""

    def __init__(self, rag_retriever) -> None:
        self.rag_retriever = rag_retriever

    def retrieve(
        self,
        query: str,
        resolved_entities: list[ResolvedEntity] | None = None,
        target_relations: list[str] | None = None,
        top_k: int = 5,
    ) -> tuple[list[RetrievedChunk], list[EvidenceFact]]:
        dense_results = self.rag_retriever.retrieve(query, k=top_k)

        retrieved_chunks: list[RetrievedChunk] = []
        evidence_facts: list[EvidenceFact] = []

        for idx, item in enumerate(dense_results):
            text = str(item.get("text", ""))
            if not text:
                continue
            score = float(item.get("score", 0.5))
            chunk_id = str(item.get("doc_id", f"dense::{idx}"))

            retrieved_chunks.append(
                RetrievedChunk(
                    chunk_id=chunk_id,
                    text=text,
                    source="dense",
                    score_dense=score,
                    metadata={"retriever": "dense_only"},
                )
            )

            synthetic_stmt = Statement(
                subject="",
                relation="dense_retrieval",
                object=text,
                confidence=score,
                source="dense",
                chunk_id=chunk_id,
            )
            evidence_facts.append(
                EvidenceFact(
                    statement=synthetic_stmt,
                    chunk_id=chunk_id,
                    source="dense",
                    trust_score=score,
                    retrieval_score=score,
                    entity_alignment_score=0.0,
                    metadata={"retriever": "dense_only", "synthetic": True},
                )
            )

        return retrieved_chunks, evidence_facts
