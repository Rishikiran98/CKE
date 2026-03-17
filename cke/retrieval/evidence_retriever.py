"""Retriever that bridges dense chunks into structured evidence facts."""

from __future__ import annotations

import logging

from cke.pipeline.types import EvidenceFact, RetrievedChunk
from cke.retrieval.chunk_fact_store import ChunkFactStore


logger = logging.getLogger(__name__)


class EvidenceRetriever:
    """Retrieve chunks and hydrate them with per-chunk extracted statements."""

    def __init__(self, rag_retriever, chunk_fact_store: ChunkFactStore) -> None:
        self.rag_retriever = rag_retriever
        self.chunk_fact_store = chunk_fact_store

    def retrieve(
        self, query: str, top_k: int = 5
    ) -> tuple[list[RetrievedChunk], list[EvidenceFact]]:
        dense_chunks = self.rag_retriever.retrieve(query, k=top_k)

        retrieved_chunks: list[RetrievedChunk] = []
        evidence_facts: list[EvidenceFact] = []

        for idx, chunk in enumerate(dense_chunks):
            chunk_id = str(chunk.get("doc_id", f"chunk-{idx}"))
            text = str(chunk.get("text", ""))
            score_dense = float(chunk.get("score", 0.0))
            source = str(chunk.get("source", chunk.get("doc_id", "unknown")))

            retrieved = RetrievedChunk(
                chunk_id=chunk_id,
                text=text,
                source=source,
                score_dense=score_dense,
                metadata={"raw": chunk},
            )
            retrieved_chunks.append(retrieved)

            for statement in self.chunk_fact_store.get_facts(chunk_id):
                evidence_facts.append(
                    EvidenceFact(
                        statement=statement,
                        chunk_id=chunk_id,
                        source=statement.source or source,
                        trust_score=(
                            float(statement.trust_score)
                            if statement.trust_score is not None
                            else 0.5
                        ),
                        retrieval_score=score_dense,
                        entity_alignment_score=0.0,
                        supporting_span=statement.supporting_span,
                    )
                )

        logger.info("Retrieved %s chunks for query.", len(retrieved_chunks))
        logger.info("Converted chunks into %s evidence facts.", len(evidence_facts))
        return retrieved_chunks, evidence_facts
