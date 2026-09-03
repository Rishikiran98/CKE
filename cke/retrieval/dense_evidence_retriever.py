"""Minimal adapter that wraps a RAGRetriever for dense-only orchestrator usage."""

from __future__ import annotations

from cke.diagnostics import DegradationMixin
from cke.models import Statement
from cke.pipeline.types import EvidenceFact, ResolvedEntity, RetrievedChunk


#: Substituted when a dense result carries no score. Not a measurement.
_MISSING_SCORE = 0.5


class DenseEvidenceRetriever(DegradationMixin):
    """Retrieve evidence from a dense retriever without requiring a ChunkFactStore."""

    def __init__(self, rag_retriever, strict: bool = False) -> None:
        self._init_degradation(strict)
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
            if "score" in item:
                score = float(item["score"])
            else:
                # One missing key used to become three independently-named
                # figures that agree with each other: confidence, trust_score
                # and retrieval_score are all this value.
                self._degrade(
                    "a dense retrieval result carried no score, so a "
                    f"substituted value ({_MISSING_SCORE}) is reported as its "
                    "confidence, trust score and retrieval score alike. None "
                    "of the three is a measurement",
                )
                score = _MISSING_SCORE
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
