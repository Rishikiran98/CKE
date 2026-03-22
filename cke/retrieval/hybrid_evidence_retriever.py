"""Adapter that wraps RetrievalRouter to match the orchestrator retriever interface."""

from __future__ import annotations

import logging
from typing import Any

from cke.entity_resolution.alias_registry import AliasRegistry
from cke.models import Statement
from cke.pipeline.types import EvidenceFact, ResolvedEntity, RetrievedChunk
from cke.retrieval.retrieval_router import RetrievalRouter

logger = logging.getLogger(__name__)


class HybridEvidenceRetriever:
    """Bridge between RetrievalRouter (EvidencePack) and the orchestrator interface.

    The orchestrator expects ``retrieve(query, resolved_entities, target_relations, top_k)``
    returning ``tuple[list[RetrievedChunk], list[EvidenceFact]]``.  RetrievalRouter returns
    an ``EvidencePack`` with ``graph_statements`` and ``fallback_chunks``.  This adapter
    converts between the two and applies entity/relation scoring for ranking.
    """

    def __init__(self, retrieval_router: RetrievalRouter) -> None:
        self.retrieval_router = retrieval_router

    # ------------------------------------------------------------------
    # Public interface (matches DefaultEvidenceRetriever / EvidenceRetriever)
    # ------------------------------------------------------------------

    def retrieve(
        self,
        query: str,
        resolved_entities: list[ResolvedEntity] | None = None,
        target_relations: list[str] | None = None,
        top_k: int = 5,
    ) -> tuple[list[RetrievedChunk], list[EvidenceFact]]:
        resolved_entities = resolved_entities or []
        relation_terms = {
            AliasRegistry.normalize(r) for r in (target_relations or []) if r
        }

        evidence_pack = self.retrieval_router.retrieve(query, max_depth=2)

        scored: list[tuple[float, RetrievedChunk, EvidenceFact]] = []

        # --- graph statements ---
        for idx, stmt in enumerate(evidence_pack.graph_statements):
            score = self._score_statement(
                stmt, resolved_entities, relation_terms, query
            )
            chunk_id = stmt.chunk_id or f"hybrid_graph::{idx}"
            chunk_text = stmt.as_text()
            source = stmt.source or "graph"
            chunk = RetrievedChunk(
                chunk_id=chunk_id,
                text=chunk_text,
                source=source,
                score_dense=score,
                metadata={"retriever": "hybrid_graph"},
            )
            fact = EvidenceFact(
                statement=stmt,
                chunk_id=chunk_id,
                source=source,
                trust_score=self._trust_score(stmt),
                retrieval_score=score,
                entity_alignment_score=self._entity_alignment(stmt, resolved_entities),
                metadata={"retriever": "hybrid_graph"},
            )
            scored.append((score, chunk, fact))

        # --- dense fallback chunks ---
        for idx, chunk_text in enumerate(evidence_pack.fallback_chunks):
            synthetic_stmt = Statement(
                subject="",
                relation="dense_fallback",
                object=chunk_text,
                confidence=0.5,
                source="dense_fallback",
                chunk_id=f"hybrid_dense::{idx}",
            )
            score = self._score_dense_chunk(chunk_text, resolved_entities, query)
            chunk_id = f"hybrid_dense::{idx}"
            chunk = RetrievedChunk(
                chunk_id=chunk_id,
                text=chunk_text,
                source="dense_fallback",
                score_dense=score,
                metadata={"retriever": "hybrid_dense", "synthetic": True},
            )
            fact = EvidenceFact(
                statement=synthetic_stmt,
                chunk_id=chunk_id,
                source="dense_fallback",
                trust_score=0.5,
                retrieval_score=score,
                entity_alignment_score=0.0,
                metadata={"retriever": "hybrid_dense", "synthetic": True},
            )
            scored.append((score, chunk, fact))

        scored.sort(key=lambda item: item[0], reverse=True)
        top_scored = scored[: max(top_k * 3, top_k)]

        retrieved_chunks = [chunk for _, chunk, _ in top_scored[:top_k]]
        evidence_facts = [fact for _, _, fact in top_scored]
        return retrieved_chunks, evidence_facts

    def metrics_snapshot(self) -> dict[str, Any]:
        return self.retrieval_router.metrics_snapshot()

    # ------------------------------------------------------------------
    # Scoring helpers (aligned with DefaultEvidenceRetriever)
    # ------------------------------------------------------------------

    def _score_statement(
        self,
        statement: Statement,
        resolved_entities: list[ResolvedEntity],
        relation_terms: set[str],
        query: str,
    ) -> float:
        query_norm = AliasRegistry.normalize(query)
        relation_norm = AliasRegistry.normalize(statement.relation)
        entity_alignment = self._entity_alignment(statement, resolved_entities)
        relation_bonus = (
            0.25 if relation_terms and relation_norm in relation_terms else 0.0
        )
        lexical_bonus = (
            0.1 if AliasRegistry.normalize(statement.as_text()) in query_norm else 0.0
        )
        return (
            0.35
            + entity_alignment
            + relation_bonus
            + lexical_bonus
            + self._trust_score(statement)
        )

    @staticmethod
    def _score_dense_chunk(
        chunk_text: str,
        resolved_entities: list[ResolvedEntity],
        query: str,
    ) -> float:
        chunk_lower = chunk_text.lower()
        overlap = sum(
            1
            for ent in resolved_entities
            if ent.canonical_name and ent.canonical_name.lower() in chunk_lower
        )
        entity_bonus = min(0.25, 0.1 * overlap) if resolved_entities else 0.0
        return 0.35 + entity_bonus

    @staticmethod
    def _entity_alignment(
        statement: Statement,
        resolved_entities: list[ResolvedEntity],
    ) -> float:
        target_norms = {
            AliasRegistry.normalize(v)
            for entity in resolved_entities
            for v in (entity.canonical_name, entity.entity_id, entity.surface_form)
            if v
        }
        if not target_norms:
            return 0.0
        statement_norms = {
            AliasRegistry.normalize(v)
            for v in (
                statement.subject,
                statement.object,
                statement.canonical_subject_id,
                statement.canonical_object_id,
            )
            if v
        }
        return 0.25 * (len(statement_norms & target_norms) / len(target_norms))

    @staticmethod
    def _trust_score(statement: Statement) -> float:
        if statement.trust_score is not None:
            return max(0.0, min(1.0, float(statement.trust_score)))
        return max(0.0, min(1.0, float(statement.confidence)))
