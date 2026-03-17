"""Retriever that bridges dense chunks into structured evidence facts."""

from __future__ import annotations

import logging

from cke.entity_resolution.alias_registry import AliasRegistry
from cke.pipeline.types import EvidenceFact, ResolvedEntity, RetrievedChunk
from cke.retrieval.chunk_fact_store import ChunkFactStore


logger = logging.getLogger(__name__)


class EvidenceRetriever:
    """Retrieve chunks and hydrate them with per-chunk extracted statements."""

    def __init__(self, rag_retriever, chunk_fact_store: ChunkFactStore) -> None:
        self.rag_retriever = rag_retriever
        self.chunk_fact_store = chunk_fact_store

    def retrieve(
        self,
        query: str,
        resolved_entities: list[ResolvedEntity] | None = None,
        target_relations: list[str] | None = None,
        top_k: int = 5,
    ) -> tuple[list[RetrievedChunk], list[EvidenceFact]]:
        dense_chunks = self.rag_retriever.retrieve(query, k=top_k)

        retrieved_chunks: list[RetrievedChunk] = []
        scored_facts: list[tuple[float, EvidenceFact]] = []

        resolved_entities = resolved_entities or []
        target_relations = [r.lower() for r in (target_relations or []) if r]

        entity_norms = {
            AliasRegistry.normalize(v)
            for ent in resolved_entities
            for v in [ent.surface_form, ent.canonical_name, ent.entity_id]
            if v
        }

        entity_lookup = {
            AliasRegistry.normalize(v): ent.entity_id
            for ent in resolved_entities
            for v in [ent.surface_form, ent.canonical_name, ent.entity_id]
            if v
        }

        logger.info("Mentions detected for retrieval: %s", [e.surface_form for e in resolved_entities])
        logger.info("Target relations inferred: %s", target_relations)

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
                subj_norm = AliasRegistry.normalize(statement.subject)
                obj_norm = AliasRegistry.normalize(statement.object)
                rel_norm = AliasRegistry.normalize(statement.relation)

                entity_bonus = 0.0
                if subj_norm in entity_norms or obj_norm in entity_norms:
                    entity_bonus += 0.25
                if statement.canonical_subject_id and AliasRegistry.normalize(statement.canonical_subject_id) in entity_norms:
                    entity_bonus += 0.20
                if statement.canonical_object_id and AliasRegistry.normalize(statement.canonical_object_id) in entity_norms:
                    entity_bonus += 0.20

                relation_bonus = 0.0
                if target_relations and any(
                    target == rel_norm or target in rel_norm or rel_norm in target
                    for target in target_relations
                ):
                    relation_bonus = 0.25

                if not statement.canonical_subject_id and subj_norm in entity_lookup:
                    statement.canonical_subject_id = entity_lookup[subj_norm]
                if not statement.canonical_object_id and obj_norm in entity_lookup:
                    statement.canonical_object_id = entity_lookup[obj_norm]

                combined_score = score_dense + entity_bonus + relation_bonus
                evidence = EvidenceFact(
                    statement=statement,
                    chunk_id=chunk_id,
                    source=statement.source or source,
                    trust_score=(float(statement.trust_score) if statement.trust_score is not None else 0.5),
                    retrieval_score=combined_score,
                    entity_alignment_score=entity_bonus,
                    supporting_span=statement.supporting_span,
                )
                scored_facts.append((combined_score, evidence))

        logger.info("Evidence facts before scoring/filtering: %s", len(scored_facts))

        deduped: dict[tuple[str, str, str, str], tuple[float, EvidenceFact]] = {}
        for score, fact in scored_facts:
            key = (
                AliasRegistry.normalize(fact.statement.subject),
                AliasRegistry.normalize(fact.statement.relation),
                AliasRegistry.normalize(fact.statement.object),
                fact.chunk_id,
            )
            current = deduped.get(key)
            if current is None or score > current[0]:
                deduped[key] = (score, fact)

        ranked = sorted(deduped.values(), key=lambda item: item[0], reverse=True)
        max_facts = max(top_k * 3, top_k)
        evidence_facts = [fact for _, fact in ranked[:max_facts]]

        logger.info("Retrieved %s chunks for query.", len(retrieved_chunks))
        logger.info("Evidence facts after scoring/filtering: %s", len(evidence_facts))
        return retrieved_chunks, evidence_facts
