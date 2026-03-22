"""Retriever that bridges dense chunks into structured evidence facts."""

from __future__ import annotations

import logging

from cke.entity_resolution.alias_registry import AliasRegistry
from cke.pipeline.types import EvidenceFact, ResolvedEntity, RetrievedChunk
from cke.retrieval.chunk_fact_store import ChunkFactStore
from cke.retrieval.ranking_config import RetrievalRankingConfig, load_ranking_config

logger = logging.getLogger(__name__)


class EvidenceRetriever:
    """Retrieve chunks and hydrate them with per-chunk extracted statements."""

    def __init__(
        self,
        rag_retriever,
        chunk_fact_store: ChunkFactStore,
        ranking_config: RetrievalRankingConfig | None = None,
        config_path: str = "configs/retrieval_ranking.yaml",
    ) -> None:
        self.rag_retriever = rag_retriever
        self.chunk_fact_store = chunk_fact_store
        self.ranking_config = ranking_config or load_ranking_config(config_path)

    def retrieve(
        self,
        query: str,
        resolved_entities: list[ResolvedEntity] | None = None,
        target_relations: list[str] | None = None,
        top_k: int = 5,
    ) -> tuple[list[RetrievedChunk], list[EvidenceFact]]:
        dense_chunks = self.rag_retriever.retrieve(query, k=top_k)

        retrieved_chunks: list[tuple[float, RetrievedChunk]] = []
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

        logger.info(
            "Mentions detected for retrieval: %s",
            [e.surface_form for e in resolved_entities],
        )
        logger.info("Target relations inferred: %s", target_relations)

        score_values = [float(chunk.get("score", 0.0)) for chunk in dense_chunks]
        normalized_dense_scores = self._normalize_dense_scores(score_values)

        for idx, chunk in enumerate(dense_chunks):
            chunk_id = str(chunk.get("doc_id", f"chunk-{idx}"))
            text = str(chunk.get("text", ""))
            score_dense = float(chunk.get("score", 0.0))
            source = str(chunk.get("source", chunk.get("doc_id", "unknown")))
            dense_relevance = normalized_dense_scores[idx]
            chunk_components = self._chunk_score_components(
                text=text,
                raw_chunk=chunk,
                dense_relevance=dense_relevance,
                entity_norms=entity_norms,
                target_relations=target_relations,
            )
            hybrid_chunk_score = sum(chunk_components.values())

            retrieved = RetrievedChunk(
                chunk_id=chunk_id,
                text=text,
                source=source,
                score_dense=score_dense,
                metadata={
                    "raw": chunk,
                    "score_components": chunk_components,
                    "hybrid_score": hybrid_chunk_score,
                    "dense_relevance": dense_relevance,
                },
            )
            retrieved_chunks.append((hybrid_chunk_score, retrieved))

            for statement in self.chunk_fact_store.get_facts(chunk_id):
                subj_norm = AliasRegistry.normalize(statement.subject)
                obj_norm = AliasRegistry.normalize(statement.object)
                rel_norm = AliasRegistry.normalize(statement.relation)

                entity_bonus = 0.0
                if subj_norm in entity_norms or obj_norm in entity_norms:
                    entity_bonus += 0.2
                if (
                    statement.canonical_subject_id
                    and AliasRegistry.normalize(statement.canonical_subject_id)
                    in entity_norms
                ):
                    entity_bonus += 0.15
                if (
                    statement.canonical_object_id
                    and AliasRegistry.normalize(statement.canonical_object_id)
                    in entity_norms
                ):
                    entity_bonus += 0.15

                relation_bonus = 0.0
                if target_relations and any(
                    target == rel_norm or target in rel_norm or rel_norm in target
                    for target in target_relations
                ):
                    relation_bonus = self.ranking_config.fact.relation_target_bonus

                if not statement.canonical_subject_id and subj_norm in entity_lookup:
                    statement.canonical_subject_id = entity_lookup[subj_norm]
                if not statement.canonical_object_id and obj_norm in entity_lookup:
                    statement.canonical_object_id = entity_lookup[obj_norm]

                combined_score = (
                    hybrid_chunk_score
                    + entity_bonus
                    + relation_bonus
                    + self._trust_score(statement.trust_score)
                )
                evidence = EvidenceFact(
                    statement=statement,
                    chunk_id=chunk_id,
                    source=statement.source or source,
                    trust_score=(
                        float(statement.trust_score)
                        if statement.trust_score is not None
                        else 0.5
                    ),
                    retrieval_score=combined_score,
                    entity_alignment_score=entity_bonus,
                    supporting_span=statement.supporting_span,
                    metadata={
                        "chunk_hybrid_score": hybrid_chunk_score,
                        "chunk_score_components": chunk_components,
                        "relation_match_bonus": relation_bonus,
                        "entity_alignment_bonus": entity_bonus,
                    },
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
        retrieved_chunk_list = [
            chunk
            for _, chunk in sorted(
                retrieved_chunks,
                key=lambda item: item[0],
                reverse=True,
            )[:top_k]
        ]

        logger.info("Retrieved %s chunks for query.", len(retrieved_chunk_list))
        logger.info("Evidence facts after scoring/filtering: %s", len(evidence_facts))
        return retrieved_chunk_list, evidence_facts

    def _chunk_score_components(
        self,
        *,
        text: str,
        raw_chunk: dict,
        dense_relevance: float,
        entity_norms: set[str],
        target_relations: list[str],
    ) -> dict[str, float]:
        lowered_text = AliasRegistry.normalize(text)
        mention_overlap = self._keyword_overlap(lowered_text, entity_norms)
        relation_overlap = self._keyword_overlap(lowered_text, set(target_relations))
        alias_overlap = self._alias_overlap(raw_chunk, entity_norms)
        source_bonus = self._source_bonus(raw_chunk)
        weights = self.ranking_config.chunk
        return {
            "dense": weights.dense_weight * dense_relevance,
            "entity_overlap": weights.entity_overlap_weight * mention_overlap,
            "relation_overlap": weights.relation_overlap_weight * relation_overlap,
            "alias_overlap": weights.alias_overlap_weight * alias_overlap,
            "source_trust": weights.source_trust_bonus * source_bonus,
        }

    def _normalize_dense_scores(self, scores: list[float]) -> list[float]:
        if not scores:
            return []
        maximum = max(scores)
        minimum = min(scores)
        if 0.0 <= minimum and maximum <= 1.0:
            return scores
        if maximum == minimum:
            return [1.0 for _ in scores]
        distance_like = minimum >= 0.0 and maximum > 1.0
        if distance_like:
            spread = max(maximum - minimum, 1e-9)
            return [1.0 - ((score - minimum) / spread) for score in scores]
        spread = max(maximum - minimum, 1e-9)
        return [(score - minimum) / spread for score in scores]

    def _keyword_overlap(self, text: str, candidates: set[str]) -> float:
        if not text or not candidates:
            return 0.0
        matches = 0
        total = 0
        for candidate in candidates:
            if not candidate:
                continue
            total += 1
            if candidate in text:
                matches += 1
                continue
            tokens = [token for token in candidate.split() if token]
            if tokens and all(token in text for token in tokens):
                matches += 1
        return matches / max(total, 1)

    def _alias_overlap(self, raw_chunk: dict, entity_norms: set[str]) -> float:
        alias_values = (
            raw_chunk.get("aliases") or raw_chunk.get("canonical_aliases") or []
        )
        normalized_aliases = {
            AliasRegistry.normalize(str(value)) for value in alias_values if value
        }
        if not normalized_aliases or not entity_norms:
            return 0.0
        return len(normalized_aliases & entity_norms) / max(1, len(entity_norms))

    def _source_bonus(self, raw_chunk: dict) -> float:
        explicit = raw_chunk.get("trust_score", raw_chunk.get("source_trust"))
        if explicit is not None:
            return self._trust_score(explicit)
        source = AliasRegistry.normalize(str(raw_chunk.get("source", "")))
        if any(token in source for token in {"wiki", "paper", "doc", "kb"}):
            return 1.0
        return 0.4 if source else 0.0

    def _trust_score(self, value) -> float:
        try:
            return max(0.0, min(1.0, float(value)))
        except (TypeError, ValueError):
            return 0.0
