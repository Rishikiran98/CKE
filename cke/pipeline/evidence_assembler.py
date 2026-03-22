"""Lightweight assembly of retrieved evidence facts."""

from __future__ import annotations

import hashlib
import json
import logging
import re
from collections import defaultdict
from typing import Any

from cke.entity_resolution.alias_registry import AliasRegistry
from cke.pipeline.types import (
    EvidenceFact,
    ReasoningContext,
    ResolvedEntity,
    RetrievedChunk,
)
from cke.retrieval.path_generator import CandidatePathGenerator
from cke.retrieval.ranking_config import RetrievalRankingConfig, load_ranking_config
from cke.retrieval.path_scorer import CandidatePathScorer
from cke.retrieval.subgraph_builder import LocalSubgraphBuilder
from cke.router.query_plan import QueryPlan


def _qualifier_hash(qualifiers: dict[str, Any]) -> str:
    """Stable hash of qualifiers for dedup keying."""
    if not qualifiers:
        return ""
    return hashlib.sha256(
        json.dumps(qualifiers, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()[:16]


def _extract_date_tokens(temporal: Any) -> list[str]:
    """Extract date-like tokens from a temporal qualifier value."""
    tokens: list[str] = []
    if isinstance(temporal, dict):
        for v in temporal.values():
            if v:
                tokens.append(str(v).lower())
    elif isinstance(temporal, str):
        tokens.append(temporal.lower())
    return tokens

logger = logging.getLogger(__name__)


class EvidenceAssembler:
    """Deduplicate and bound retrieved evidence facts."""

    def __init__(
        self,
        max_facts: int = 20,
        max_candidate_paths: int = 5,
        subgraph_builder: LocalSubgraphBuilder | None = None,
        path_generator: CandidatePathGenerator | None = None,
        path_scorer: CandidatePathScorer | None = None,
        ranking_config: RetrievalRankingConfig | None = None,
        config_path: str = "configs/retrieval_ranking.yaml",
    ) -> None:
        self.max_facts = max_facts
        self.max_candidate_paths = max_candidate_paths
        self.subgraph_builder = subgraph_builder or LocalSubgraphBuilder()
        self.path_generator = path_generator or CandidatePathGenerator()
        self.ranking_config = ranking_config or load_ranking_config(config_path)
        self.path_scorer = path_scorer or CandidatePathScorer(
            ranking_config=self.ranking_config
        )

    def assemble(
        self,
        query: str,
        query_plan: QueryPlan,
        resolved_entities: list[ResolvedEntity],
        chunks: list[RetrievedChunk],
        facts: list[EvidenceFact],
    ) -> ReasoningContext:
        deduped_facts = self._dedupe_facts(facts)
        filtered_facts = self._filter_facts(
            deduped_facts, resolved_entities, query_plan, query
        )
        subgraph = self.subgraph_builder.build(resolved_entities, filtered_facts)
        raw_candidate_paths = self.path_generator.generate(subgraph, query, query_plan)
        candidate_paths = self._select_candidate_paths(
            raw_candidate_paths, query_plan, resolved_entities
        )
        debug_subgraph = self._build_subgraph_debug_view(subgraph)

        logger.info("Subgraph entity count: %s", len(subgraph.entities))
        logger.info("Subgraph edge count: %s", len(subgraph.edges))
        logger.info("Candidate paths before scoring: %s", len(raw_candidate_paths))
        logger.info("Candidate paths after scoring: %s", len(candidate_paths))
        logger.info(
            "Top path scores: %s",
            [round(path.path_score, 3) for path in candidate_paths[:3]],
        )

        return ReasoningContext(
            query=query,
            query_plan=query_plan,
            resolved_entities=resolved_entities,
            retrieved_chunks=chunks,
            evidence_facts=filtered_facts,
            candidate_paths=candidate_paths,
            subgraph=subgraph,
            decomposition=list(getattr(query_plan, "decomposition", [])),
            trace_metadata={
                "evidence_facts_before_filtering": len(facts),
                "evidence_facts_after_filtering": len(filtered_facts),
                "candidate_paths_before_scoring": len(raw_candidate_paths),
                "candidate_paths": len(candidate_paths),
                "subgraph_entity_count": len(subgraph.entities),
                "subgraph_edge_count": len(subgraph.edges),
                "top_path_scores": [
                    round(path.path_score, 3) for path in candidate_paths[:3]
                ],
                "facts_by_canonical_subject": debug_subgraph.get(
                    "facts_by_canonical_subject", {}
                ),
                "facts_by_relation": debug_subgraph.get("facts_by_relation", {}),
                "top_retrieved_chunks": self._top_chunk_debug(chunks),
                "top_evidence_facts": self._top_fact_debug(filtered_facts),
                "top_candidate_paths": self._top_path_debug(candidate_paths),
            },
        )

    def _dedupe_facts(self, facts: list[EvidenceFact]) -> list[EvidenceFact]:
        deduped: dict[tuple[str, ...], EvidenceFact] = {}
        for fact in facts:
            key = (
                AliasRegistry.normalize(fact.statement.subject),
                AliasRegistry.normalize(fact.statement.relation),
                AliasRegistry.normalize(fact.statement.object),
                _qualifier_hash(fact.statement.qualifiers),
            )
            current = deduped.get(key)
            if current is None or fact.retrieval_score > current.retrieval_score:
                deduped[key] = fact
        return list(deduped.values())

    def _filter_facts(
        self,
        facts: list[EvidenceFact],
        resolved_entities: list[ResolvedEntity],
        query_plan: QueryPlan,
        query: str,
    ) -> list[EvidenceFact]:
        relation_terms = self._relation_terms(query_plan)
        target_entities = self._entity_terms(resolved_entities)
        query_type = self._query_type(query, query_plan)

        scored: list[tuple[float, EvidenceFact]] = []
        for fact in facts:
            score, metadata = self._fact_score(
                fact=fact,
                relation_terms=relation_terms,
                target_entities=target_entities,
                query_type=query_type,
                resolved_entities=resolved_entities,
                query=query,
            )
            fact.metadata.update(metadata)
            fact.statement.retrieval_score = score
            fact.retrieval_score = score
            scored.append((score, fact))

        scored.sort(key=lambda item: item[0], reverse=True)
        selected = [fact for _, fact in scored[: self.max_facts]]

        if getattr(query_plan, "multi_hop_hint", False):
            selected = self._preserve_connected_facts(selected, scored)

        if len(resolved_entities) >= 2:
            selected = self._preserve_dual_entity_coverage(
                selected, scored, resolved_entities
            )

        if not selected:
            ranked = sorted(facts, key=lambda fact: fact.retrieval_score, reverse=True)
            selected = ranked[: self.max_facts]

        return selected

    def _qualifier_relevance_score(
        self, fact: EvidenceFact, query: str
    ) -> float:
        """Score boost/penalty based on qualifier match with query context."""
        qualifiers = fact.statement.qualifiers
        if not qualifiers:
            return 0.0
        score = 0.0
        temporal = qualifiers.get("temporal")
        if temporal:
            date_tokens = _extract_date_tokens(temporal)
            lowered = query.lower()
            if any(token in lowered for token in date_tokens):
                score += 0.15
        modality = qualifiers.get("modality")
        if modality == "deprecated":
            score -= 0.1
        elif modality == "typical":
            score += 0.05
        return score

    def _fact_score(
        self,
        *,
        fact: EvidenceFact,
        relation_terms: set[str],
        target_entities: set[str],
        query_type: str,
        resolved_entities: list[ResolvedEntity],
        query: str = "",
    ) -> tuple[float, dict[str, float | str]]:
        relation = AliasRegistry.normalize(fact.statement.relation)
        fact_entities = {
            AliasRegistry.normalize(fact.statement.subject),
            AliasRegistry.normalize(fact.statement.object),
            AliasRegistry.normalize(fact.statement.canonical_subject_id or ""),
            AliasRegistry.normalize(fact.statement.canonical_object_id or ""),
        }
        fact_entities.discard("")

        entity_alignment = (
            len(fact_entities & target_entities) / max(1, len(target_entities))
            if target_entities
            else 0.0
        )
        relation_match = (
            max(
                (
                    1.0
                    for term in relation_terms
                    if term == relation or term in relation or relation in term
                ),
                default=0.0,
            )
            if relation_terms
            else 0.0
        )
        canonical_bonus = self._canonical_match_bonus(fact, resolved_entities)
        operator_bonus = self._operator_support_bonus(
            query_type, relation, relation_terms
        )
        weights = self.ranking_config.fact
        score_components = {
            "chunk_score": weights.chunk_weight * max(0.0, fact.retrieval_score),
            "entity_alignment": weights.entity_weight * entity_alignment,
            "relation_match": weights.relation_weight * relation_match,
            "trust": weights.trust_weight * max(0.0, min(1.0, fact.trust_score)),
            "canonical_match": canonical_bonus,
            "relation_target_bonus": (
                weights.relation_target_bonus if relation_match else 0.0
            ),
            "operator_support": operator_bonus,
            "qualifier_relevance": self._qualifier_relevance_score(fact, query),
        }
        score = sum(score_components.values())
        return score, {
            "fact_score": score,
            "query_type": query_type,
            **score_components,
        }

    def _relation_terms(self, query_plan: QueryPlan) -> set[str]:
        terms: set[str] = {
            AliasRegistry.normalize(v)
            for v in getattr(query_plan, "target_relations", [])
            if v
        }
        for step in getattr(query_plan, "decomposition", []):
            if str(step.get("type", "")) != "relation":
                continue
            value = str(step.get("value", "")).strip().lower()
            if value:
                terms.add(AliasRegistry.normalize(value))
        return terms

    def _select_candidate_paths(
        self,
        candidate_paths,
        query_plan: QueryPlan,
        resolved_entities: list[ResolvedEntity],
    ):
        scored_paths = self.path_scorer.score(
            candidate_paths, query_plan=query_plan, resolved_entities=resolved_entities
        )
        return scored_paths[: self.max_candidate_paths]

    def _build_subgraph_debug_view(
        self,
        subgraph,
    ) -> dict[str, list[dict[str, str]]]:
        facts_by_entity: dict[str, list[dict[str, str]]] = defaultdict(list)
        facts_by_relation: dict[str, list[dict[str, str]]] = defaultdict(list)
        facts_by_canonical_subject: dict[str, list[dict[str, str]]] = defaultdict(list)

        for statement in subgraph.edges:
            payload = {
                "subject": statement.subject,
                "relation": statement.relation,
                "object": statement.object,
            }
            facts_by_entity[statement.subject].append(payload)
            facts_by_entity[statement.object].append(payload)
            facts_by_relation[statement.relation].append(payload)
            canonical_subject = statement.canonical_subject_id or statement.subject
            facts_by_canonical_subject[canonical_subject].append(payload)

        return {
            "entities": list(subgraph.entities),
            "facts_by_entity": dict(facts_by_entity),
            "facts_by_relation": dict(facts_by_relation),
            "facts_by_canonical_subject": dict(facts_by_canonical_subject),
        }

    def _preserve_connected_facts(
        self,
        selected: list[EvidenceFact],
        scored: list[tuple[float, EvidenceFact]],
    ) -> list[EvidenceFact]:
        selected_keys = {fact.statement.key() for fact in selected}
        connector_entities = {
            AliasRegistry.normalize(fact.statement.subject) for fact in selected
        } | {AliasRegistry.normalize(fact.statement.object) for fact in selected}
        keep = list(selected)
        for _, fact in scored:
            if fact.statement.key() in selected_keys:
                continue
            if AliasRegistry.normalize(
                fact.statement.subject
            ) in connector_entities or (
                AliasRegistry.normalize(fact.statement.object) in connector_entities
            ):
                keep.append(fact)
                selected_keys.add(fact.statement.key())
            if len(keep) >= self.max_facts:
                break
        return keep[: self.max_facts]

    def _preserve_dual_entity_coverage(
        self,
        selected: list[EvidenceFact],
        scored: list[tuple[float, EvidenceFact]],
        resolved_entities: list[ResolvedEntity],
    ) -> list[EvidenceFact]:
        keep = list(selected)
        selected_keys = {fact.statement.key() for fact in keep}
        for entity in resolved_entities[:2]:
            entity_terms = {
                AliasRegistry.normalize(value)
                for value in (
                    entity.surface_form,
                    entity.canonical_name,
                    entity.entity_id,
                )
                if value
            }
            if any(self._fact_mentions_entity(fact, entity_terms) for fact in keep):
                continue
            for _, fact in scored:
                if fact.statement.key() in selected_keys:
                    continue
                if self._fact_mentions_entity(fact, entity_terms):
                    keep.append(fact)
                    selected_keys.add(fact.statement.key())
                    break
        keep.sort(key=lambda fact: fact.retrieval_score, reverse=True)
        return keep[: self.max_facts]

    def _entity_terms(self, resolved_entities: list[ResolvedEntity]) -> set[str]:
        return {
            AliasRegistry.normalize(value)
            for entity in resolved_entities
            for value in (
                entity.canonical_name,
                entity.entity_id,
                entity.surface_form,
            )
            if value
        }

    def _fact_mentions_entity(self, fact: EvidenceFact, entity_terms: set[str]) -> bool:
        values = {
            AliasRegistry.normalize(fact.statement.subject),
            AliasRegistry.normalize(fact.statement.object),
            AliasRegistry.normalize(fact.statement.canonical_subject_id or ""),
            AliasRegistry.normalize(fact.statement.canonical_object_id or ""),
        }
        return bool({value for value in values if value} & entity_terms)

    def _canonical_match_bonus(
        self,
        fact: EvidenceFact,
        resolved_entities: list[ResolvedEntity],
    ) -> float:
        weights = self.ranking_config.fact
        for entity in resolved_entities:
            entity_terms = {
                AliasRegistry.normalize(value)
                for value in (entity.canonical_name, entity.entity_id)
                if value
            }
            if entity_terms and self._fact_mentions_entity(fact, entity_terms):
                return weights.canonical_match_bonus
        return 0.0

    def _operator_support_bonus(
        self,
        query_type: str,
        relation: str,
        relation_terms: set[str],
    ) -> float:
        weights = self.ranking_config.fact
        if query_type == "count":
            return weights.operator_bonus if "child" in relation else 0.0
        if query_type == "comparison":
            return (
                weights.operator_bonus
                if relation_terms and relation in relation_terms
                else 0.0
            )
        if query_type == "date_compare":
            return (
                weights.operator_bonus
                if "date" in relation or "year" in relation
                else 0.0
            )
        if query_type == "direct_lookup":
            return (
                (weights.operator_bonus / 2.0)
                if relation_terms and relation in relation_terms
                else 0.0
            )
        return 0.0

    def _query_type(self, query: str, query_plan: QueryPlan) -> str:
        operator_hint = str(getattr(query_plan, "operator_hint", "") or "").lower()
        lowered = query.lower()
        if operator_hint == "count" or lowered.startswith("how many"):
            return "count"
        if operator_hint in {"equals", "equality"} or "same" in lowered:
            return "comparison"
        if "earlier" in lowered or "later" in lowered or "date" in lowered:
            return "date_compare"
        if getattr(query_plan, "multi_hop_hint", False):
            return "multi_hop"
        return "direct_lookup"

    def _top_chunk_debug(
        self,
        chunks: list[RetrievedChunk],
    ) -> list[dict[str, object]]:
        return [
            {
                "chunk_id": chunk.chunk_id,
                "source": chunk.source,
                "hybrid_score": chunk.metadata.get("hybrid_score"),
                "score_components": chunk.metadata.get("score_components", {}),
            }
            for chunk in chunks[:5]
        ]

    def _top_fact_debug(self, facts: list[EvidenceFact]) -> list[dict[str, object]]:
        return [
            {
                "statement": fact.statement.as_text(),
                "chunk_id": fact.chunk_id,
                "retrieval_score": fact.retrieval_score,
                "metadata": dict(fact.metadata),
            }
            for fact in facts[:5]
        ]

    def _top_path_debug(self, candidate_paths) -> list[dict[str, object]]:
        return [
            {
                "summary": path.summary,
                "path_score": path.path_score,
                "metadata": dict(getattr(path, "metadata", {})),
            }
            for path in candidate_paths[:5]
        ]
