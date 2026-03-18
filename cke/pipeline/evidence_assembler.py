"""Lightweight assembly of retrieved evidence facts."""

from __future__ import annotations

import logging
from collections import defaultdict

from cke.entity_resolution.alias_registry import AliasRegistry
from cke.pipeline.types import (
    EvidenceFact,
    ReasoningContext,
    ResolvedEntity,
    RetrievedChunk,
)
from cke.retrieval.path_generator import CandidatePathGenerator
from cke.retrieval.path_scorer import CandidatePathScorer
from cke.retrieval.subgraph_builder import LocalSubgraphBuilder
from cke.router.query_plan import QueryPlan


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
    ) -> None:
        self.max_facts = max_facts
        self.max_candidate_paths = max_candidate_paths
        self.subgraph_builder = subgraph_builder or LocalSubgraphBuilder()
        self.path_generator = path_generator or CandidatePathGenerator()
        self.path_scorer = path_scorer or CandidatePathScorer()

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
            deduped_facts, resolved_entities, query_plan
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
            },
        )

    def _dedupe_facts(self, facts: list[EvidenceFact]) -> list[EvidenceFact]:
        seen: set[tuple[str, str, str, str]] = set()
        deduped: list[EvidenceFact] = []
        for fact in facts:
            key = (
                fact.chunk_id,
                AliasRegistry.normalize(fact.statement.subject),
                AliasRegistry.normalize(fact.statement.relation),
                AliasRegistry.normalize(fact.statement.object),
            )
            if key in seen:
                continue
            seen.add(key)
            deduped.append(fact)
        return deduped

    def _filter_facts(
        self,
        facts: list[EvidenceFact],
        resolved_entities: list[ResolvedEntity],
        query_plan: QueryPlan,
    ) -> list[EvidenceFact]:
        relation_terms = self._relation_terms(query_plan)
        canonical_names = [
            AliasRegistry.normalize(e.canonical_name)
            for e in resolved_entities
            if e.canonical_name
        ]
        canonical_ids = [
            AliasRegistry.normalize(e.entity_id)
            for e in resolved_entities
            if e.entity_id
        ]
        target_entities = set(canonical_names + canonical_ids)

        scored: list[tuple[float, EvidenceFact]] = []
        for fact in facts:
            relation = AliasRegistry.normalize(fact.statement.relation)
            subject = AliasRegistry.normalize(fact.statement.subject)
            obj = AliasRegistry.normalize(fact.statement.object)
            subject_id = AliasRegistry.normalize(
                fact.statement.canonical_subject_id or ""
            )
            object_id = AliasRegistry.normalize(
                fact.statement.canonical_object_id or ""
            )

            entity_match = int(
                bool(
                    subject in target_entities
                    or obj in target_entities
                    or (subject_id and subject_id in target_entities)
                    or (object_id and object_id in target_entities)
                )
            )
            relation_match = int(
                any(term == relation or term in relation for term in relation_terms)
            )
            score = (
                fact.retrieval_score + (0.2 * entity_match) + (0.15 * relation_match)
            )
            scored.append((score, fact))

        scored.sort(key=lambda item: item[0], reverse=True)
        selected = [fact for _, fact in scored[: self.max_facts]]

        if getattr(query_plan, "multi_hop_hint", False):
            selected = self._preserve_connected_facts(selected, scored)

        if len(resolved_entities) >= 2:
            # Preserve top fact for both sides in comparison queries.
            keep = list(selected)
            seen_entities = {AliasRegistry.normalize(f.statement.subject) for f in keep}
            for entity in resolved_entities[:2]:
                key = AliasRegistry.normalize(entity.canonical_name)
                if key in seen_entities:
                    continue
                for _, fact in scored:
                    if AliasRegistry.normalize(fact.statement.subject) == key:
                        keep.append(fact)
                        break
            selected = keep[: self.max_facts]

        if not selected:
            ranked = sorted(facts, key=lambda fact: fact.retrieval_score, reverse=True)
            selected = ranked[: self.max_facts]

        return selected

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
