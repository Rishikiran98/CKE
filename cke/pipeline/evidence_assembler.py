"""Lightweight assembly of retrieved evidence facts."""

from __future__ import annotations

from collections import defaultdict

from cke.entity_resolution.alias_registry import AliasRegistry
from cke.models import Statement
from cke.pipeline.types import (
    EvidenceFact,
    ReasoningContext,
    ResolvedEntity,
    RetrievedChunk,
)
from cke.router.query_plan import QueryPlan


class EvidenceAssembler:
    """Deduplicate and bound retrieved evidence facts."""

    def __init__(self, max_facts: int = 20) -> None:
        self.max_facts = max_facts

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
        candidate_paths = self._build_candidate_paths(filtered_facts, query_plan)
        subgraph = self._build_subgraph(filtered_facts)

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
                "candidate_paths": len(candidate_paths),
                "facts_by_canonical_subject": subgraph.get(
                    "facts_by_canonical_subject", {}
                ),
                "facts_by_relation": subgraph.get("facts_by_relation", {}),
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
                subject in target_entities
                or obj in target_entities
                or (subject_id and subject_id in target_entities)
                or (object_id and object_id in target_entities)
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

    def _build_candidate_paths(
        self,
        facts: list[EvidenceFact],
        query_plan: QueryPlan,
    ) -> list[list[Statement]]:
        relation_terms = self._relation_terms(query_plan)
        paths: list[list[Statement]] = []

        for fact in facts:
            relation = AliasRegistry.normalize(fact.statement.relation)
            if relation_terms and any(term in relation for term in relation_terms):
                paths.append([fact.statement])

        for left in facts:
            for right in facts:
                if left is right:
                    continue
                if AliasRegistry.normalize(
                    left.statement.object
                ) == AliasRegistry.normalize(right.statement.subject):
                    paths.append([left.statement, right.statement])

        deduped_paths: list[list[Statement]] = []
        seen_paths: set[tuple[tuple[str, str, str], ...]] = set()
        for path in paths:
            key = tuple(statement.key() for statement in path)
            if key in seen_paths:
                continue
            seen_paths.add(key)
            deduped_paths.append(path)
        return deduped_paths

    def _build_subgraph(
        self, facts: list[EvidenceFact]
    ) -> dict[str, list[dict[str, str]]]:
        facts_by_entity: dict[str, list[dict[str, str]]] = defaultdict(list)
        facts_by_relation: dict[str, list[dict[str, str]]] = defaultdict(list)
        facts_by_canonical_subject: dict[str, list[dict[str, str]]] = defaultdict(list)
        entities: set[str] = set()

        for fact in facts:
            payload = {
                "subject": fact.statement.subject,
                "relation": fact.statement.relation,
                "object": fact.statement.object,
            }
            entities.update((fact.statement.subject, fact.statement.object))
            facts_by_entity[fact.statement.subject].append(payload)
            facts_by_entity[fact.statement.object].append(payload)
            facts_by_relation[fact.statement.relation].append(payload)
            canonical_subject = (
                fact.statement.canonical_subject_id or fact.statement.subject
            )
            facts_by_canonical_subject[canonical_subject].append(payload)

        return {
            "entities": sorted(entities),
            "facts_by_entity": dict(facts_by_entity),
            "facts_by_relation": dict(facts_by_relation),
            "facts_by_canonical_subject": dict(facts_by_canonical_subject),
        }
