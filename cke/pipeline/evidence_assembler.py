"""Lightweight assembly of retrieved evidence facts."""

from __future__ import annotations

from collections import defaultdict

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
            },
        )

    def _dedupe_facts(self, facts: list[EvidenceFact]) -> list[EvidenceFact]:
        seen: set[tuple[str, str, str, str]] = set()
        deduped: list[EvidenceFact] = []
        for fact in facts:
            key = (
                fact.chunk_id,
                fact.statement.subject,
                fact.statement.relation,
                fact.statement.object,
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
        entity_terms = {
            token
            for entity in resolved_entities
            for token in (entity.surface_form, entity.canonical_name)
            if token
        }
        relation_terms = self._relation_terms(query_plan)

        selected: list[EvidenceFact] = []
        for fact in facts:
            subject = fact.statement.subject.lower()
            obj = fact.statement.object.lower()
            relation = fact.statement.relation.lower()

            entity_overlap = any(
                term.lower() in subject or term.lower() in obj for term in entity_terms
            )
            relation_overlap = any(term in relation for term in relation_terms)
            if entity_overlap or relation_overlap:
                selected.append(fact)

        if not selected:
            ranked = sorted(facts, key=lambda fact: fact.retrieval_score, reverse=True)
            selected = ranked[: self.max_facts]

        return selected[: self.max_facts]

    def _relation_terms(self, query_plan: QueryPlan) -> set[str]:
        terms: set[str] = set()
        for step in getattr(query_plan, "decomposition", []):
            if str(step.get("type", "")) != "relation":
                continue
            value = str(step.get("value", "")).strip().lower()
            if value:
                terms.add(value)
        return terms

    def _build_candidate_paths(
        self,
        facts: list[EvidenceFact],
        query_plan: QueryPlan,
    ) -> list[list[Statement]]:
        relation_terms = self._relation_terms(query_plan)
        paths: list[list[Statement]] = []

        for fact in facts:
            relation = fact.statement.relation.lower()
            if relation_terms and any(term in relation for term in relation_terms):
                paths.append([fact.statement])

        for left in facts:
            for right in facts:
                if left is right:
                    continue
                if left.statement.object == right.statement.subject:
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

        return {
            "entities": sorted(entities),
            "facts_by_entity": dict(facts_by_entity),
            "facts_by_relation": dict(facts_by_relation),
        }
