"""Transparent heuristic scoring for shallow candidate paths."""

from __future__ import annotations

from cke.entity_resolution.alias_registry import AliasRegistry
from cke.pipeline.types import ResolvedEntity
from cke.retrieval.path_types import CandidatePath
from cke.router.query_plan import QueryPlan


class CandidatePathScorer:
    """Score path candidates with inspectable linear heuristics."""

    def score(
        self,
        candidate_paths: list[CandidatePath],
        query_plan: QueryPlan,
        resolved_entities: list[ResolvedEntity],
    ) -> list[CandidatePath]:
        relation_terms = self._relation_terms(query_plan)
        entity_terms = self._entity_terms(resolved_entities)

        scored: list[CandidatePath] = []
        for candidate in candidate_paths:
            path_entities = set()
            relation_hits = 0
            trust_values: list[float] = []
            for statement in candidate.statements:
                path_entities.update(
                    [
                        AliasRegistry.normalize(statement.subject),
                        AliasRegistry.normalize(statement.object),
                        AliasRegistry.normalize(statement.canonical_subject_id or ""),
                        AliasRegistry.normalize(statement.canonical_object_id or ""),
                    ]
                )
                relation = AliasRegistry.normalize(statement.relation)
                if relation_terms and any(
                    term == relation or term in relation or relation in term
                    for term in relation_terms
                ):
                    relation_hits += 1
                trust_values.append(
                    float(statement.trust_score)
                    if statement.trust_score is not None
                    else float(statement.confidence)
                )

            normalized_entities = {entity for entity in path_entities if entity}
            entity_overlap = (
                len(normalized_entities & entity_terms) / max(1, len(entity_terms))
                if entity_terms
                else 0.0
            )
            relation_match = relation_hits / max(1, len(candidate.statements))
            trust_score = min(trust_values) if trust_values else 0.0
            short_path_bonus = 0.15 if len(candidate.statements) == 1 else 0.05

            path_score = (
                (0.4 * entity_overlap)
                + (0.25 * relation_match)
                + (0.25 * trust_score)
                + short_path_bonus
            )
            if (
                getattr(query_plan, "multi_hop_hint", False)
                and len(candidate.statements) == 2
            ):
                path_score += 0.08
            if getattr(query_plan, "bridge_entities_expected", False) and entity_terms:
                if len(normalized_entities & entity_terms) >= 2:
                    path_score += 0.07

            scored.append(
                CandidatePath(
                    statements=list(candidate.statements),
                    path_score=path_score,
                    entity_overlap_score=entity_overlap,
                    relation_match_score=relation_match,
                    trust_score=trust_score,
                    summary=candidate.summary,
                )
            )

        scored.sort(
            key=lambda candidate: (
                candidate.path_score,
                candidate.trust_score,
                candidate.relation_match_score,
                -len(candidate.statements),
            ),
            reverse=True,
        )
        return scored

    def _relation_terms(self, query_plan: QueryPlan) -> set[str]:
        terms = {
            AliasRegistry.normalize(term)
            for term in getattr(query_plan, "target_relations", [])
            if term
        }
        for step in getattr(query_plan, "decomposition", []):
            if str(step.get("type", "")) == "relation":
                value = AliasRegistry.normalize(str(step.get("value", "")))
                if value:
                    terms.add(value)
        return terms

    def _entity_terms(self, resolved_entities: list[ResolvedEntity]) -> set[str]:
        return {
            AliasRegistry.normalize(value)
            for entity in resolved_entities
            for value in (
                entity.surface_form,
                entity.canonical_name,
                entity.entity_id,
            )
            if value
        }
