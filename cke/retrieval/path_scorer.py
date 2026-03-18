"""Transparent heuristic scoring for shallow candidate paths."""

from __future__ import annotations

from cke.entity_resolution.alias_registry import AliasRegistry
from cke.pipeline.types import ResolvedEntity
from cke.retrieval.ranking_config import RetrievalRankingConfig, load_ranking_config
from cke.retrieval.path_types import CandidatePath
from cke.router.query_plan import QueryPlan


class CandidatePathScorer:
    """Score path candidates with inspectable linear heuristics."""

    def __init__(
        self,
        ranking_config: RetrievalRankingConfig | None = None,
        config_path: str = "configs/retrieval_ranking.yaml",
    ) -> None:
        self.ranking_config = ranking_config or load_ranking_config(config_path)

    def score(
        self,
        candidate_paths: list[CandidatePath],
        query_plan: QueryPlan,
        resolved_entities: list[ResolvedEntity],
    ) -> list[CandidatePath]:
        relation_terms = self._relation_terms(query_plan)
        entity_terms = self._entity_terms(resolved_entities)
        query_type = self._query_type(query_plan, len(resolved_entities))
        seen_signatures: dict[tuple[str, ...], int] = {}

        scored: list[CandidatePath] = []
        for candidate in candidate_paths:
            path_entities = set()
            relation_hits = 0
            bridge_hits = 0
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
                if statement.canonical_object_id or statement.canonical_subject_id:
                    bridge_hits += 1
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
            trust_score = sum(trust_values) / max(1, len(trust_values))
            direct_bridge = bridge_hits / max(1, len(candidate.statements))
            query_bonus = self._query_type_bonus(
                query_type=query_type,
                path_length=len(candidate.statements),
                entity_hits=len(normalized_entities & entity_terms),
                bridge_score=direct_bridge,
            )
            length_penalty = self.ranking_config.path.length_penalty * max(
                0, len(candidate.statements) - 1
            )
            signature = tuple(
                AliasRegistry.normalize(statement.relation)
                for statement in candidate.statements
            )
            duplicate_count = seen_signatures.get(signature, 0)
            diversity_bonus = (
                self.ranking_config.path.diversity_bonus
                if duplicate_count == 0
                else 0.0
            )
            seen_signatures[signature] = duplicate_count + 1

            path_score = (
                (self.ranking_config.path.entity_weight * entity_overlap)
                + (self.ranking_config.path.relation_weight * relation_match)
                + (self.ranking_config.path.trust_weight * trust_score)
                + (self.ranking_config.path.direct_bridge_bonus * direct_bridge)
                + query_bonus
                + diversity_bonus
                - length_penalty
            )
            metadata = {
                "query_type": query_type,
                "entity_overlap": entity_overlap,
                "relation_match": relation_match,
                "trust": trust_score,
                "direct_bridge": direct_bridge,
                "query_bonus": query_bonus,
                "diversity_bonus": diversity_bonus,
                "length_penalty": length_penalty,
                "relation_signature": signature,
            }

            scored.append(
                CandidatePath(
                    statements=list(candidate.statements),
                    path_score=path_score,
                    entity_overlap_score=entity_overlap,
                    relation_match_score=relation_match,
                    trust_score=trust_score,
                    summary=candidate.summary,
                    metadata=metadata,
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

    def _query_type(self, query_plan: QueryPlan, entity_count: int) -> str:
        operator_hint = str(getattr(query_plan, "operator_hint", "") or "").lower()
        if getattr(query_plan, "multi_hop_hint", False):
            return "multi_hop"
        if operator_hint in {"equals", "equality"} or entity_count >= 2:
            return "comparison"
        return "direct_lookup"

    def _query_type_bonus(
        self,
        *,
        query_type: str,
        path_length: int,
        entity_hits: int,
        bridge_score: float,
    ) -> float:
        weights = self.ranking_config.path
        if query_type == "direct_lookup":
            return weights.direct_bridge_bonus if path_length == 1 else 0.0
        if query_type == "multi_hop":
            bonus = weights.multi_hop_bonus if path_length == 2 else 0.0
            if bridge_score > 0 and entity_hits >= 1:
                bonus += weights.direct_bridge_bonus / 2.0
            return bonus
        if query_type == "comparison":
            return weights.comparison_bonus if entity_hits >= 2 else 0.0
        return 0.0

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
