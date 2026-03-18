"""Helpers for building a shallow query-local evidence subgraph."""

from __future__ import annotations

from cke.entity_resolution.alias_registry import AliasRegistry
from cke.pipeline.types import EvidenceFact, ResolvedEntity
from cke.retrieval.path_types import LocalSubgraph


class LocalSubgraphBuilder:
    """Construct a bounded local subgraph around resolved entities."""

    def build(
        self,
        resolved_entities: list[ResolvedEntity],
        evidence_facts: list[EvidenceFact],
        graph_engine=None,
    ) -> LocalSubgraph:
        del graph_engine  # Optional future integration point.

        seed_entities = self._seed_entities(resolved_entities)
        seed_norms = self._seed_norms(resolved_entities)
        facts = [fact.statement for fact in evidence_facts]

        selected_keys: set[tuple[str, str, str]] = set()
        selected_edges = []
        frontier_entities: set[str] = set(seed_norms)
        participating_entities: dict[str, str] = {}

        if not seed_norms:
            for statement in facts:
                self._add_edge(
                    statement, selected_keys, selected_edges, participating_entities
                )
            return LocalSubgraph(
                seed_entities=seed_entities,
                entities=sorted(participating_entities.values()),
                edges=selected_edges,
                metadata={"hop_depth": 0, "expansion_mode": "all_evidence"},
            )

        one_hop = [
            statement
            for statement in facts
            if self._touches_entities(statement, frontier_entities)
        ]
        for statement in one_hop:
            self._add_edge(
                statement, selected_keys, selected_edges, participating_entities
            )
            frontier_entities.update(self._statement_entity_norms(statement))

        two_hop = [
            statement
            for statement in facts
            if statement.key() not in selected_keys
            and self._touches_entities(statement, frontier_entities)
        ]
        for statement in two_hop:
            self._add_edge(
                statement, selected_keys, selected_edges, participating_entities
            )

        return LocalSubgraph(
            seed_entities=seed_entities,
            entities=sorted(participating_entities.values()),
            edges=selected_edges,
            metadata={
                "hop_depth": 2 if two_hop else 1,
                "one_hop_edges": len(one_hop),
                "two_hop_edges": len(two_hop),
                "expansion_mode": "evidence_first",
            },
        )

    def _seed_entities(self, resolved_entities: list[ResolvedEntity]) -> list[str]:
        values: list[str] = []
        seen: set[str] = set()
        for entity in resolved_entities:
            for candidate in (entity.canonical_name, entity.surface_form):
                normalized = AliasRegistry.normalize(candidate or "")
                if not normalized or normalized in seen:
                    continue
                seen.add(normalized)
                values.append(candidate)
        return values

    def _seed_norms(self, resolved_entities: list[ResolvedEntity]) -> set[str]:
        return {
            AliasRegistry.normalize(candidate)
            for entity in resolved_entities
            for candidate in (
                entity.canonical_name,
                entity.entity_id,
                entity.surface_form,
            )
            if candidate
        }

    def _touches_entities(self, statement, entity_norms: set[str]) -> bool:
        return bool(self._statement_entity_norms(statement) & entity_norms)

    def _statement_entity_norms(self, statement) -> set[str]:
        norms = {
            AliasRegistry.normalize(statement.subject),
            AliasRegistry.normalize(statement.object),
        }
        if statement.canonical_subject_id:
            norms.add(AliasRegistry.normalize(statement.canonical_subject_id))
        if statement.canonical_object_id:
            norms.add(AliasRegistry.normalize(statement.canonical_object_id))
        return {value for value in norms if value}

    def _add_edge(
        self,
        statement,
        selected_keys: set[tuple[str, str, str]],
        selected_edges: list,
        participating_entities: dict[str, str],
    ) -> None:
        if statement.key() in selected_keys:
            return
        selected_keys.add(statement.key())
        selected_edges.append(statement)

        for value in (
            statement.subject,
            statement.object,
            statement.canonical_subject_id,
            statement.canonical_object_id,
        ):
            normalized = AliasRegistry.normalize(value or "")
            if normalized and normalized not in participating_entities:
                participating_entities[normalized] = str(value)
