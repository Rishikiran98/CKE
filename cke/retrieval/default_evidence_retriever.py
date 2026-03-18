"""Default graph-backed evidence retrieval for orchestrator runtime wiring."""

from __future__ import annotations

from cke.entity_resolution.alias_registry import AliasRegistry
from cke.models import Statement
from cke.pipeline.types import EvidenceFact, ResolvedEntity, RetrievedChunk


class DefaultEvidenceRetriever:
    """Retrieve evidence directly from the in-memory graph when no RAG stack exists."""

    def __init__(self, graph_engine) -> None:
        self.graph_engine = graph_engine

    def retrieve(
        self,
        query: str,
        resolved_entities: list[ResolvedEntity] | None = None,
        target_relations: list[str] | None = None,
        top_k: int = 5,
    ) -> tuple[list[RetrievedChunk], list[EvidenceFact]]:
        resolved_entities = resolved_entities or []
        relation_terms = {
            AliasRegistry.normalize(relation) for relation in (target_relations or []) if relation
        }
        candidate_statements = self._candidate_statements(resolved_entities, relation_terms)
        scored: list[tuple[float, Statement]] = []
        for statement in candidate_statements:
            score = self._score_statement(statement, resolved_entities, relation_terms, query)
            scored.append((score, statement))

        scored.sort(key=lambda item: item[0], reverse=True)
        top_scored = scored[: max(top_k * 3, top_k)]

        retrieved_chunks: list[RetrievedChunk] = []
        evidence_facts: list[EvidenceFact] = []
        for index, (score, statement) in enumerate(top_scored):
            chunk_id = statement.chunk_id or f"graph::{index}"
            chunk_text = statement.as_text()
            source = statement.source or "graph"
            retrieved_chunks.append(
                RetrievedChunk(
                    chunk_id=chunk_id,
                    text=chunk_text,
                    source=source,
                    score_dense=score,
                    metadata={"raw": {"source": source, "text": chunk_text}},
                )
            )
            evidence_facts.append(
                EvidenceFact(
                    statement=statement,
                    chunk_id=chunk_id,
                    source=source,
                    trust_score=self._trust_score(statement),
                    retrieval_score=score,
                    entity_alignment_score=self._entity_alignment(statement, resolved_entities),
                    metadata={"retriever": "graph_default"},
                )
            )

        return retrieved_chunks[:top_k], evidence_facts

    def _candidate_statements(
        self,
        resolved_entities: list[ResolvedEntity],
        relation_terms: set[str],
    ) -> list[Statement]:
        seen: dict[tuple[str, str, str], Statement] = {}
        seed_norms = {
            AliasRegistry.normalize(value)
            for entity in resolved_entities
            for value in (entity.canonical_name, entity.entity_id, entity.surface_form)
            if value
        }

        def add(statement: Statement) -> None:
            seen.setdefault(statement.key(), statement)

        if seed_norms:
            one_hop: list[Statement] = []
            for entity in resolved_entities:
                for candidate in (entity.canonical_name, entity.surface_form):
                    if not candidate:
                        continue
                    for statement in self.graph_engine.get_neighbors(candidate):
                        add(statement)
                        one_hop.append(statement)
            for statement in one_hop:
                for neighbor in self.graph_engine.get_neighbors(statement.object):
                    add(neighbor)

        for relation in relation_terms:
            for statement in self.graph_engine.edges_for_relation(relation):
                add(statement)

        if not seen:
            for entity in self.graph_engine.all_entities():
                for statement in self.graph_engine.get_neighbors(entity):
                    add(statement)

        return list(seen.values())

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
        relation_bonus = 0.25 if relation_terms and relation_norm in relation_terms else 0.0
        lexical_bonus = 0.1 if AliasRegistry.normalize(statement.as_text()) in query_norm else 0.0
        return 0.35 + entity_alignment + relation_bonus + lexical_bonus + self._trust_score(statement)

    def _entity_alignment(
        self,
        statement: Statement,
        resolved_entities: list[ResolvedEntity],
    ) -> float:
        target_norms = {
            AliasRegistry.normalize(value)
            for entity in resolved_entities
            for value in (entity.canonical_name, entity.entity_id, entity.surface_form)
            if value
        }
        if not target_norms:
            return 0.0
        statement_norms = {
            AliasRegistry.normalize(value)
            for value in (
                statement.subject,
                statement.object,
                statement.canonical_subject_id,
                statement.canonical_object_id,
            )
            if value
        }
        return 0.25 * (len(statement_norms & target_norms) / len(target_norms))

    @staticmethod
    def _trust_score(statement: Statement) -> float:
        if statement.trust_score is not None:
            return max(0.0, min(1.0, float(statement.trust_score)))
        return max(0.0, min(1.0, float(statement.confidence)))
