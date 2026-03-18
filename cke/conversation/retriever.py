"""Semantic retrieval over conversational turns with graph enrichment."""

from __future__ import annotations

from collections import defaultdict
import math
import re

import numpy as np

from cke.conversation.memory import ConversationalMemoryStore
from cke.conversation.reference_resolution import ConversationalReferenceResolver
from cke.conversation.types import RetrievalBundle, RetrievedMemory
from cke.models import Statement
from cke.retrieval.embedding_model import EmbeddingModel


class ConversationalRetriever:
    """Dense-first conversational retrieval followed by fact/graph enrichment."""

    def __init__(
        self,
        memory_store: ConversationalMemoryStore,
        embedding_model: EmbeddingModel | None = None,
        reference_resolver: ConversationalReferenceResolver | None = None,
    ) -> None:
        self.memory_store = memory_store
        self.embedding_model = embedding_model or EmbeddingModel()
        self.reference_resolver = reference_resolver or ConversationalReferenceResolver(
            memory_store
        )

    def retrieve(
        self,
        query: str,
        *,
        conversation_id: str,
        top_k: int = 5,
    ) -> RetrievalBundle:
        turns = self.memory_store.get_turns(conversation_id)
        if not turns:
            return RetrievalBundle(rewritten_query=query)

        seed_hits = self._retrieve_turn_hits(query, turns, top_k=top_k)
        rewritten_query, bindings = self.reference_resolver.rewrite(
            query,
            conversation_id=conversation_id,
            retrieved_turns=seed_hits,
        )
        final_hits = self._retrieve_turn_hits(rewritten_query, turns, top_k=top_k)
        facts = self._collect_facts(final_hits)
        graph_neighbors = self._collect_graph_neighbors(final_hits, facts)
        candidate_paths = self._candidate_paths(facts, graph_neighbors)
        return RetrievalBundle(
            rewritten_query=rewritten_query,
            retrieved_turns=final_hits,
            retrieved_facts=facts,
            graph_neighbors=graph_neighbors,
            candidate_paths=candidate_paths,
            metadata={"reference_bindings": bindings},
        )

    def _retrieve_turn_hits(
        self,
        query: str,
        turns,
        *,
        top_k: int,
    ) -> list[RetrievedMemory]:
        query_vec = self.embedding_model.embed_text(query)
        turn_vectors = self.embedding_model.embed_texts([turn.text for turn in turns])
        results: list[RetrievedMemory] = []
        for turn, vector in zip(turns, turn_vectors):
            dense_score = self._cosine(query_vec, vector)
            overlap = self._keyword_overlap(query, turn.text)
            recency_bonus = turn.turn_order / max(1, len(turns)) * 0.1
            entity_bonus = (
                0.08
                if any(entity.lower() in query.lower() for entity in turn.entities)
                else 0.0
            )
            score = dense_score + overlap + recency_bonus + entity_bonus
            results.append(
                RetrievedMemory(
                    memory_id=turn.turn_id,
                    memory_type="turn",
                    text=turn.text,
                    score=score,
                    conversation_id=turn.conversation_id,
                    turn_id=turn.turn_id,
                    turn_order=turn.turn_order,
                    role=turn.role,
                    entities=list(turn.entities),
                    facts=list(turn.facts),
                    metadata={"timestamp": turn.timestamp},
                )
            )
        results.sort(key=lambda item: item.score, reverse=True)
        return results[:top_k]

    def _collect_facts(self, hits: list[RetrievedMemory]) -> list[Statement]:
        ranked: list[tuple[float, Statement]] = []
        for hit in hits:
            for fact in hit.facts:
                bonus = (
                    0.05
                    if fact.context.get("topic")
                    in {"preference", "timeline", "communication"}
                    else 0.0
                )
                ranked.append((hit.score + bonus, fact))
        ranked.sort(
            key=lambda item: (
                item[0],
                float(item[1].confidence),
            ),
            reverse=True,
        )
        deduped: dict[tuple[str, str, str], Statement] = {}
        for _, fact in ranked:
            deduped.setdefault((fact.subject, fact.relation, fact.object), fact)
        return list(deduped.values())

    def _collect_graph_neighbors(
        self,
        hits: list[RetrievedMemory],
        facts: list[Statement],
    ) -> list[Statement]:
        graph = self.memory_store.graph_engine
        candidate_entities = []
        for hit in hits:
            candidate_entities.extend(hit.entities)
        for fact in facts:
            candidate_entities.extend([fact.subject, fact.object])
        seen: dict[tuple[str, str, str], Statement] = {}
        for entity in candidate_entities:
            for neighbor in graph.get_neighbors(entity):
                seen.setdefault(neighbor.key(), neighbor)
        return list(seen.values())

    def _candidate_paths(
        self,
        facts: list[Statement],
        neighbors: list[Statement],
    ) -> list[list[Statement]]:
        all_facts = facts + neighbors
        by_subject: dict[str, list[Statement]] = defaultdict(list)
        for fact in all_facts:
            by_subject[fact.subject].append(fact)
        paths: list[list[Statement]] = []
        for fact in all_facts:
            for follow in by_subject.get(fact.object, []):
                paths.append([fact, follow])
        return paths[:5]

    def _keyword_overlap(self, query: str, text: str) -> float:
        query_terms = {
            token
            for token in re.findall(r"[a-z0-9]+", query.lower())
            if token not in {"the", "a", "an", "that", "again", "did", "i", "you"}
        }
        if not query_terms:
            return 0.0
        text_terms = set(re.findall(r"[a-z0-9]+", text.lower()))
        return 0.25 * len(query_terms & text_terms) / len(query_terms)

    def _cosine(self, left: np.ndarray, right: np.ndarray) -> float:
        denom = float(np.linalg.norm(left) * np.linalg.norm(right))
        if math.isclose(denom, 0.0):
            return 0.0
        return float(np.dot(left, right) / denom)
