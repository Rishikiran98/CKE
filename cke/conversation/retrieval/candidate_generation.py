"""Candidate generation across raw events and canonical memories."""

from __future__ import annotations

import math

from cke.conversation.config import RetrievalConfig
from cke.conversation.memory_store import ConversationMemoryStore
from cke.conversation.patterns import lexical_overlap
from cke.conversation.types import RetrievedMemory
from cke.retrieval.embedding_model import EmbeddingModel


class CandidateGenerator:
    """Generate mixed raw-turn and canonical-memory retrieval candidates."""

    def __init__(
        self,
        memory_store: ConversationMemoryStore,
        embedding_model: EmbeddingModel | None = None,
        config: RetrievalConfig | None = None,
    ) -> None:
        self.memory_store = memory_store
        self.embedding_model = embedding_model or EmbeddingModel()
        self.config = config or RetrievalConfig()
        self._vector_cache: dict[str, tuple[str, object]] = {}

    def generate(self, query: str, *, conversation_id: str) -> list[RetrievedMemory]:
        events = self.memory_store.get_events(conversation_id)
        memories = self.memory_store.get_canonical_memories(conversation_id)
        query_vec = self.embedding_model.embed_text(query)
        candidates: list[RetrievedMemory] = []

        for event, vector in zip(events, self._embed_events(events)):
            score = self._score(
                query,
                query_vec,
                event.text,
                vector,
                recency_hint=event.turn_order / max(1, len(events)),
            )
            candidates.append(
                RetrievedMemory(
                    memory_id=event.event_id,
                    memory_type="event",
                    text=event.text,
                    score=score,
                    conversation_id=event.conversation_id,
                    turn_id=event.turn_id,
                    turn_order=event.turn_order,
                    role=event.role,
                    metadata={"timestamp": event.timestamp},
                )
            )
        for memory in memories:
            text = f"{memory.subject} {memory.relation} {memory.object}"
            score = (
                lexical_overlap(query, text, stop_words=self.config.stop_words)
                * self.config.lexical_weight
            ) + 0.1
            candidates.append(
                RetrievedMemory(
                    memory_id=memory.memory_id,
                    memory_type="canonical_memory",
                    text=text,
                    score=score + 0.05,
                    conversation_id=memory.conversation_id,
                    turn_id=memory.provenance[0].turn_id if memory.provenance else None,
                    subject=memory.subject,
                    relation=memory.relation,
                    object=memory.object,
                    metadata={
                        "kind": memory.kind.value,
                        "mention_count": memory.mention_count,
                    },
                )
            )
        candidates.sort(key=lambda item: item.score, reverse=True)
        return candidates

    def _embed_events(self, events):
        vectors = [None] * len(events)
        missing_indexes = []
        missing_texts = []
        for index, event in enumerate(events):
            signature = f"{event.timestamp}:{event.text}"
            cached = self._vector_cache.get(event.event_id)
            if cached and cached[0] == signature:
                vectors[index] = cached[1]
            else:
                missing_indexes.append(index)
                missing_texts.append(event.text)
        if missing_texts:
            fresh_vectors = self.embedding_model.embed_texts(missing_texts)
            for index, vector in zip(missing_indexes, fresh_vectors):
                event = events[index]
                signature = f"{event.timestamp}:{event.text}"
                self._vector_cache[event.event_id] = (signature, vector)
                vectors[index] = vector
        return [vector for vector in vectors if vector is not None]

    def _score(
        self, query: str, query_vec, text: str, text_vec, *, recency_hint: float
    ) -> float:
        lexical = lexical_overlap(query, text, stop_words=self.config.stop_words)
        dense = self._cosine(query_vec, text_vec)
        return (
            (dense * self.config.dense_weight)
            + (lexical * self.config.lexical_weight)
            + (recency_hint * self.config.recency_weight)
        )

    def _cosine(self, left, right) -> float:
        denom = float((left @ left) ** 0.5 * (right @ right) ** 0.5)
        if math.isclose(denom, 0.0):
            return 0.0
        return float((left @ right) / denom)
