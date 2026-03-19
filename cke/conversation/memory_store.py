
"""Storage for raw events, canonical memories, and retrieval-facing indexes."""

from __future__ import annotations

from collections import defaultdict

from cke.conversation.config import ConversationConfig
from cke.conversation.types import CanonicalMemory, ConversationEvent, ConversationTurn, EvidenceSet
from cke.graph_engine.graph_engine import KnowledgeGraphEngine
from cke.models import Statement


class ConversationMemoryStore:
    """In-memory store with clear separation between events and canonical memories."""

    def __init__(
        self,
        *,
        graph_engine: KnowledgeGraphEngine | None = None,
        config: ConversationConfig | None = None,
    ) -> None:
        self.config = config or ConversationConfig()
        self.graph_engine = graph_engine or KnowledgeGraphEngine()
        self._events_by_conversation: dict[str, list[ConversationEvent]] = defaultdict(list)
        self._turns_by_conversation: dict[str, list[ConversationTurn]] = defaultdict(list)
        self._canonical_by_conversation: dict[str, list[CanonicalMemory]] = defaultdict(list)
        self._summaries_by_conversation: dict[str, list[str]] = defaultdict(list)

    def store_event(self, event: ConversationEvent) -> None:
        bucket = self._events_by_conversation[event.conversation_id]
        bucket.append(event)
        overflow = len(bucket) - self.config.retention.max_events_per_conversation
        if overflow > 0:
            del bucket[:overflow]

    def store_turn_view(self, turn: ConversationTurn) -> None:
        bucket = self._turns_by_conversation[turn.conversation_id]
        bucket.append(turn)
        overflow = len(bucket) - self.config.retention.max_events_per_conversation
        if overflow > 0:
            del bucket[:overflow]

    def store_canonical_memory(self, memory: CanonicalMemory) -> None:
        bucket = self._canonical_by_conversation[memory.conversation_id]
        bucket.append(memory)
        if self.config.indexing.enable_graph_projection and memory.status.value == "accepted":
            statement = memory.as_statement()
            self.graph_engine.add_statement(
                statement.subject,
                statement.relation,
                statement.object,
                context=statement.context,
                confidence=statement.confidence,
                source=statement.source,
                timestamp=statement.timestamp,
            )
        overflow = len(bucket) - self.config.retention.max_canonical_memories_per_conversation
        if overflow > 0:
            del bucket[:overflow]

    def get_events(self, conversation_id: str) -> list[ConversationEvent]:
        return list(self._events_by_conversation.get(conversation_id, []))

    def latest_events(self, conversation_id: str, limit: int = 5) -> list[ConversationEvent]:
        events = self._events_by_conversation.get(conversation_id, [])
        return list(events[-limit:]) if limit > 0 else []

    def get_turns(self, conversation_id: str) -> list[ConversationTurn]:
        return list(self._turns_by_conversation.get(conversation_id, []))

    def latest_turns(self, conversation_id: str, limit: int = 5) -> list[ConversationTurn]:
        turns = self._turns_by_conversation.get(conversation_id, [])
        return list(turns[-limit:]) if limit > 0 else []

    def get_canonical_memories(self, conversation_id: str) -> list[CanonicalMemory]:
        return list(self._canonical_by_conversation.get(conversation_id, []))

    def memory_history(self, conversation_id: str) -> list[CanonicalMemory]:
        return self.get_canonical_memories(conversation_id)

    def facts_for_conversation(self, conversation_id: str) -> list[Statement]:
        return [memory.as_statement() for memory in self.get_canonical_memories(conversation_id) if memory.status.value == "accepted"]

    def evidence_lookup(self, conversation_id: str, memory_ids: list[str]) -> EvidenceSet:
        memories = [memory for memory in self.get_canonical_memories(conversation_id) if memory.memory_id in set(memory_ids)]
        return EvidenceSet(
            query="",
            rewritten_query="",
            supporting_memories=memories,
            supporting_facts=[memory.as_statement() for memory in memories],
        )

    def all_conversation_ids(self) -> list[str]:
        return sorted(set(self._events_by_conversation) | set(self._canonical_by_conversation))
