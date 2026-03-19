
"""Fact retrieval over canonical conversational memories."""

from __future__ import annotations

from cke.conversation.memory_store import ConversationMemoryStore
from cke.conversation.types import CanonicalMemory
from cke.models import Statement


class FactRetriever:
    """Return canonical memories and their statement projections."""

    def __init__(self, memory_store: ConversationMemoryStore) -> None:
        self.memory_store = memory_store

    def retrieve(self, conversation_id: str, *, limit: int = 10) -> tuple[list[CanonicalMemory], list[Statement]]:
        memories = [memory for memory in self.memory_store.get_canonical_memories(conversation_id) if memory.status.value == "accepted"]
        memories = sorted(memories, key=lambda item: (item.last_seen_at or "", item.mention_count), reverse=True)
        selected = memories[:limit]
        return selected, [memory.as_statement() for memory in selected]
