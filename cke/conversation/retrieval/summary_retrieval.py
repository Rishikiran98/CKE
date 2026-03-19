"""Summary retrieval adapter."""

from __future__ import annotations

from cke.conversation.memory_store import ConversationMemoryStore


class SummaryRetriever:
    """Retrieve conversation summaries when available."""

    def __init__(self, memory_store: ConversationMemoryStore) -> None:
        self.memory_store = memory_store

    def retrieve(self, conversation_id: str, *, limit: int = 3) -> list[str]:
        summaries = getattr(self.memory_store, "_summaries_by_conversation", {}).get(
            conversation_id, []
        )
        return list(summaries[-limit:])
