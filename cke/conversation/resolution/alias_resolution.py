
"""Alias resolution from canonical memories and recent entities."""

from __future__ import annotations

from cke.conversation.memory_store import ConversationMemoryStore
from cke.conversation.patterns import normalize_text_token


class AliasResolver:
    """Resolve explicit aliases using canonical alias memories when present."""

    def __init__(self, memory_store: ConversationMemoryStore) -> None:
        self.memory_store = memory_store

    def resolve(self, query: str, *, conversation_id: str) -> tuple[str, dict[str, str]]:
        rewritten = query
        bindings: dict[str, str] = {}
        for memory in self.memory_store.get_canonical_memories(conversation_id):
            if memory.kind.value != "alias":
                continue
            for alias in memory.aliases + [memory.object]:
                alias_text = normalize_text_token(alias)
                if alias_text and alias_text.lower() in rewritten.lower() and alias_text != memory.subject:
                    rewritten = rewritten.replace(alias_text, memory.subject)
                    bindings[alias_text] = memory.subject
        return rewritten, bindings
