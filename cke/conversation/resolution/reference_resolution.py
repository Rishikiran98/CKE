"""Evidence-aware reference resolution for conversational retrieval."""

from __future__ import annotations

import re

from cke.conversation.config import ResolutionConfig
from cke.conversation.memory_store import ConversationMemoryStore
from cke.conversation.types import RetrievedMemory
from .alias_resolution import AliasResolver
from .temporal_resolution import TemporalReferenceResolver


class ConversationalReferenceResolver:
    """Resolve aliases, pronouns, and underspecified references using evidence."""

    def __init__(
        self,
        memory_store: ConversationMemoryStore,
        config: ResolutionConfig | None = None,
    ) -> None:
        self.memory_store = memory_store
        self.config = config or ResolutionConfig()
        self.alias_resolver = AliasResolver(memory_store)
        self.temporal_resolver = TemporalReferenceResolver(memory_store, self.config)

    def rewrite(
        self,
        query: str,
        *,
        conversation_id: str,
        retrieved_turns: list[RetrievedMemory] | None = None,
    ) -> tuple[str, dict[str, str]]:
        rewritten, bindings = self.alias_resolver.resolve(
            query, conversation_id=conversation_id
        )
        rewritten, temporal_bindings = self.temporal_resolver.resolve(
            rewritten,
            conversation_id=conversation_id,
            retrieved_turns=retrieved_turns,
        )
        bindings.update(temporal_bindings)

        focus = self._latest_entity(conversation_id, retrieved_turns or [])
        latest_role = self._latest_role(conversation_id, retrieved_turns or [])
        if latest_role and re.search(
            r"\b(?:that|this)\s+role\b", rewritten, flags=re.IGNORECASE
        ):
            rewritten = re.sub(
                r"\b(?:that|this)\s+role\b",
                latest_role,
                rewritten,
                flags=re.IGNORECASE,
            )
            bindings.setdefault("role_reference", latest_role)
        if focus and self._contains_generic_reference(rewritten):
            rewritten = re.sub(
                r"\b(?:that|this)\s+(?:company|person|place|thing)\b",
                focus,
                rewritten,
                flags=re.IGNORECASE,
            )
            bindings.setdefault("generic_reference", focus)
        elif focus:
            pronoun_pattern = r"\b(?:" + "|".join(self.config.pronouns) + r")\b"
            if re.search(pronoun_pattern, rewritten, flags=re.IGNORECASE):
                rewritten = re.sub(
                    pronoun_pattern,
                    focus,
                    rewritten,
                    count=1,
                    flags=re.IGNORECASE,
                )
                bindings.setdefault("pronoun", focus)
        return rewritten, bindings

    def _contains_generic_reference(self, query: str) -> bool:
        return bool(
            re.search(
                r"\b(?:that|this)\s+(?:company|person|place|thing)\b",
                query,
                flags=re.IGNORECASE,
            )
        )

    def _latest_entity(
        self, conversation_id: str, retrieved_turns: list[RetrievedMemory]
    ) -> str | None:
        ignored = {"The", "A", "An", "This", "That"}
        for memory in reversed(
            self.memory_store.get_canonical_memories(conversation_id)
        ):
            if memory.subject != "user" and memory.subject not in ignored:
                return memory.subject
            if (
                memory.object
                and memory.object[:1].isupper()
                and memory.object not in ignored
            ):
                return memory.object
        for turn in reversed(self.memory_store.latest_turns(conversation_id, limit=6)):
            for entity in reversed(turn.entities):
                if entity not in ignored:
                    return entity
        for hit in retrieved_turns:
            if hit.subject and hit.subject != "user" and hit.subject not in ignored:
                return hit.subject
        return None

    def _latest_role(
        self, conversation_id: str, retrieved_turns: list[RetrievedMemory]
    ) -> str | None:
        pattern = re.compile(
            r"\b((?:backend|frontend|platform|systems|software|data|product|design)\s+"
            r"(?:engineer|developer|designer|manager)(?:\s+role|\s+position)?)\b",
            flags=re.IGNORECASE,
        )
        for turn in reversed(self.memory_store.latest_turns(conversation_id, limit=6)):
            match = pattern.search(turn.text)
            if match:
                return match.group(1)
        for hit in retrieved_turns:
            match = pattern.search(hit.text)
            if match:
                return match.group(1)
        return None
