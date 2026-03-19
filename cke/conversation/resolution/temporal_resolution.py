"""Temporal follow-up resolution using retrieved evidence and recent memories."""

from __future__ import annotations

import re

from cke.conversation.config import ResolutionConfig
from cke.conversation.memory_store import ConversationMemoryStore
from cke.conversation.patterns import extract_date_phrase
from cke.conversation.types import RetrievedMemory


class TemporalReferenceResolver:
    """Resolve underspecified temporal references such as 'that' or 'again'."""

    def __init__(
        self,
        memory_store: ConversationMemoryStore,
        config: ResolutionConfig | None = None,
    ) -> None:
        self.memory_store = memory_store
        self.config = config or ResolutionConfig()

    def resolve(
        self,
        query: str,
        *,
        conversation_id: str,
        retrieved_turns: list[RetrievedMemory] | None = None,
    ) -> tuple[str, dict[str, str]]:
        rewritten = query
        bindings: dict[str, str] = {}
        if not any(
            token in query.lower() for token in self.config.temporal_reference_tokens
        ):
            return rewritten, bindings
        latest_date = None
        for turn in list(self.memory_store.latest_turns(conversation_id, limit=6))[
            ::-1
        ]:
            latest_date = extract_date_phrase(turn.text)
            if latest_date:
                break
        if not latest_date and retrieved_turns:
            for turn in retrieved_turns:
                latest_date = extract_date_phrase(turn.text)
                if latest_date:
                    break
        if latest_date and re.search(r"that|it", rewritten, flags=re.IGNORECASE):
            rewritten = re.sub(
                r"that|it", latest_date, rewritten, count=1, flags=re.IGNORECASE
            )
            bindings["temporal_reference"] = latest_date
        return rewritten, bindings
