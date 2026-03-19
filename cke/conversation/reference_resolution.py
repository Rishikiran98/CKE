"""Conversational reference resolution over recent and retrieved memory."""

from __future__ import annotations

import re

from cke.conversation.config import ReferenceResolutionConfig
from cke.conversation.memory import ConversationalMemoryStore
from cke.conversation.patterns import ROLE_PATTERN, extract_date_phrase
from cke.conversation.types import RetrievedMemory


class ConversationalReferenceResolver:
    """Expand shorthand references using recent turns and retrieved evidence."""

    def __init__(
        self,
        memory_store: ConversationalMemoryStore,
        config: ReferenceResolutionConfig | None = None,
    ) -> None:
        self.memory_store = memory_store
        self.config = config or ReferenceResolutionConfig()

    def rewrite(
        self,
        query: str,
        *,
        conversation_id: str,
        retrieved_turns: list[RetrievedMemory] | None = None,
    ) -> tuple[str, dict[str, str]]:
        rewritten = query
        bindings: dict[str, str] = {}
        recent_turns = self.memory_store.latest_turns(conversation_id, limit=6)
        retrieved_turns = retrieved_turns or []

        latest_company = self._latest_matching_entity(
            recent_turns, retrieved_turns, kind="company"
        )
        latest_person = self._latest_matching_entity(
            recent_turns, retrieved_turns, kind="person"
        )
        latest_role = self._latest_role(recent_turns, retrieved_turns)
        latest_date = self._latest_date(recent_turns, retrieved_turns)

        explicit_values = {
            "company": latest_company,
            "person": latest_person,
            "role": latest_role,
        }
        for phrase, kind in self.config.explicit_reference_map.items():
            value = explicit_values.get(kind)
            if value and phrase in rewritten.lower():
                rewritten = re.sub(phrase, value, rewritten, flags=re.IGNORECASE)
                bindings[phrase] = value

        if (
            latest_date
            and re.search(r"\bthat\b", rewritten, flags=re.IGNORECASE)
            and any(
                token in rewritten.lower() for token in self.config.date_context_tokens
            )
        ):
            rewritten = re.sub(
                r"\bthat\b", latest_date, rewritten, count=1, flags=re.IGNORECASE
            )
            bindings["that"] = latest_date
        elif latest_company:
            pronoun_pattern = r"\b(" + "|".join(self.config.company_pronouns) + r")\b"
            if re.search(pronoun_pattern, rewritten, flags=re.IGNORECASE):
                rewritten = re.sub(
                    pronoun_pattern,
                    latest_company,
                    rewritten,
                    count=1,
                    flags=re.IGNORECASE,
                )
                bindings["pronoun"] = latest_company

        return rewritten, bindings

    def _latest_matching_entity(
        self, recent_turns, retrieved_turns, *, kind: str
    ) -> str | None:
        candidates: list[str] = []
        for turn in list(recent_turns)[::-1]:
            candidates.extend(turn.entities)
        for hit in retrieved_turns:
            candidates.extend(hit.entities)
        for candidate in candidates:
            if kind == "company" and self._looks_like_company(candidate):
                return candidate
            if kind == "person" and self._looks_like_person(candidate):
                return candidate
        return None

    def _latest_role(self, recent_turns, retrieved_turns) -> str | None:
        for item in [*list(recent_turns)[::-1], *retrieved_turns]:
            match = re.search(ROLE_PATTERN, item.text, flags=re.IGNORECASE)
            if match:
                return match.group(1)
        return None

    def _latest_date(self, recent_turns, retrieved_turns) -> str | None:
        for item in [*list(recent_turns)[::-1], *retrieved_turns]:
            date_value = extract_date_phrase(item.text)
            if date_value:
                return date_value
        return None

    def _looks_like_company(self, text: str) -> bool:
        cleaned = text.strip()
        if cleaned.lower() in {"recruiter", "hiring manager", "backend roles"}:
            return False
        if cleaned in {"The", "A", "An", "My", "Our"}:
            return False
        return bool(
            re.match(r"^[A-Z][A-Za-z0-9&.-]+(?:\s+[A-Z][A-Za-z0-9&.-]+)*$", cleaned)
        )

    def _looks_like_person(self, text: str) -> bool:
        return bool(re.match(r"^[A-Z][a-z]+\s+[A-Z][a-z]+$", text.strip()))
