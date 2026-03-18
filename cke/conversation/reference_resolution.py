"""Conversational reference resolution over recent and retrieved memory."""

from __future__ import annotations

import re

from cke.conversation.memory import ConversationalMemoryStore
from cke.conversation.types import RetrievedMemory


class ConversationalReferenceResolver:
    """Expand shorthand references using recent turns and retrieved evidence."""

    _PRONOUNS = {"it", "they", "them", "that", "those", "he", "she"}
    _ROLE_PATTERN = (
        r"\b((?:backend|frontend|platform|infra|"
        r"full stack|full-stack|ml|data)"
        r"\s+(?:engineer|developer)(?:\s+role|\s+position)?|"
        r"(?:backend|frontend|platform|infra|"
        r"full stack|full-stack|ml|data)"
        r"\s+roles?)\b"
    )

    def __init__(self, memory_store: ConversationalMemoryStore) -> None:
        self.memory_store = memory_store

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

        replacements = {
            "that company": latest_company,
            "the company": latest_company,
            "that recruiter": latest_person,
            "the recruiter": latest_person,
            "that role": latest_role,
            "the role": latest_role,
        }
        for phrase, value in replacements.items():
            if value and phrase in rewritten.lower():
                rewritten = re.sub(phrase, value, rewritten, flags=re.IGNORECASE)
                bindings[phrase] = value

        if (
            latest_date
            and re.search(r"\bthat\b", rewritten, flags=re.IGNORECASE)
            and any(token in rewritten.lower() for token in ["when", "date", "again"])
        ):
            rewritten = re.sub(
                r"\bthat\b", latest_date, rewritten, count=1, flags=re.IGNORECASE
            )
            bindings["that"] = latest_date
        elif latest_company and re.search(
            r"\b(it|they|them)\b", rewritten, flags=re.IGNORECASE
        ):
            rewritten = re.sub(
                r"\b(it|they|them)\b",
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
        for turn in list(recent_turns)[::-1]:
            match = re.search(
                self._ROLE_PATTERN,
                turn.text,
                flags=re.IGNORECASE,
            )
            if match:
                return match.group(1)
        for hit in retrieved_turns:
            match = re.search(
                self._ROLE_PATTERN,
                hit.text,
                flags=re.IGNORECASE,
            )
            if match:
                return match.group(1)
        return None

    def _latest_date(self, recent_turns, retrieved_turns) -> str | None:
        patterns = [
            r"\b(?:Monday|Tuesday|Wednesday|Thursday|Friday|Saturday|Sunday)\b",
            (
                r"\b(?:January|February|March|April|May|June|July|August|"
                r"September|October|November|December)\s+\d{1,2}(?:st|nd|rd|th)?\b"
            ),
            r"\b\d{4}-\d{2}-\d{2}\b",
            r"\bnext\s+(?:week|month|Monday|Tuesday|Wednesday|Thursday|Friday)\b",
        ]
        for item in list(recent_turns)[::-1]:
            for pattern in patterns:
                match = re.search(pattern, item.text, flags=re.IGNORECASE)
                if match:
                    return match.group(0)
        for item in retrieved_turns:
            for pattern in patterns:
                match = re.search(pattern, item.text, flags=re.IGNORECASE)
                if match:
                    return match.group(0)
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
