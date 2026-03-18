"""Heuristic conversational extraction for entities, facts, and updates."""

from __future__ import annotations

import re
from dataclasses import dataclass

from cke.models import Statement


@dataclass(slots=True)
class TurnExtraction:
    entities: list[str]
    facts: list[Statement]


class ConversationalTurnExtractor:
    """Extract lightweight conversational memory from free-form user turns.

    The goal is not perfect IE. Instead we preserve raw text and add enough
    structure for retrieval, preference tracking, updates, and graph support.
    """

    _MONTHS = (
        "january|february|march|april|may|june|july|august|"
        "september|october|november|december"
    )
    _WEEKDAYS = "monday|tuesday|wednesday|thursday|friday|saturday|sunday"

    def extract(self, text: str, *, role: str = "user") -> TurnExtraction:
        entities = self._extract_entities(text)
        facts = self._extract_facts(text=text, role=role, entities=entities)
        return TurnExtraction(entities=entities, facts=facts)

    def _extract_entities(self, text: str) -> list[str]:
        found: list[str] = []
        patterns = [
            r"\b(?:[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)\b",
            r"\b[A-Z]{2,}\b",
        ]
        for pattern in patterns:
            for match in re.findall(pattern, text):
                cleaned = match.strip(" .,!?:;\"'")
                if cleaned.lower() in {"i", "we", "they", "it", "he", "she"}:
                    continue
                if cleaned and cleaned not in found:
                    found.append(cleaned)

        nounish_patterns = [
            r"\b(backend roles?|frontend roles?|platform roles?|infra roles?)\b",
            r"\b(Apple interview|Google interview|Meta interview|Stripe interview)\b",
            r"\b(recruiter|hiring manager|onsite|phone screen|take-home)\b",
        ]
        lowered = text.lower()
        for pattern in nounish_patterns:
            for match in re.findall(pattern, lowered, flags=re.IGNORECASE):
                cleaned = str(match).strip()
                if cleaned and cleaned not in found:
                    found.append(cleaned)
        return found

    def _extract_facts(
        self, *, text: str, role: str, entities: list[str]
    ) -> list[Statement]:
        if role != "user":
            return []

        facts: list[Statement] = []
        lowered = text.lower()
        date_value = self._extract_date_phrase(text)
        company = self._extract_company(text)
        role_name = self._extract_role(text)
        person = self._extract_person(text)

        def add(subject: str, relation: str, object_: str, **context) -> None:
            if not subject or not object_:
                return
            facts.append(
                Statement(
                    subject=subject,
                    relation=relation,
                    object=object_,
                    context=context,
                    confidence=0.88,
                )
            )

        if company and "interview" in lowered:
            add("user", "mentioned_interview", company, topic="interview")
        if (
            company
            and date_value
            and any(token in lowered for token in ["interview", "onsite", "screen"])
        ):
            add(company, "scheduled_for", date_value, topic="timeline")
        if (
            company
            and role_name
            and any(token in lowered for token in ["role", "position", "job", "apply"])
        ):
            add(company, "has_role", role_name, topic="job_search")
            add("user", "interested_in_role", role_name, topic="job_search")
        if company and any(
            token in lowered for token in ["apply", "applying", "application"]
        ):
            add("user", "applied_to", company, topic="job_search")
        has_pending_reply = company and any(
            token in lowered
            for token in [
                "hasn't replied",
                "hasnt replied",
                "didn't reply",
                "didnt reply",
                "waiting on",
                "still waiting",
            ]
        )
        if has_pending_reply:
            add(company, "reply_status", "pending", topic="communication")
        if (
            company
            and not has_pending_reply
            and any(token in lowered for token in ["replied", "got back", "responded"])
        ):
            add(company, "reply_status", "replied", topic="communication")
        if person and company and "recruiter" in lowered:
            add(person, "recruiter_for", company, topic="communication")
            add(company, "has_recruiter", person, topic="communication")
        if any(
            token in lowered
            for token in ["prefer", "preferred", "i'm leaning", "i am leaning"]
        ):
            preference = self._extract_preference(text)
            if preference:
                add("user", "prefers", preference, topic="preference")
        if company and any(
            token in lowered for token in ["remote", "hybrid", "onsite"]
        ):
            work_mode = self._extract_work_mode(lowered)
            if work_mode:
                add(company, "work_mode", work_mode, topic="job_search")
        if company and any(
            token in lowered for token in ["salary", "comp", "pays", "pay"]
        ):
            compensation = self._extract_compensation(text)
            if compensation:
                add(company, "compensation", compensation, topic="job_search")
        if company and any(
            token in lowered for token in ["backend", "frontend", "platform", "infra"]
        ):
            role_flavor = self._extract_role_flavor(lowered)
            if role_flavor:
                add(company, "role_focus", role_flavor, topic="job_search")
        if company and any(
            token in lowered for token in ["faster", "slower", "moved", "rescheduled"]
        ):
            status = self._extract_status_phrase(text)
            if status:
                add(company, "process_update", status, topic="timeline")

        for entity in entities:
            add("turn", "mentions", entity, topic="entity_mention")

        return self._dedupe(facts)

    def _extract_company(self, text: str) -> str | None:
        patterns = [
            (
                r"\b(?:at|with|from|for)\s+"
                r"([A-Z][A-Za-z0-9&.-]+(?:\s+[A-Z][A-Za-z0-9&.-]+)*)"
            ),
            (
                r"\b([A-Z][A-Za-z0-9&.-]+(?:\s+[A-Z][A-Za-z0-9&.-]+)*)\s+"
                r"(?:interview|role|recruiter|onsite|screen)"
            ),
        ]
        for pattern in patterns:
            match = re.search(pattern, text)
            if match:
                candidate = match.group(1).strip(" .,!?:;")
                if candidate.lower() not in {"I", "My"}:
                    return candidate
        return None

    def _extract_person(self, text: str) -> str | None:
        match = re.search(r"\b([A-Z][a-z]+\s+[A-Z][a-z]+)\b", text)
        if match:
            return match.group(1)
        return None

    def _extract_role(self, text: str) -> str | None:
        match = re.search(
            (
                r"\b((?:backend|frontend|platform|infra|full stack|full-stack|ml|data)"
                r"\s+(?:engineer|developer)(?:\s+role|\s+position)?|"
                r"(?:backend|frontend|platform|infra|full stack|full-stack|ml|data)"
                r"\s+roles?)\b"
            ),
            text,
            flags=re.IGNORECASE,
        )
        if match:
            return match.group(1)
        return None

    def _extract_role_flavor(self, lowered: str) -> str | None:
        for token in [
            "backend",
            "frontend",
            "platform",
            "infra",
            "full stack",
            "full-stack",
        ]:
            if token in lowered:
                return token.replace("full stack", "full-stack")
        return None

    def _extract_work_mode(self, lowered: str) -> str | None:
        for token in ["remote", "hybrid", "onsite"]:
            if token in lowered:
                return token
        return None

    def _extract_preference(self, text: str) -> str | None:
        match = re.search(
            r"\bprefer(?:red)?\s+([^.,;!?]+)",
            text,
            flags=re.IGNORECASE,
        )
        if match:
            return match.group(1).strip(" .,!?:;")
        for token in [
            "backend roles",
            "frontend roles",
            "platform roles",
            "remote work",
        ]:
            if token in text.lower():
                return token
        return None

    def _extract_date_phrase(self, text: str) -> str | None:
        patterns = [
            rf"\b(?:{self._WEEKDAYS})\b",
            rf"\b(?:{self._MONTHS})\s+\d{{1,2}}(?:st|nd|rd|th)?\b",
            r"\b\d{4}-\d{2}-\d{2}\b",
            r"\bnext\s+(?:week|month|monday|tuesday|wednesday|thursday|friday)\b",
        ]
        for pattern in patterns:
            match = re.search(pattern, text, flags=re.IGNORECASE)
            if match:
                return match.group(0)
        return None

    def _extract_compensation(self, text: str) -> str | None:
        match = re.search(r"\$\d[\d,]*(?:k|K)?", text)
        if match:
            return match.group(0)
        return None

    def _extract_status_phrase(self, text: str) -> str | None:
        lowered = text.lower()
        if "rescheduled" in lowered:
            return "rescheduled"
        if "moved faster" in lowered:
            return "moved faster"
        if "slowed down" in lowered:
            return "slowed down"
        return None

    def _dedupe(self, facts: list[Statement]) -> list[Statement]:
        seen: set[tuple[str, str, str]] = set()
        out: list[Statement] = []
        for fact in facts:
            key = (fact.subject, fact.relation, fact.object)
            if key in seen:
                continue
            seen.add(key)
            out.append(fact)
        return out
