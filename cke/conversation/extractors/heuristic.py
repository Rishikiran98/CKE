"""Open-domain heuristic extractor used as a fallback candidate proposer."""

from __future__ import annotations

import re
from uuid import uuid4

from cke.conversation.patterns import normalize_text_token
from cke.conversation.types import (
    CandidateMemory,
    ConfidenceBand,
    ConversationEvent,
    MemoryKind,
    MemorySourceSpan,
)


class HeuristicMemoryExtractor:
    """Fallback rule-based extractor that emits proposals instead of truth."""

    name = "heuristic"

    _FIRST_PERSON_PATTERNS: tuple[tuple[str, MemoryKind, float], ...] = (
        (r"\bI\s+(?:prefer|preferred)\s+([^.;!?]+)", MemoryKind.PREFERENCE, 0.68),
        (
            r"\bI\s+(?:like|love|enjoy|need|want|plan(?:\s+to)?)\s+([^.;!?]+)",
            MemoryKind.OBSERVATION,
            0.52,
        ),
        (r"\bI\s+am\s+([^.;!?]+)", MemoryKind.OBSERVATION, 0.46),
        (r"\bmy\s+([^.;!?]+?)\s+is\s+([^.;!?]+)", MemoryKind.FACT, 0.54),
    )
    _THIRD_PERSON_PATTERNS: tuple[tuple[str, str, MemoryKind, float], ...] = (
        (
            r"\b([A-Z][A-Za-z0-9&./-]*(?:\s+[A-Z][A-Za-z0-9&./-]*)*)\s+is\s+([^.;!?]+)",
            "is",
            MemoryKind.FACT,
            0.58,
        ),
        (
            (
                r"\b([A-Z][A-Za-z0-9&./-]*"
                r"(?:\s+[A-Z][A-Za-z0-9&./-]*)*)\s+has\s+([^.;!?]+)"
            ),
            "has",
            MemoryKind.FACT,
            0.56,
        ),
        (
            (
                r"\b([A-Z][A-Za-z0-9&./-]*"
                r"(?:\s+[A-Z][A-Za-z0-9&./-]*)*)\s+"
                r"(?:hasn't|hasnt|didn't|didnt)\s+replied\b"
            ),
            "reply_status",
            MemoryKind.STATUS,
            0.72,
        ),
        (
            (
                r"\b([A-Z][A-Za-z0-9&./-]*"
                r"(?:\s+[A-Z][A-Za-z0-9&./-]*)*)\s+(?:already\s+)?replied\b"
            ),
            "reply_status",
            MemoryKind.STATUS,
            0.7,
        ),
    )
    _ENTITY_PATTERN = re.compile(
        r"\b(?:[A-Z][A-Za-z0-9&./-]+(?:\s+[A-Z][A-Za-z0-9&./-]+)*)\b"
    )
    _ROLE_PATTERN = re.compile(
        r"\b((?:backend|frontend|platform|systems|software|data|product|design)\s+"
        r"(?:engineer|developer|designer|manager)(?:\s+role|\s+position)?)\b",
        flags=re.IGNORECASE,
    )

    def extract(self, event: ConversationEvent) -> list[CandidateMemory]:
        if not event.text.strip():
            return []
        candidates: list[CandidateMemory] = []
        text = event.text
        lowered = text.lower()

        for pattern, kind, confidence in self._FIRST_PERSON_PATTERNS:
            for match in re.finditer(pattern, text, flags=re.IGNORECASE):
                if kind is MemoryKind.PREFERENCE:
                    relation = "prefers"
                    subject = "user"
                    object_ = normalize_text_token(match.group(1))
                elif pattern.startswith(r"\bmy"):
                    subject = "user"
                    relation = normalize_text_token(match.group(1))
                    object_ = normalize_text_token(match.group(2))
                else:
                    subject = "user"
                    relation = "describes"
                    object_ = normalize_text_token(match.group(1))
                candidates.append(
                    self._candidate(
                        event,
                        subject,
                        relation,
                        object_,
                        kind,
                        confidence,
                        match.start(),
                        match.end(),
                        match.group(0),
                    )
                )

        for pattern, relation, kind, confidence in self._THIRD_PERSON_PATTERNS:
            for match in re.finditer(pattern, text):
                subject = normalize_text_token(match.group(1))
                object_ = (
                    "pending"
                    if relation == "reply_status"
                    and "replied" not in lowered[match.start() : match.end()]
                    else "replied"
                )
                if relation in {"is", "has"}:
                    object_ = normalize_text_token(match.group(2))
                candidates.append(
                    self._candidate(
                        event,
                        subject,
                        relation,
                        object_,
                        kind,
                        confidence,
                        match.start(),
                        match.end(),
                        match.group(0),
                    )
                )

        company = self._first_entity(text)
        role_match = self._ROLE_PATTERN.search(text)
        if company and "interview" in lowered:
            start = text.lower().find("interview")
            end = start + len("interview") if start >= 0 else len(text)
            candidates.append(
                self._candidate(
                    event,
                    "user",
                    "mentioned_interview",
                    company,
                    MemoryKind.FACT,
                    0.64,
                    max(0, start - len(company)),
                    end,
                    text[max(0, start - len(company)) : end] if start >= 0 else text,
                )
            )
        if "replied" in lowered:
            from_match = re.search(
                r"\bfrom\s+([A-Z][A-Za-z0-9&./-]+(?:\s+[A-Z][A-Za-z0-9&./-]+)*)", text
            )
            status_subject = (
                normalize_text_token(from_match.group(1)) if from_match else company
            )
            if status_subject:
                status = (
                    "pending"
                    if any(
                        token in lowered
                        for token in (
                            "hasn't replied",
                            "hasnt replied",
                            "didn't replied",
                            "didnt replied",
                            "still hasn't replied",
                            "still hasnt replied",
                        )
                    )
                    else "replied"
                )
                status_start = lowered.find("replied")
                candidates.append(
                    self._candidate(
                        event,
                        status_subject,
                        "reply_status",
                        status,
                        MemoryKind.STATUS,
                        0.72 if status == "pending" else 0.68,
                        max(0, status_start - len(status_subject)),
                        (
                            status_start + len("replied")
                            if status_start >= 0
                            else len(text)
                        ),
                        (
                            text[
                                max(
                                    0, status_start - len(status_subject)
                                ) : status_start
                                + len("replied")
                            ]
                            if status_start >= 0
                            else text
                        ),
                    )
                )
        if company and role_match:
            candidates.append(
                self._candidate(
                    event,
                    company,
                    "has_role",
                    normalize_text_token(role_match.group(1)),
                    MemoryKind.FACT,
                    0.66,
                    role_match.start(),
                    role_match.end(),
                    role_match.group(0),
                )
            )
        if company and "interview" in lowered:
            temporal_match = re.search(
                r"\b(next\s+\w+|tomorrow|today|yesterday|[A-Z][a-z]+\s+\d{1,2})\b",
                text,
                flags=re.IGNORECASE,
            )
            if temporal_match:
                candidates.append(
                    self._candidate(
                        event,
                        company,
                        "scheduled_for",
                        normalize_text_token(temporal_match.group(1)),
                        MemoryKind.TEMPORAL,
                        0.7,
                        temporal_match.start(),
                        temporal_match.end(),
                        temporal_match.group(1),
                    )
                )

        entities: list[str] = []
        for match in self._ENTITY_PATTERN.finditer(text):
            entity = normalize_text_token(match.group(0))
            if entity and entity not in {"I"} and entity not in entities:
                entities.append(entity)
                candidates.append(
                    self._candidate(
                        event,
                        "turn",
                        "mentions",
                        entity,
                        MemoryKind.OBSERVATION,
                        0.35,
                        match.start(),
                        match.end(),
                        match.group(0),
                        attributes={"entity": True},
                    )
                )
        return candidates

    def _first_entity(self, text: str) -> str | None:
        for match in self._ENTITY_PATTERN.finditer(text):
            entity = normalize_text_token(match.group(0))
            if entity not in {"I", "The", "A", "An"}:
                return entity
        return None

    def _candidate(
        self,
        event: ConversationEvent,
        subject: str,
        relation: str,
        object_: str,
        kind: MemoryKind,
        confidence: float,
        start: int,
        end: int,
        span_text: str,
        *,
        attributes: dict | None = None,
    ) -> CandidateMemory:
        band = (
            ConfidenceBand.HIGH
            if confidence >= 0.7
            else ConfidenceBand.MEDIUM if confidence >= 0.45 else ConfidenceBand.LOW
        )
        return CandidateMemory(
            candidate_id=f"cand-{uuid4().hex[:12]}",
            conversation_id=event.conversation_id,
            event_id=event.event_id,
            turn_id=event.turn_id,
            kind=kind,
            subject=normalize_text_token(subject),
            relation=relation,
            object=normalize_text_token(object_),
            confidence=confidence,
            confidence_band=band,
            provenance=[
                MemorySourceSpan(
                    turn_id=event.turn_id,
                    start=start,
                    end=end,
                    text=span_text,
                    extractor=self.name,
                )
            ],
            attributes=dict(attributes or {}),
        )
