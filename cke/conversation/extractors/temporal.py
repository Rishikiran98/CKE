
"""Temporal candidate extraction for dates and time references."""

from __future__ import annotations

import re
from uuid import uuid4

from cke.conversation.patterns import extract_date_phrases, normalize_text_token
from cke.conversation.types import (
    CandidateMemory,
    ConfidenceBand,
    ConversationEvent,
    MemoryKind,
    MemorySourceSpan,
)


class TemporalMemoryExtractor:
    """Extract date-bearing memory proposals with explicit provenance."""

    name = "temporal"

    def extract(self, event: ConversationEvent) -> list[CandidateMemory]:
        candidates: list[CandidateMemory] = []
        for start, end, value in extract_date_phrases(event.text):
            sentence = self._sentence_for_span(event.text, start, end)
            subject = self._infer_subject(sentence)
            relation = "occurs_on"
            confidence = 0.72 if subject != "conversation" else 0.48
            candidates.append(
                CandidateMemory(
                    candidate_id=f"cand-{uuid4().hex[:12]}",
                    conversation_id=event.conversation_id,
                    event_id=event.event_id,
                    turn_id=event.turn_id,
                    kind=MemoryKind.TEMPORAL,
                    subject=subject,
                    relation=relation,
                    object=normalize_text_token(value),
                    confidence=confidence,
                    confidence_band=ConfidenceBand.HIGH if confidence >= 0.7 else ConfidenceBand.MEDIUM,
                    provenance=[MemorySourceSpan(turn_id=event.turn_id, start=start, end=end, text=value, extractor=self.name)],
                    attributes={"sentence": sentence},
                )
            )
        return candidates

    def _sentence_for_span(self, text: str, start: int, end: int) -> str:
        left = max(text.rfind(".", 0, start), text.rfind("!", 0, start), text.rfind("?", 0, start))
        right_candidates = [idx for idx in (text.find(".", end), text.find("!", end), text.find("?", end)) if idx != -1]
        right = min(right_candidates) if right_candidates else len(text)
        return normalize_text_token(text[left + 1:right])

    def _infer_subject(self, sentence: str) -> str:
        match = re.search(
            r"(?:the\s+)?([A-Z][A-Za-z0-9&./-]*(?:\s+[A-Z][A-Za-z0-9&./-]*)*|[a-z][a-z0-9_-]*(?:\s+[a-z][a-z0-9_-]*)*)\s+(?:is|was|on|for|scheduled|happens)",
            sentence,
        )
        if match:
            return normalize_text_token(match.group(1))
        return "conversation"
