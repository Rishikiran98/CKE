
"""Compatibility wrapper for candidate extraction."""

from __future__ import annotations

from dataclasses import dataclass

from cke.conversation.extractors import HeuristicMemoryExtractor, TemporalMemoryExtractor
from cke.conversation.types import ConversationEvent


@dataclass(slots=True)
class TurnExtraction:
    entities: list[str]
    facts: list


class ConversationalTurnExtractor:
    """Backward-compatible adapter over pluggable candidate extractors."""

    def __init__(self) -> None:
        self.extractors = [TemporalMemoryExtractor(), HeuristicMemoryExtractor()]

    def extract(self, text: str, *, role: str = "user") -> TurnExtraction:
        event = ConversationEvent(
            conversation_id="compat",
            event_id="compat-event",
            turn_id="compat-turn",
            turn_order=1,
            role=role,
            text=text,
            timestamp="",
        )
        candidates = []
        for extractor in self.extractors:
            candidates.extend(extractor.extract(event))
        entities = sorted({candidate.object for candidate in candidates if candidate.relation == "mentions"})
        facts = [candidate.as_statement() for candidate in candidates if candidate.relation != "mentions"]
        return TurnExtraction(entities=entities, facts=facts)
