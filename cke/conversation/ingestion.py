"""Conversation ingestion pipeline from raw turn to canonical memory writes."""

from __future__ import annotations

from datetime import datetime, timezone
from uuid import uuid4

from cke.conversation.config import ConversationConfig
from cke.conversation.consolidation import MemoryConsolidator
from cke.conversation.extractors import (
    HeuristicMemoryExtractor,
    TemporalMemoryExtractor,
)
from cke.conversation.memory_store import ConversationMemoryStore
from cke.conversation.types import (
    ConversationEvent,
    ConversationTurn,
    TurnIngestionResult,
)
from cke.conversation.validation import CandidateMemoryValidator


class ConversationIngestionPipeline:
    """Deterministic ingestion pipeline for event storage and memory promotion."""

    def __init__(
        self,
        memory_store: ConversationMemoryStore,
        *,
        extractors: list | None = None,
        validator: CandidateMemoryValidator | None = None,
        consolidator: MemoryConsolidator | None = None,
        config: ConversationConfig | None = None,
    ) -> None:
        self.memory_store = memory_store
        self.config = config or memory_store.config
        self.extractors = extractors or [
            TemporalMemoryExtractor(),
            HeuristicMemoryExtractor(),
        ]
        self.validator = validator or CandidateMemoryValidator(self.config.validation)
        self.consolidator = consolidator or MemoryConsolidator(
            self.config.consolidation
        )

    def ingest_turn(
        self,
        conversation_id: str,
        role: str,
        text: str,
        *,
        timestamp: str | None = None,
        metadata: dict | None = None,
    ) -> TurnIngestionResult:
        ts = timestamp or datetime.now(timezone.utc).isoformat()
        turn_order = len(self.memory_store.get_events(conversation_id)) + 1
        event_id = f"{conversation_id}-event-{turn_order}-{uuid4().hex[:8]}"
        turn_id = f"{conversation_id}-turn-{turn_order}-{uuid4().hex[:8]}"
        event = ConversationEvent(
            conversation_id=conversation_id,
            event_id=event_id,
            turn_id=turn_id,
            turn_order=turn_order,
            role=role,
            text=text,
            timestamp=ts,
            metadata=dict(metadata or {}),
        )
        self.memory_store.store_event(event)

        raw_candidates = []
        for extractor in self.extractors:
            raw_candidates.extend(extractor.extract(event))
        valid_candidates, rejected_candidates = self.validator.validate(raw_candidates)
        decisions = self.consolidator.consolidate(
            valid_candidates,
            self.memory_store.get_canonical_memories(conversation_id),
            timestamp=ts,
        )
        accepted_memories = []
        for decision in decisions:
            if decision.canonical_memory is not None and decision.decision.value in {
                "accept",
                "update",
            }:
                self.memory_store.store_canonical_memory(decision.canonical_memory)
                accepted_memories.append(decision.canonical_memory)
        entities = sorted(
            {
                candidate.object
                for candidate in raw_candidates
                if candidate.relation == "mentions"
            }
            | {
                candidate.subject
                for candidate in accepted_memories
                if candidate.subject != "user"
            }
        )
        turn = ConversationTurn(
            conversation_id=conversation_id,
            turn_id=turn_id,
            turn_order=turn_order,
            role=role,
            text=text,
            timestamp=ts,
            entities=entities,
            facts=[memory.as_statement() for memory in accepted_memories],
            metadata=dict(metadata or {}),
            event_id=event_id,
        )
        self.memory_store.store_turn_view(turn)
        return TurnIngestionResult(
            event=event,
            turn=turn,
            raw_candidates=raw_candidates,
            accepted_memories=accepted_memories,
            rejected_candidates=rejected_candidates
            + [
                decision.candidate
                for decision in decisions
                if decision.decision.value == "reject"
            ],
            decisions=decisions,
        )
