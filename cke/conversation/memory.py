"""Conversation-first memory store preserving raw turns and structured facts."""

from __future__ import annotations

from collections import defaultdict
from datetime import datetime, timezone
from uuid import uuid4

from cke.conversation.extractor import ConversationalTurnExtractor
from cke.conversation.patterns import normalize_fact_parts
from cke.conversation.types import ConversationTurn
from cke.graph_engine.graph_engine import KnowledgeGraphEngine
from cke.models import Statement


class ConversationalMemoryStore:
    """Store conversational turns, extracted facts, and graph-friendly memory."""

    def __init__(
        self,
        extractor: ConversationalTurnExtractor | None = None,
        graph_engine: KnowledgeGraphEngine | None = None,
    ) -> None:
        self.extractor = extractor or ConversationalTurnExtractor()
        self.graph_engine = graph_engine or KnowledgeGraphEngine()
        self._turns_by_conversation: dict[str, list[ConversationTurn]] = defaultdict(
            list
        )

    def ingest_turn(
        self,
        conversation_id: str,
        role: str,
        text: str,
        *,
        timestamp: str | None = None,
        metadata: dict | None = None,
    ) -> ConversationTurn:
        turns = self._turns_by_conversation[conversation_id]
        turn_order = len(turns) + 1
        turn_id = f"{conversation_id}-turn-{turn_order}-{uuid4().hex[:8]}"
        ts = timestamp or datetime.now(timezone.utc).isoformat()
        extraction = self.extractor.extract(text, role=role)

        facts: list[Statement] = []
        for index, fact in enumerate(extraction.facts):
            normalized_subject, normalized_relation, normalized_object = (
                normalize_fact_parts(fact.subject, fact.relation, fact.object)
            )
            if not normalized_subject or not normalized_object:
                continue
            enriched_context = dict(fact.context)
            enriched_context.update(
                {
                    "conversation_id": conversation_id,
                    "turn_id": turn_id,
                    "turn_order": turn_order,
                    "speaker_role": role,
                }
            )
            enriched = Statement(
                subject=normalized_subject,
                relation=normalized_relation,
                object=normalized_object,
                context=enriched_context,
                confidence=fact.confidence,
                source=conversation_id,
                timestamp=ts,
                statement_id=f"{turn_id}-fact-{index}",
                chunk_id=turn_id,
                source_doc_id=conversation_id,
            )
            facts.append(enriched)
            self.graph_engine.add_statement(
                enriched.subject,
                enriched.relation,
                enriched.object,
                context=enriched.context,
                confidence=enriched.confidence,
                source=enriched.source,
                timestamp=enriched.timestamp,
            )

        turn = ConversationTurn(
            conversation_id=conversation_id,
            turn_id=turn_id,
            turn_order=turn_order,
            role=role,
            text=text,
            timestamp=ts,
            entities=list(extraction.entities),
            facts=facts,
            metadata=dict(metadata or {}),
        )
        turns.append(turn)
        return turn

    def get_turns(self, conversation_id: str) -> list[ConversationTurn]:
        return list(self._turns_by_conversation.get(conversation_id, []))

    def latest_turns(
        self, conversation_id: str, limit: int = 5
    ) -> list[ConversationTurn]:
        turns = self._turns_by_conversation.get(conversation_id, [])
        if limit <= 0:
            return []
        return list(turns[-limit:])

    def facts_for_conversation(self, conversation_id: str) -> list[Statement]:
        facts: list[Statement] = []
        for turn in self._turns_by_conversation.get(conversation_id, []):
            facts.extend(turn.facts)
        return facts

    def all_conversation_ids(self) -> list[str]:
        return sorted(self._turns_by_conversation)
