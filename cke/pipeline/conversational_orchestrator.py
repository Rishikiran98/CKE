"""Conversation-first orchestration for live natural-language memory and RAG."""

from __future__ import annotations

from cke.conversation.answering import GroundedAnswerComposer
from cke.conversation.memory import ConversationalMemoryStore
from cke.conversation.retriever import ConversationalRetriever
from cke.conversation.types import ConversationAnswer, ConversationTurn


class ConversationalOrchestrator:
    """Ingest turns, semantically retrieve prior context, and answer naturally."""

    def __init__(
        self,
        memory_store: ConversationalMemoryStore | None = None,
        retriever: ConversationalRetriever | None = None,
        answer_composer: GroundedAnswerComposer | None = None,
    ) -> None:
        self.memory_store = memory_store or ConversationalMemoryStore()
        self.retriever = retriever or ConversationalRetriever(self.memory_store)
        self.answer_composer = answer_composer or GroundedAnswerComposer()
        self.last_bundle = None
        self.last_answer: ConversationAnswer | None = None

    def ingest_turn(
        self,
        conversation_id: str,
        role: str,
        text: str,
        *,
        timestamp: str | None = None,
        metadata: dict | None = None,
    ) -> ConversationTurn:
        return self.memory_store.ingest_turn(
            conversation_id,
            role,
            text,
            timestamp=timestamp,
            metadata=metadata,
        )

    def answer(self, conversation_id: str, query: str) -> ConversationAnswer:
        bundle = self.retriever.retrieve(query, conversation_id=conversation_id)
        answer = self.answer_composer.compose(query, bundle)
        self.last_bundle = bundle
        self.last_answer = answer
        return answer
