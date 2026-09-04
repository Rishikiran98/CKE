"""Conversation-first orchestration for live natural-language memory and RAG."""

from __future__ import annotations

from cke.conversation.answering import GroundedAnswerComposer
from cke.diagnostics import DegradationMixin, require_strict_component
from cke.conversation.memory import ConversationalMemoryStore
from cke.conversation.retriever import ConversationalRetriever
from cke.conversation.types import ConversationAnswer, ConversationTurn


class ConversationalOrchestrator(DegradationMixin):
    """Ingest turns, semantically retrieve prior context, and answer naturally."""

    def __init__(
        self,
        memory_store: ConversationalMemoryStore | None = None,
        retriever: ConversationalRetriever | None = None,
        answer_composer: GroundedAnswerComposer | None = None,
        strict: bool = False,
    ) -> None:
        # This took no strict at all, so the whole conversation subsystem was
        # unreachable in strict mode: without sentence-transformers it ran on
        # the hashed embedder by construction, whatever the caller asked for.
        self._init_degradation(strict)
        require_strict_component(
            type(self).__name__, memory_store, "memory store", strict
        )
        require_strict_component(type(self).__name__, retriever, "retriever", strict)
        self.memory_store = memory_store or ConversationalMemoryStore(strict=strict)
        self.retriever = retriever or ConversationalRetriever(
            self.memory_store, strict=strict
        )
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
