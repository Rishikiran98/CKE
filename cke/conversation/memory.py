"""Compatibility adapter exposing the historical memory store API."""

from __future__ import annotations

from cke.conversation.ingestion import ConversationIngestionPipeline
from cke.conversation.memory_store import ConversationMemoryStore
from cke.conversation.types import ConversationTurn


class ConversationalMemoryStore(ConversationMemoryStore):
    """Backward-compatible conversational memory store built on the new lifecycle."""

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        # The pipeline is a contract component with live _degrade calls, and
        # it was built without the store's strictness, so it could never be
        # strict through this path however the store was constructed.
        self.ingestion_pipeline = ConversationIngestionPipeline(
            self, strict=getattr(self, "strict", False)
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
        result = self.ingestion_pipeline.ingest_turn(
            conversation_id,
            role,
            text,
            timestamp=timestamp,
            metadata=metadata,
        )
        return result.turn
