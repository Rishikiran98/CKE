
"""Extractor interfaces for candidate memory proposal generation."""

from __future__ import annotations

from typing import Protocol

from cke.conversation.types import CandidateMemory, ConversationEvent


class BaseMemoryExtractor(Protocol):
    """Protocol for deterministic memory proposal generators."""

    name: str

    def extract(self, event: ConversationEvent) -> list[CandidateMemory]:
        """Return candidate memories proposed from a raw conversation event."""
