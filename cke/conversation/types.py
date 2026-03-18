"""Core data contracts for conversational memory and retrieval."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from cke.models import Statement


@dataclass(slots=True)
class ConversationTurn:
    """Single stored conversational turn with raw text and extracted structure."""

    conversation_id: str
    turn_id: str
    turn_order: int
    role: str
    text: str
    timestamp: str
    entities: list[str] = field(default_factory=list)
    facts: list[Statement] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class RetrievedMemory:
    """A retrieved turn or fact chunk with semantic and heuristic scores."""

    memory_id: str
    memory_type: str
    text: str
    score: float
    conversation_id: str
    turn_id: str
    turn_order: int
    role: str
    entities: list[str] = field(default_factory=list)
    facts: list[Statement] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class RetrievalBundle:
    """Aggregated retrieval output used for answer grounding."""

    rewritten_query: str
    retrieved_turns: list[RetrievedMemory] = field(default_factory=list)
    retrieved_facts: list[Statement] = field(default_factory=list)
    graph_neighbors: list[Statement] = field(default_factory=list)
    candidate_paths: list[list[Statement]] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class ConversationAnswer:
    """Final grounded answer plus supporting evidence payload."""

    answer: str
    confidence: float
    grounded: bool
    retrieved_turns: list[RetrievedMemory] = field(default_factory=list)
    retrieved_facts: list[Statement] = field(default_factory=list)
    graph_neighbors: list[Statement] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)
