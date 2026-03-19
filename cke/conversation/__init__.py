"""Conversation memory subsystem with explicit memory lifecycle stages."""

from __future__ import annotations

from importlib import import_module

__all__ = [
    "CandidateMemory",
    "CanonicalMemory",
    "ConversationAnswer",
    "ConversationEvent",
    "ConversationIngestionPipeline",
    "ConversationMemoryStore",
    "ConversationTurn",
    "ConversationalMemoryStore",
    "ConversationalReferenceResolver",
    "ConversationalRetriever",
    "ConversationalTurnExtractor",
    "EvidenceSet",
    "GroundedAnswerComposer",
    "MemoryConflict",
    "MemoryWriteDecision",
    "RetrievedMemory",
    "RetrievalBundle",
    "TurnIngestionResult",
]

_LAZY_IMPORTS = {
    "CandidateMemory": ("cke.conversation.types", "CandidateMemory"),
    "CanonicalMemory": ("cke.conversation.types", "CanonicalMemory"),
    "ConversationAnswer": ("cke.conversation.types", "ConversationAnswer"),
    "ConversationEvent": ("cke.conversation.types", "ConversationEvent"),
    "ConversationIngestionPipeline": (
        "cke.conversation.ingestion",
        "ConversationIngestionPipeline",
    ),
    "ConversationMemoryStore": (
        "cke.conversation.memory_store",
        "ConversationMemoryStore",
    ),
    "ConversationTurn": ("cke.conversation.types", "ConversationTurn"),
    "ConversationalMemoryStore": (
        "cke.conversation.memory",
        "ConversationalMemoryStore",
    ),
    "ConversationalReferenceResolver": (
        "cke.conversation.reference_resolution",
        "ConversationalReferenceResolver",
    ),
    "ConversationalRetriever": (
        "cke.conversation.retriever",
        "ConversationalRetriever",
    ),
    "ConversationalTurnExtractor": (
        "cke.conversation.extractor",
        "ConversationalTurnExtractor",
    ),
    "EvidenceSet": ("cke.conversation.types", "EvidenceSet"),
    "GroundedAnswerComposer": ("cke.conversation.answering", "GroundedAnswerComposer"),
    "MemoryConflict": ("cke.conversation.types", "MemoryConflict"),
    "MemoryWriteDecision": ("cke.conversation.types", "MemoryWriteDecision"),
    "RetrievedMemory": ("cke.conversation.types", "RetrievedMemory"),
    "RetrievalBundle": ("cke.conversation.types", "RetrievalBundle"),
    "TurnIngestionResult": ("cke.conversation.types", "TurnIngestionResult"),
}


def __getattr__(name: str):
    if name not in _LAZY_IMPORTS:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module_name, attr_name = _LAZY_IMPORTS[name]
    module = import_module(module_name)
    return getattr(module, attr_name)
