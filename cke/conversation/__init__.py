"""Conversation-first memory, retrieval, and answering utilities."""

from __future__ import annotations

from importlib import import_module

__all__ = [
    "ConversationAnswer",
    "ConversationTurn",
    "ConversationalMemoryStore",
    "ConversationalReferenceResolver",
    "ConversationalRetriever",
    "ConversationalTurnExtractor",
    "GroundedAnswerComposer",
    "RetrievedMemory",
    "RetrievalBundle",
]

_LAZY_IMPORTS = {
    "ConversationAnswer": ("cke.conversation.types", "ConversationAnswer"),
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
    "GroundedAnswerComposer": ("cke.conversation.answering", "GroundedAnswerComposer"),
    "RetrievedMemory": ("cke.conversation.types", "RetrievedMemory"),
    "RetrievalBundle": ("cke.conversation.types", "RetrievalBundle"),
}


def __getattr__(name: str):
    if name not in _LAZY_IMPORTS:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module_name, attr_name = _LAZY_IMPORTS[name]
    module = import_module(module_name)
    return getattr(module, attr_name)
