"""Pipeline package for orchestrated query execution."""

from __future__ import annotations

from importlib import import_module

__all__ = [
    "ConversationalOrchestrator",
    "EvidenceFact",
    "QueryOrchestrator",
    "QueryResult",
    "ReasoningContext",
    "ResolvedEntity",
    "RetrievedChunk",
]

_LAZY_IMPORTS = {
    "ConversationalOrchestrator": (
        "cke.pipeline.conversational_orchestrator",
        "ConversationalOrchestrator",
    ),
    "EvidenceFact": ("cke.pipeline.types", "EvidenceFact"),
    "QueryOrchestrator": ("cke.pipeline.query_orchestrator", "QueryOrchestrator"),
    "QueryResult": ("cke.pipeline.types", "QueryResult"),
    "ReasoningContext": ("cke.pipeline.types", "ReasoningContext"),
    "ResolvedEntity": ("cke.pipeline.types", "ResolvedEntity"),
    "RetrievedChunk": ("cke.pipeline.types", "RetrievedChunk"),
}


def __getattr__(name: str):
    if name not in _LAZY_IMPORTS:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module_name, attr_name = _LAZY_IMPORTS[name]
    module = import_module(module_name)
    return getattr(module, attr_name)
