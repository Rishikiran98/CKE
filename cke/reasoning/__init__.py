"""Reasoning module exports with lazy loading for heavy backends."""

from __future__ import annotations

from importlib import import_module

__all__ = ["TemplateReasoner", "PathReasoner", "LLMReasoner", "LLMReasonerConfig"]

_LAZY_IMPORTS = {
    "LLMReasoner": ("cke.reasoning.llm_reasoner", "LLMReasoner"),
    "LLMReasonerConfig": ("cke.reasoning.llm_reasoner", "LLMReasonerConfig"),
    "PathReasoner": ("cke.reasoning.path_reasoner", "PathReasoner"),
    "TemplateReasoner": ("cke.reasoning.reasoner", "TemplateReasoner"),
}


def __getattr__(name: str):
    if name not in _LAZY_IMPORTS:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module_name, attr_name = _LAZY_IMPORTS[name]
    module = import_module(module_name)
    value = getattr(module, attr_name)
    globals()[name] = value
    return value
