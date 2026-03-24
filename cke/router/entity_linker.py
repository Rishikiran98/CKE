"""Deprecated – use :class:`cke.entity_resolution.entity_resolver.EntityResolver`.

This shim preserves backward compatibility for existing imports.
"""

from __future__ import annotations

from cke.entity_resolution.entity_resolver import EntityResolver


class EntityLinker:
    """Extract seed entities from natural language queries.

    .. deprecated::
        Use :class:`cke.entity_resolution.entity_resolver.EntityResolver` directly.
    """

    def __init__(self, graph_engine=None) -> None:
        self._resolver = EntityResolver(graph_engine=graph_engine)

    def extract_entities(self, query: str) -> list[str]:
        return self._resolver.extract_entities(query)
