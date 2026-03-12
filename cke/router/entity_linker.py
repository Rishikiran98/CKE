"""Query-time entity extraction and linking."""

from __future__ import annotations

import re

from cke.graph_engine.graph_engine import KnowledgeGraphEngine


class EntityLinker:
    """Extract seed entities from natural language queries."""

    def __init__(self, graph_engine: KnowledgeGraphEngine | None = None) -> None:
        self.graph_engine = graph_engine

    def extract_entities(self, query: str) -> list[str]:
        candidates: set[str] = set()
        query_lower = query.lower()

        known_entities = self.graph_engine.all_entities() if self.graph_engine else []

        # Graph entity string matching.
        for entity in known_entities:
            if entity.lower() in query_lower:
                candidates.add(entity)

        # Capitalized phrase detection (simple NER proxy).
        for phrase in re.findall(r"\b(?:[A-Z][\w-]*)(?:\s+[A-Z][\w-]*)*\b", query):
            phrase = phrase.strip()
            if phrase and len(phrase) > 1:
                candidates.add(phrase)

        # Context clue matching: if relation/object mention appears in query,
        # include connected subject entity.
        clue_tokens = set(re.findall(r"[a-z0-9_+-]+", query_lower))
        for entity in known_entities:
            for statement in (
                self.graph_engine.get_neighbors(entity) if self.graph_engine else []
            ):
                relation_tokens = set(
                    re.findall(r"[a-z0-9_+-]+", statement.relation.lower())
                )
                object_tokens = set(
                    re.findall(r"[a-z0-9_+-]+", statement.object.lower())
                )
                if clue_tokens.intersection(relation_tokens | object_tokens):
                    candidates.add(entity)

        return sorted(candidates)
