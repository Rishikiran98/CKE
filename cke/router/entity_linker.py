"""Query-time entity extraction and linking."""

from __future__ import annotations

import re

from cke.graph_engine.graph_engine import KnowledgeGraphEngine


class EntityLinker:
    """Extract seed entities from natural language queries."""

    _QUESTION_WORDS = {
        "what",
        "how",
        "where",
        "when",
        "why",
        "which",
        "who",
        "whom",
    }
    _AUXILIARIES = {
        "did",
        "do",
        "does",
        "is",
        "are",
        "was",
        "were",
        "has",
        "have",
        "had",
        "can",
        "could",
        "will",
        "would",
        "should",
        "may",
        "might",
    }

    def __init__(self, graph_engine: KnowledgeGraphEngine | None = None) -> None:
        self.graph_engine = graph_engine

    def extract_entities(self, query: str) -> list[str]:
        candidates: set[str] = set()
        query_lower = query.lower()

        known_entities = self.graph_engine.all_entities() if self.graph_engine else []

        # Graph entity string matching.
        for entity in known_entities:
            if self._entity_in_query(entity, query):
                candidates.add(entity)

        # Capitalized phrase detection (simple NER proxy).
        for phrase in re.findall(r"\b(?:[A-Z][\w-]*)(?:\s+[A-Z][\w-]*)*\b", query):
            phrase = self._clean_phrase(phrase)
            if self._keep_phrase(phrase, query):
                candidates.add(phrase)

        # Context clue matching: if relation/object mention appears in query,
        # include connected subject entity.
        clue_tokens = set(re.findall(r"[a-z0-9_+-]+", query_lower))
        for entity in known_entities:
            for statement in (
                self.graph_engine.get_neighbors(entity) if self.graph_engine else []
            ):
                object_text = statement.object.strip()
                object_tokens = set(
                    re.findall(r"[a-z0-9_+-]+", statement.object.lower())
                )
                if len(object_text) == 1:
                    continue
                if clue_tokens.intersection(object_tokens):
                    candidates.add(entity)

        return sorted(candidates)

    def _entity_in_query(self, entity: str, query: str) -> bool:
        cleaned = entity.strip()
        if not cleaned:
            return False
        if len(cleaned) == 1:
            return re.search(rf"\b{re.escape(cleaned)}\b", query) is not None
        return re.search(rf"\b{re.escape(cleaned)}\b", query, flags=re.IGNORECASE) is not None

    def _clean_phrase(self, phrase: str) -> str:
        tokens = phrase.strip().split()
        while tokens and tokens[0].lower() in (self._QUESTION_WORDS | self._AUXILIARIES):
            tokens = tokens[1:]
        return " ".join(tokens).strip(" ?!.,")

    def _keep_phrase(self, phrase: str, query: str) -> bool:
        if not phrase:
            return False
        lowered = phrase.lower()
        if lowered in self._QUESTION_WORDS:
            return False
        if len(phrase) == 1:
            return re.search(rf"\b{re.escape(phrase)}\b", query) is not None
        return len(phrase) > 1
