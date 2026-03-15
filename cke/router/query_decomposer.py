"""Query decomposition utilities for multi-hop graph retrieval."""

from __future__ import annotations

import re
from dataclasses import dataclass, field


@dataclass(slots=True)
class QueryStep:
    """A decomposition unit representing an entity or relation hop."""

    step_type: str
    value: str
    confidence: float = 1.0


@dataclass(slots=True)
class DecomposedQuery:
    """Structured representation of a decomposed query."""

    original_query: str
    steps: list[QueryStep] = field(default_factory=list)


class QueryDecomposer:
    """Heuristic decomposer that derives relation chains from natural language."""

    _QUESTION_PREFIXES = (
        "what",
        "which",
        "who",
        "where",
        "when",
        "how",
    )

    _RELATION_PATTERNS = [
        ("starred", "starred_in"),
        ("directed", "directed_by"),
        ("director", "directed_by"),
        ("written", "written_by"),
        ("author", "written_by"),
        ("nationality", "nationality"),
        ("born", "born_in"),
        ("located", "located_in"),
        ("capital", "capital_of"),
        ("part of", "part_of"),
        ("uses", "uses"),
        ("supports", "supports"),
        ("through", "via"),
        ("via", "via"),
    ]

    def decompose(
        self, query: str, entities: list[str] | None = None
    ) -> DecomposedQuery:
        """Decompose a query into ordered entity + relation steps."""
        normalized = (query or "").strip()
        entities = entities or []

        steps: list[QueryStep] = []
        seen: set[tuple[str, str]] = set()

        for entity in entities:
            key = ("entity", entity)
            if key not in seen:
                steps.append(
                    QueryStep(step_type="entity", value=entity, confidence=0.95)
                )
                seen.add(key)

        lower_query = normalized.lower()
        for marker, relation in self._RELATION_PATTERNS:
            if marker in lower_query:
                key = ("relation", relation)
                if key in seen:
                    continue
                confidence = 0.9 if marker in {"through", "via"} else 0.86
                steps.append(
                    QueryStep(
                        step_type="relation", value=relation, confidence=confidence
                    )
                )
                seen.add(key)

        tail_relation = self._infer_target_relation(lower_query)
        if tail_relation:
            key = ("relation", tail_relation)
            if key not in seen:
                steps.append(
                    QueryStep(
                        step_type="relation", value=tail_relation, confidence=0.82
                    )
                )

        return DecomposedQuery(original_query=normalized, steps=steps)

    def _infer_target_relation(self, query: str) -> str | None:
        stripped = query.strip(" ?")
        for prefix in self._QUESTION_PREFIXES:
            if stripped.startswith(prefix):
                if " nationality" in f" {stripped} ":
                    return "nationality"
                if " director" in f" {stripped} ":
                    return "directed_by"
                if " author" in f" {stripped} ":
                    return "written_by"
                if " where" == f" {prefix}" and "located" in stripped:
                    return "located_in"
        return None
