"""Entity canonicalization and resolution for graph retrieval."""

from __future__ import annotations

from dataclasses import dataclass
from difflib import SequenceMatcher
from typing import Callable
import re


@dataclass(slots=True)
class EntityResolver:
    """Resolve entity mentions to canonical graph node names."""

    aliases: dict[str, str] | None = None
    fuzzy_threshold: float = 0.84
    embedding_threshold: float = 0.8
    embedding_similarity_fn: Callable[[str, str], float] | None = None
    _alias_map: dict[str, str] | None = None
    _canonical_set: set[str] | None = None

    def __post_init__(self) -> None:
        self._alias_map = {}
        self._canonical_set = set()
        if self.aliases:
            for alias, canonical in self.aliases.items():
                self.register_alias(alias, canonical)

    @staticmethod
    def _normalize(name: str) -> str:
        return re.sub(r"\s+", " ", name.strip().lower())

    def register_alias(self, alias: str, canonical: str) -> None:
        canonical_name = canonical.strip()
        self._canonical_set.add(canonical_name)
        self._alias_map[self._normalize(alias)] = canonical_name
        self._alias_map[self._normalize(canonical_name)] = canonical_name

    def canonicalize(self, name: str) -> str:
        """Return canonical entity if known, else normalized title form."""
        return self.resolve(name)

    def resolve(self, name: str) -> str:
        normalized = self._normalize(name)
        if normalized in self._alias_map:
            return self._alias_map[normalized]

        best_match = ""
        best_score = 0.0
        for canonical in self._canonical_set:
            score = SequenceMatcher(a=normalized, b=self._normalize(canonical)).ratio()
            if self.embedding_similarity_fn is not None:
                emb_score = self.embedding_similarity_fn(name, canonical)
                if emb_score >= self.embedding_threshold:
                    score = max(score, emb_score)
            if score > best_score:
                best_match = canonical
                best_score = score

        if best_match and best_score >= self.fuzzy_threshold:
            self._alias_map[normalized] = best_match
            return best_match

        canonical = " ".join(part.capitalize() for part in normalized.split())
        self.register_alias(name, canonical)
        return canonical
