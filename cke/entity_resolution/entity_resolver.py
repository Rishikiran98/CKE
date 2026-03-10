"""Entity resolution module.

Combines string similarity and embedding similarity for canonicalization.
"""

from __future__ import annotations

import math
import re
from difflib import SequenceMatcher
from typing import Dict, Iterable, Optional

try:
    import numpy as np
except Exception:  # pragma: no cover
    np = None

try:
    from sentence_transformers import SentenceTransformer
except Exception:  # pragma: no cover - optional runtime dependency
    SentenceTransformer = None


class EntityResolver:
    """Resolve mentions to canonical entity names."""

    def __init__(
        self,
        aliases: Optional[Dict[str, str]] = None,
        string_threshold: float = 0.82,
        embedding_threshold: float = 0.75,
    ) -> None:
        self._canonical_entities: set[str] = set()
        self._aliases: dict[str, str] = {}
        self.string_threshold = string_threshold
        self.embedding_threshold = embedding_threshold
        if aliases:
            for alias, canonical in aliases.items():
                self.register_alias(alias, canonical)

        self._model = self._load_embedding_model()
        self._embedding_cache: dict[str, list[float]] = {}

    def register_alias(self, alias: str, canonical: str) -> None:
        normalized_alias = self._normalize(alias)
        self._aliases[normalized_alias] = canonical
        self._canonical_entities.add(canonical)

    def resolve_entity(self, name: str) -> str:
        normalized = self._normalize(name)
        if normalized in self._aliases:
            return self._aliases[normalized]

        if not self._canonical_entities:
            canonical = self._title_case_entity(name)
            self._canonical_entities.add(canonical)
            self._aliases[normalized] = canonical
            return canonical

        best_match: Optional[str] = None
        best_score = -1.0
        for canonical in self._canonical_entities:
            score = max(
                self._string_similarity(normalized, self._normalize(canonical)),
                self._embedding_similarity(name, canonical),
            )
            if score > best_score:
                best_match, best_score = canonical, score

        if best_match and best_score >= min(self.string_threshold, self.embedding_threshold):
            self._aliases[normalized] = best_match
            return best_match

        canonical = self._title_case_entity(name)
        self._canonical_entities.add(canonical)
        self._aliases[normalized] = canonical
        return canonical

    def merge_entities(self, entity_a: str, entity_b: str) -> str:
        canonical_a = self.resolve_entity(entity_a)
        canonical_b = self.resolve_entity(entity_b)
        if canonical_a == canonical_b:
            return canonical_a

        survivor = min([canonical_a, canonical_b], key=len)
        removed = canonical_b if survivor == canonical_a else canonical_a
        for alias, canonical in list(self._aliases.items()):
            if canonical == removed:
                self._aliases[alias] = survivor
        self._canonical_entities.discard(removed)
        self._canonical_entities.add(survivor)
        return survivor

    def known_entities(self) -> Iterable[str]:
        return sorted(self._canonical_entities)

    @staticmethod
    def _normalize(text: str) -> str:
        return re.sub(r"\s+", " ", text.strip().lower())

    @staticmethod
    def _title_case_entity(text: str) -> str:
        clean = re.sub(r"\s+", " ", text.strip())
        if clean.isupper() or len(clean) <= 5:
            return clean
        return " ".join(part.capitalize() for part in clean.split())

    @staticmethod
    def _string_similarity(left: str, right: str) -> float:
        return SequenceMatcher(a=left, b=right).ratio()

    def _load_embedding_model(self):
        if SentenceTransformer is None:
            return None
        try:
            return SentenceTransformer("all-MiniLM-L6-v2")
        except Exception:
            return None

    def _embed(self, text: str) -> list[float]:
        if text in self._embedding_cache:
            return self._embedding_cache[text]

        if self._model is not None and np is not None:
            vec = self._model.encode([text], normalize_embeddings=True)[0].tolist()
        else:
            vec = [0.0] * 128
            for token in re.findall(r"\w+", text.lower()):
                vec[hash(token) % 128] += 1.0
            norm = math.sqrt(sum(v * v for v in vec)) or 1.0
            vec = [v / norm for v in vec]

        self._embedding_cache[text] = vec
        return vec

    def _embedding_similarity(self, left: str, right: str) -> float:
        lvec, rvec = self._embed(left), self._embed(right)
        denom = (math.sqrt(sum(v * v for v in lvec)) * math.sqrt(sum(v * v for v in rvec))) or 1.0
        return sum(a * b for a, b in zip(lvec, rvec)) / denom
