"""Entity resolution module.

Combines canonical alias mapping, string similarity, and embedding similarity
for robust canonicalization.
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
        self._canonical_by_key: dict[str, str] = {}
        self.string_threshold = string_threshold
        self.embedding_threshold = embedding_threshold

        self._model = self._load_embedding_model()
        self._embedding_cache: dict[str, list[float]] = {}

        if aliases:
            for alias, canonical in aliases.items():
                self.register_alias(alias, canonical)

    def register_alias(self, alias: str, canonical: str) -> None:
        """Register alias -> canonical and maintain canonical form mappings."""
        canonical_name = self._title_case_entity(canonical)
        normalized_alias = self._normalize(alias)
        self._aliases[normalized_alias] = canonical_name
        self._canonical_entities.add(canonical_name)
        self._canonical_by_key[self._canonical_key(canonical_name)] = canonical_name
        self._canonical_by_key[self._canonical_key(alias)] = canonical_name

    def resolve_entity(self, name: str) -> str:
        """Resolve mention to canonical entity using hybrid matching."""
        normalized = self._normalize(name)
        if normalized in self._aliases:
            return self._aliases[normalized]

        key = self._canonical_key(name)
        if key in self._canonical_by_key:
            canonical = self._canonical_by_key[key]
            self._aliases[normalized] = canonical
            return canonical

        if not self._canonical_entities:
            canonical = self._title_case_entity(name)
            self.register_alias(name, canonical)
            return canonical

        best_match: Optional[str] = None
        best_string_score = -1.0
        best_embedding_score = -1.0
        best_combined_score = -1.0

        for canonical in self._canonical_entities:
            normalized_canonical = self._normalize(canonical)
            key_canonical = self._canonical_key(canonical)
            string_score = max(
                self._string_similarity(normalized, normalized_canonical),
                self._string_similarity(key, key_canonical),
            )
            embedding_score = self._embedding_similarity(name, canonical)
            combined_score = 0.65 * string_score + 0.35 * embedding_score
            if combined_score > best_combined_score:
                best_match = canonical
                best_string_score = string_score
                best_embedding_score = embedding_score
                best_combined_score = combined_score

        if best_match and (
            best_string_score >= self.string_threshold
            or best_embedding_score >= self.embedding_threshold
            or best_combined_score >= min(self.string_threshold, self.embedding_threshold)
        ):
            self._aliases[normalized] = best_match
            self._canonical_by_key[key] = best_match
            return best_match

        canonical = self._title_case_entity(name)
        self.register_alias(name, canonical)
        return canonical

    def merge_entities(self, entity_a: str, entity_b: str) -> str:
        """Merge two entities and return the surviving canonical name."""
        canonical_a = self.resolve_entity(entity_a)
        canonical_b = self.resolve_entity(entity_b)
        if canonical_a == canonical_b:
            return canonical_a

        survivor = min([canonical_a, canonical_b], key=len)
        removed = canonical_b if survivor == canonical_a else canonical_a

        for alias, canonical in list(self._aliases.items()):
            if canonical == removed:
                self._aliases[alias] = survivor

        for form, canonical in list(self._canonical_by_key.items()):
            if canonical == removed:
                self._canonical_by_key[form] = survivor

        self._canonical_entities.discard(removed)
        self._canonical_entities.add(survivor)
        self._canonical_by_key[self._canonical_key(survivor)] = survivor
        return survivor

    def known_entities(self) -> Iterable[str]:
        """List known canonical entities."""
        return sorted(self._canonical_entities)

    @staticmethod
    def _normalize(text: str) -> str:
        return re.sub(r"\s+", " ", text.strip().lower())

    def _canonical_key(self, text: str) -> str:
        normalized = self._normalize(text)
        normalized = normalized.replace("_", " ").replace("-", " ")
        tokens = [tok for tok in re.findall(r"[a-z0-9]+", normalized) if tok not in {"db", "database", "server"}]
        return " ".join(tokens)

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
