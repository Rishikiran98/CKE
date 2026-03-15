"""Entity linking utilities for canonical graph entity resolution."""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from difflib import SequenceMatcher
from typing import Any

try:
    from rapidfuzz import fuzz
except Exception:  # pragma: no cover - optional dependency
    fuzz = None

try:
    import numpy as np
except Exception:  # pragma: no cover
    np = None

try:
    from sentence_transformers import SentenceTransformer
except Exception:  # pragma: no cover
    SentenceTransformer = None


@dataclass(slots=True)
class ResolutionResult:
    canonical: str
    confidence: float


class EntityResolver:
    """Resolve entity mentions to canonical graph nodes."""

    def __init__(
        self,
        graph_engine: Any,
        aliases: dict[str, str] | None = None,
        fuzzy_threshold: float = 0.9,
        embedding_threshold: float = 0.8,
    ) -> None:
        self.graph_engine = graph_engine
        self.aliases: dict[str, str] = {}
        self.fuzzy_threshold = fuzzy_threshold
        self.embedding_threshold = embedding_threshold
        self._embedding_cache: dict[str, list[float]] = {}
        self._embedding_model = self._load_embedding_model()

        for alias, canonical in (aliases or {}).items():
            self.aliases[self._normalize(alias)] = canonical

    def resolve(self, entity: str) -> str:
        return self.resolve_with_score(entity).canonical

    def resolve_with_score(self, entity: str) -> ResolutionResult:
        normalized = self._normalize(entity)
        if not normalized:
            return ResolutionResult(canonical=str(entity).strip(), confidence=0.0)

        if normalized in self.aliases:
            return ResolutionResult(self.aliases[normalized], confidence=1.0)

        candidates = self._graph_candidates()
        if not candidates:
            canonical = self._title_case(entity)
            self.aliases[normalized] = canonical
            return ResolutionResult(canonical=canonical, confidence=0.8)

        best_fuzzy_candidate, best_fuzzy = self._best_fuzzy(normalized, candidates)
        if best_fuzzy_candidate and best_fuzzy >= self.fuzzy_threshold:
            self.aliases[normalized] = best_fuzzy_candidate
            return ResolutionResult(best_fuzzy_candidate, confidence=best_fuzzy)

        best_emb_candidate, best_emb = self._best_embedding(entity, candidates)
        if best_emb_candidate and best_emb >= self.embedding_threshold:
            self.aliases[normalized] = best_emb_candidate
            return ResolutionResult(best_emb_candidate, confidence=best_emb)

        canonical = self._title_case(entity)
        self.aliases[normalized] = canonical
        return ResolutionResult(canonical=canonical, confidence=max(best_fuzzy, best_emb, 0.6))

    def _graph_candidates(self) -> list[str]:
        getter = getattr(self.graph_engine, "get_entities", None)
        if callable(getter):
            entities = getter()
        else:
            entities = self.graph_engine.all_entities()
        return [str(e) for e in entities if str(e).strip()]

    def _best_fuzzy(self, mention: str, candidates: list[str]) -> tuple[str | None, float]:
        best: str | None = None
        best_score = 0.0
        for candidate in candidates:
            score = self._fuzzy_score(mention, self._normalize(candidate))
            if score > best_score:
                best = candidate
                best_score = score
        return best, best_score

    def _best_embedding(self, mention: str, candidates: list[str]) -> tuple[str | None, float]:
        best: str | None = None
        best_score = 0.0
        for candidate in candidates:
            score = self._embedding_similarity(mention, candidate)
            if score > best_score:
                best = candidate
                best_score = score
        return best, best_score

    @staticmethod
    def _normalize(text: str) -> str:
        lowered = str(text).lower().replace("_", " ")
        cleaned = re.sub(r"[^\w\s]", " ", lowered)
        return re.sub(r"\s+", " ", cleaned).strip()

    @staticmethod
    def _title_case(text: str) -> str:
        compact = re.sub(r"\s+", " ", str(text)).strip()
        return " ".join(part.capitalize() for part in compact.split()) if compact else compact

    @staticmethod
    def _fuzzy_score(left: str, right: str) -> float:
        if fuzz is not None:
            return float(fuzz.ratio(left, right) / 100.0)
        return SequenceMatcher(a=left, b=right).ratio()

    def _load_embedding_model(self):
        if SentenceTransformer is None:
            return None
        try:
            return SentenceTransformer("all-MiniLM-L6-v2")
        except Exception:
            return None

    def _embed(self, text: str) -> list[float]:
        key = str(text)
        if key in self._embedding_cache:
            return self._embedding_cache[key]

        if self._embedding_model is not None and np is not None:
            vector = self._embedding_model.encode([text], normalize_embeddings=True)[0]
            out = vector.tolist()
            self._embedding_cache[key] = out
            return out

        vec = [0.0] * 128
        for token in re.findall(r"\w+", self._normalize(text)):
            vec[hash(token) % 128] += 1.0
        norm = math.sqrt(sum(v * v for v in vec)) or 1.0
        out = [v / norm for v in vec]
        self._embedding_cache[key] = out
        return out

    def _embedding_similarity(self, left: str, right: str) -> float:
        lvec, rvec = self._embed(left), self._embed(right)
        denom = (
            math.sqrt(sum(v * v for v in lvec)) * math.sqrt(sum(v * v for v in rvec))
        ) or 1.0
        return sum(a * b for a, b in zip(lvec, rvec)) / denom
