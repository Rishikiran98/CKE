"""Unified entity mention detection, canonical resolution, and linking.

Consolidates query-time detection (formerly ``cke.router.entity_linker``),
alias + normalisation resolution, fuzzy matching (rapidfuzz / SequenceMatcher),
and embedding similarity (SentenceTransformer) into a single resolver used by
both the ingestion pipeline and the query pipeline.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from difflib import SequenceMatcher
from typing import TYPE_CHECKING, Any, Callable, Iterable

from cke.entity_resolution.alias_registry import AliasRegistry

if TYPE_CHECKING:
    from cke.pipeline.types import ResolvedEntity

# ---------------------------------------------------------------------------
# Optional heavy dependencies – graceful degradation when absent.
# ---------------------------------------------------------------------------

try:
    from rapidfuzz import fuzz  # type: ignore[import-untyped]
except Exception:  # pragma: no cover – optional dependency
    fuzz = None

try:
    import numpy as np  # type: ignore[import-untyped]
except Exception:  # pragma: no cover
    np = None

try:
    from sentence_transformers import SentenceTransformer  # type: ignore[import-untyped]
except Exception:  # pragma: no cover
    SentenceTransformer = None


# ---------------------------------------------------------------------------
# Lightweight result dataclass shared by resolve_with_score().
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class ResolutionResult:
    """Canonical name paired with a resolution confidence score."""

    canonical: str
    confidence: float


# ---------------------------------------------------------------------------
# Consolidated EntityResolver
# ---------------------------------------------------------------------------


class EntityResolver:
    """Resolve entity mentions to canonical names and extract entities from queries.

    Resolution chain (in order):
        1. Exact canonical match  → confidence 0.95
        2. Alias-registry lookup  → confidence 0.90
        3. Normalised / key match → confidence 0.75
        4. Fuzzy string match     → confidence = fuzzy score  (≥ *fuzzy_threshold*)
        5. Embedding similarity   → confidence = emb score    (≥ *embedding_threshold*)
        6. Title-case fallback    → confidence = max(fuzzy, emb, 0.50)
    """

    # Tokens stripped from the beginning of detected phrases.
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
    _LEADING_SCAFFOLD = _QUESTION_WORDS | {
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

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------

    def __init__(
        self,
        aliases: dict[str, str] | None = None,
        *,
        graph_engine: Any | None = None,
        fuzzy_threshold: float = 0.9,
        embedding_threshold: float = 0.8,
        embedding_similarity_fn: Callable[[str, str], float] | None = None,
    ) -> None:
        self.registry = AliasRegistry()
        self._canonical_entities: set[str] = set()
        self.graph_engine = graph_engine
        self.fuzzy_threshold = fuzzy_threshold
        self.embedding_threshold = embedding_threshold
        self.embedding_similarity_fn = embedding_similarity_fn

        # Embedding model is loaded lazily on first call to _embed().
        self._embedding_model: Any | None = _SENTINEL
        self._embedding_cache: dict[str, list[float]] = {}

        if aliases:
            canonical_to_aliases: dict[str, list[str]] = {}
            for alias, canonical in aliases.items():
                canonical_to_aliases.setdefault(canonical, []).append(alias)
            for canonical, values in canonical_to_aliases.items():
                self.register_aliases(canonical, values)

    # ------------------------------------------------------------------
    # Alias registration
    # ------------------------------------------------------------------

    def register_alias(self, alias: str, canonical: str) -> None:
        self.register_aliases(canonical, [alias])

    def register_aliases(self, canonical: str, aliases: list[str]) -> None:
        canonical_name = str(canonical).strip()
        if not canonical_name:
            return
        self._canonical_entities.add(canonical_name)
        self.registry.add(canonical_name, aliases)

    # ------------------------------------------------------------------
    # Core resolution — single entity
    # ------------------------------------------------------------------

    def resolve_with_score(self, entity: str) -> ResolutionResult:
        """Full resolution chain returning canonical name + confidence."""
        name = str(entity).strip()
        normalized = self._normalize(name)
        if not normalized:
            return ResolutionResult(canonical=name or "", confidence=0.0)

        # 1. Exact canonical match
        if name in self._canonical_entities:
            return ResolutionResult(canonical=name, confidence=0.95)

        # 2. Alias-registry lookup
        resolved = self.registry.resolve(name)
        if resolved:
            return ResolutionResult(canonical=resolved, confidence=0.90)

        # 3. Normalised / canonical-key match
        norm = AliasRegistry.normalize(name)
        key = self._canonical_key(name)
        for known in self._canonical_entities:
            if (
                AliasRegistry.normalize(known) == norm
                or self._canonical_key(known) == key
            ):
                self.register_alias(name, known)
                return ResolutionResult(canonical=known, confidence=0.75)

        # 4-6. Fuzzy → embedding → fallback (need candidates)
        candidates = self._graph_candidates()
        if not candidates:
            canonical = self._title_case_entity(name)
            self.register_alias(name, canonical)
            return ResolutionResult(canonical=canonical, confidence=0.50)

        best_fuzzy_candidate, best_fuzzy = self._best_fuzzy(normalized, candidates)
        if best_fuzzy_candidate and best_fuzzy >= self.fuzzy_threshold:
            self.register_alias(name, best_fuzzy_candidate)
            return ResolutionResult(best_fuzzy_candidate, confidence=best_fuzzy)

        best_emb_candidate, best_emb = self._best_embedding(name, candidates)
        if best_emb_candidate and best_emb >= self.embedding_threshold:
            self.register_alias(name, best_emb_candidate)
            return ResolutionResult(best_emb_candidate, confidence=best_emb)

        canonical = self._title_case_entity(name)
        self.register_alias(name, canonical)
        return ResolutionResult(
            canonical=canonical,
            confidence=max(best_fuzzy, best_emb, 0.50),
        )

    def resolve_entity(self, name: str) -> str:
        """Resolve *name* to a canonical entity string."""
        return self.resolve_with_score(name).canonical

    # Convenience aliases used by GraphRetriever.
    def resolve(self, name: str) -> str:  # noqa: D102
        return self.resolve_entity(name)

    def canonicalize(self, name: str) -> str:  # noqa: D102
        return self.resolve_entity(name)

    # ------------------------------------------------------------------
    # Mention detection (from a query string)
    # ------------------------------------------------------------------

    def detect_mentions(
        self,
        query: str,
        candidate_entities: Iterable[str] | None = None,
    ) -> list[str]:
        """Detect entity mentions in *query* using candidates + registry + NER heuristics."""
        q = query or ""
        mentions: list[str] = []

        for canonical in sorted(set(candidate_entities or []), key=len, reverse=True):
            cleaned = self._clean_mention(canonical)
            if cleaned and self._mention_in_query(cleaned, q):
                mentions.append(cleaned)

        for canonical, aliases in self.registry.canonical_to_aliases.items():
            for alias in aliases:
                cleaned_alias = self._clean_mention(alias)
                if cleaned_alias and self._mention_in_query(cleaned_alias, q):
                    mentions.append(cleaned_alias)
                cleaned_canonical = self._clean_mention(canonical)
                if cleaned_canonical and self._mention_in_query(cleaned_canonical, q):
                    mentions.append(cleaned_canonical)

        name_chunks = re.findall(
            r"\b(?:[A-Z][a-z0-9'/-]+(?:\s+[A-Z][a-z0-9'/-]+)+)\b",
            q,
        )
        mentions.extend(self._clean_mention(chunk) for chunk in name_chunks)

        if not mentions:
            mentions.extend(
                self._clean_mention(match)
                for match in re.findall(r"\b[A-Z][a-zA-Z0-9_/-]*\b", q)
            )

        deduped: list[str] = []
        seen: set[str] = set()
        for mention in mentions:
            if not self._keep_mention(mention, q):
                continue
            mk = AliasRegistry.normalize(mention)
            if mk and mk not in seen:
                deduped.append(mention)
                seen.add(mk)
        return deduped

    def resolve_mentions(
        self,
        query: str,
        candidate_entities: Iterable[str] | None = None,
    ) -> list["ResolvedEntity"]:
        """Detect mentions then resolve each to a canonical entity with confidence."""
        mentions = self.detect_mentions(query, candidate_entities)
        from cke.pipeline.types import ResolvedEntity

        resolved_entities: list[ResolvedEntity] = []

        for mention in mentions:
            result = self.resolve_with_score(mention)
            aliases_matched: list[str] = []
            if result.confidence >= 0.90:
                aliases_matched = self.registry.aliases_for(result.canonical)
            if not aliases_matched:
                aliases_matched = [mention]

            resolved_entities.append(
                ResolvedEntity(
                    surface_form=mention,
                    canonical_name=result.canonical,
                    entity_id=AliasRegistry.normalize(result.canonical),
                    link_confidence=result.confidence,
                    aliases_matched=sorted(set(aliases_matched)),
                )
            )

        return resolved_entities

    # ------------------------------------------------------------------
    # Query-time entity extraction (ported from router/entity_linker)
    # ------------------------------------------------------------------

    def extract_entities(self, query: str) -> list[str]:
        """Extract seed entity mentions from a natural-language query.

        Three-stage pipeline:
        1. Graph entity string matching (case-insensitive word boundary).
        2. Capitalised phrase detection (simple NER proxy).
        3. Context-clue matching — if relation/object tokens appear in the query,
           include the connected subject entity.
        """
        candidates: set[str] = set()
        query_lower = query.lower()

        known_entities = self._graph_candidates()

        # Stage 1: graph entity string matching.
        for entity in known_entities:
            if self._entity_in_query(entity, query):
                candidates.add(entity)

        # Stage 2: capitalised phrase detection.
        for phrase in re.findall(
            r"\b(?:[A-Z][\w-]*)(?:\s+[A-Z][\w-]*)*\b", query
        ):
            phrase = self._clean_phrase(phrase)
            if self._keep_phrase(phrase, query):
                candidates.add(phrase)

        # Stage 3: context clue matching.
        if self.graph_engine is not None:
            clue_tokens = set(re.findall(r"[a-z0-9_+-]+", query_lower))
            for entity in known_entities:
                for statement in self.graph_engine.get_neighbors(entity):
                    object_text = statement.object.strip()
                    if len(object_text) == 1:
                        continue
                    object_tokens = set(
                        re.findall(r"[a-z0-9_+-]+", statement.object.lower())
                    )
                    if clue_tokens.intersection(object_tokens):
                        candidates.add(entity)

        return sorted(candidates)

    # ------------------------------------------------------------------
    # Entity merging
    # ------------------------------------------------------------------

    def merge_entities(self, entity_a: str, entity_b: str) -> str:
        canonical_a = self.resolve_entity(entity_a)
        canonical_b = self.resolve_entity(entity_b)
        if canonical_a == canonical_b:
            return canonical_a

        survivor = min([canonical_a, canonical_b], key=len)
        removed = canonical_b if survivor == canonical_a else canonical_a

        for alias in self.registry.aliases_for(removed):
            self.register_alias(alias, survivor)

        if removed in self._canonical_entities:
            self._canonical_entities.remove(removed)
        self._canonical_entities.add(survivor)
        return survivor

    def known_entities(self) -> Iterable[str]:
        return sorted(self._canonical_entities)

    # ------------------------------------------------------------------
    # Private: normalisation helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _normalize(text: str) -> str:
        lowered = str(text).lower().replace("_", " ")
        cleaned = re.sub(r"[^\w\s]", " ", lowered)
        return re.sub(r"\s+", " ", cleaned).strip()

    @staticmethod
    def _canonical_key(text: str) -> str:
        normalized = AliasRegistry.normalize(text).replace("_", " ").replace("-", " ")
        tokens = [
            tok
            for tok in re.findall(r"[a-z0-9]+", normalized)
            if tok not in {"db", "database", "server"}
        ]
        return " ".join(tokens)

    @staticmethod
    def _title_case_entity(text: str) -> str:
        clean = re.sub(r"\s+", " ", text.strip())
        if clean.isupper() or len(clean) <= 5:
            return clean
        return " ".join(part.capitalize() for part in clean.split())

    # ------------------------------------------------------------------
    # Private: mention detection helpers
    # ------------------------------------------------------------------

    def _mention_in_query(self, mention: str, query: str) -> bool:
        if len(mention) == 1:
            return re.search(rf"\b{re.escape(mention)}\b", query) is not None
        return (
            re.search(rf"\b{re.escape(mention)}\b", query, flags=re.IGNORECASE)
            is not None
        )

    def _clean_mention(self, mention: str) -> str:
        tokens = str(mention).strip(" ?!.,").split()
        while tokens and tokens[0].lower() in self._LEADING_SCAFFOLD:
            tokens = tokens[1:]
        return " ".join(tokens).strip(" ?!.,")

    def _keep_mention(self, mention: str, query: str) -> bool:
        if not mention:
            return False
        lowered = mention.lower()
        if lowered in self._QUESTION_WORDS:
            return False
        if len(mention) == 1:
            return re.search(rf"\b{re.escape(mention)}\b", query) is not None
        return True

    # Aliases used by extract_entities (ported from router/entity_linker).
    _clean_phrase = _clean_mention

    def _keep_phrase(self, phrase: str, query: str) -> bool:
        if not phrase:
            return False
        if phrase.lower() in self._QUESTION_WORDS:
            return False
        if len(phrase) == 1:
            return re.search(rf"\b{re.escape(phrase)}\b", query) is not None
        return len(phrase) > 1

    @staticmethod
    def _entity_in_query(entity: str, query: str) -> bool:
        cleaned = entity.strip()
        if not cleaned:
            return False
        if len(cleaned) == 1:
            return re.search(rf"\b{re.escape(cleaned)}\b", query) is not None
        return (
            re.search(rf"\b{re.escape(cleaned)}\b", query, flags=re.IGNORECASE)
            is not None
        )

    # ------------------------------------------------------------------
    # Private: graph candidate helpers
    # ------------------------------------------------------------------

    def _graph_candidates(self) -> list[str]:
        if self.graph_engine is not None:
            getter = getattr(self.graph_engine, "get_entities", None)
            if callable(getter):
                entities = getter()
            else:
                entities = self.graph_engine.all_entities()
            return [str(e) for e in entities if str(e).strip()]
        return sorted(self._canonical_entities)

    # ------------------------------------------------------------------
    # Private: fuzzy matching
    # ------------------------------------------------------------------

    @staticmethod
    def _fuzzy_score(left: str, right: str) -> float:
        if fuzz is not None:
            return float(fuzz.ratio(left, right) / 100.0)
        return SequenceMatcher(a=left, b=right).ratio()

    def _best_fuzzy(
        self, mention: str, candidates: list[str]
    ) -> tuple[str | None, float]:
        best: str | None = None
        best_score = 0.0
        for candidate in candidates:
            score = self._fuzzy_score(mention, self._normalize(candidate))
            if score > best_score:
                best = candidate
                best_score = score
        return best, best_score

    # ------------------------------------------------------------------
    # Private: embedding similarity
    # ------------------------------------------------------------------

    def _load_embedding_model(self) -> Any | None:
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

        # Lazy model load on first call.
        if self._embedding_model is _SENTINEL:
            self._embedding_model = self._load_embedding_model()

        if self._embedding_model is not None and np is not None:
            vector = self._embedding_model.encode(
                [text], normalize_embeddings=True
            )[0]
            out = vector.tolist()
            self._embedding_cache[key] = out
            return out

        # Bag-of-hashes fallback when SentenceTransformer is unavailable.
        vec = [0.0] * 128
        for token in re.findall(r"\w+", self._normalize(text)):
            vec[hash(token) % 128] += 1.0
        norm = math.sqrt(sum(v * v for v in vec)) or 1.0
        out = [v / norm for v in vec]
        self._embedding_cache[key] = out
        return out

    def _embedding_similarity(self, left: str, right: str) -> float:
        if self.embedding_similarity_fn is not None:
            return self.embedding_similarity_fn(left, right)
        lvec, rvec = self._embed(left), self._embed(right)
        denom = (
            math.sqrt(sum(v * v for v in lvec))
            * math.sqrt(sum(v * v for v in rvec))
        ) or 1.0
        return sum(a * b for a, b in zip(lvec, rvec)) / denom

    def _best_embedding(
        self, mention: str, candidates: list[str]
    ) -> tuple[str | None, float]:
        best: str | None = None
        best_score = 0.0
        for candidate in candidates:
            score = self._embedding_similarity(mention, candidate)
            if score > best_score:
                best = candidate
                best_score = score
        return best, best_score


# Sentinel for lazy embedding-model initialisation.
_SENTINEL = object()
