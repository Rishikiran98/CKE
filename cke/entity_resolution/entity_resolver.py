"""Unified entity mention detection, canonical resolution, and linking.

Consolidates query-time detection (formerly ``cke.router.entity_linker``),
alias + normalisation resolution, fuzzy matching (rapidfuzz / SequenceMatcher),
and embedding similarity (SentenceTransformer) into a single resolver used by
both the ingestion pipeline and the query pipeline.
"""

from __future__ import annotations

import hashlib
import math
import re
from dataclasses import dataclass
from difflib import SequenceMatcher
from typing import TYPE_CHECKING, Any, Callable, Iterable

from cke.diagnostics import (
    DegradationMixin,
    record_loaded_model,
    revision_pin_problem,
)
from cke.retrieval.embedding_model import (
    DEFAULT_EMBEDDING_MODEL,
    DEFAULT_EMBEDDING_REVISION,
)
from cke.entity_resolution.alias_registry import AliasRegistry

if TYPE_CHECKING:
    from cke.pipeline.types import ResolvedEntity

# ---------------------------------------------------------------------------
# Optional heavy dependencies – graceful degradation when absent.
# ---------------------------------------------------------------------------

try:
    from rapidfuzz import fuzz  # type: ignore[import-untyped]
except ImportError:  # pragma: no cover – optional dependency
    fuzz = None

try:
    import numpy as np  # type: ignore[import-untyped]
except ImportError:  # pragma: no cover
    np = None

try:
    from sentence_transformers import SentenceTransformer  # type: ignore[import-untyped]  # noqa: E501
except ImportError:  # pragma: no cover
    SentenceTransformer = None


#: The same embedder, at the same commit, as :mod:`cke.retrieval.embedding_model`
#: loads. It used to be named here without the organisation prefix and without a
#: revision, which made it a second, separately-resolved copy of the same
#: weights whose identity no run could state.
_EMBEDDING_MODEL_NAME = DEFAULT_EMBEDDING_MODEL
_EMBEDDING_MODEL_REVISION = DEFAULT_EMBEDDING_REVISION

#: Loaded models, keyed by name, shared across every resolver in the process.
#: A resolver is constructed per GraphRetriever, and a benchmark builds one per
#: item, so without this the transformer was loaded from disk three hundred
#: times in a run and every latency figure measured the load. EmbeddingModel
#: already caches this way; this is the same fix on the other loader.
_MODEL_CACHE: dict[str, Any] = {}

#: Names whose load already failed, with the reason. A failure is declared once
#: per resolver, as before, but retried no more than once per process.
_FAILED_MODEL_LOADS: dict[str, str] = {}

#: Width of the hashed fallback vector. Not a semantic embedding dimension.
_FALLBACK_DIM = 128

#: Words that name nothing on their own. A mention made only of these cannot
#: be resolved by containment: "the" sits inside most entity names in a graph,
#: and matching on it would attach a query to whichever one happened to be
#: unique.
_UNINFORMATIVE_TOKENS = frozenset(
    """a an the of in on at to for from by with and or is are was were be been
    what which who whom whose when where why how that this these those it its
    his her their he she they i we you as also than then""".split()
)


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


class EntityResolver(DegradationMixin):
    """Resolve entity mentions to canonical names and extract entities from queries.

    Resolution chain (in order):
        1. Exact canonical match  → confidence 0.95
        2. Alias-registry lookup  → confidence 0.90
        3. Normalised / key match → confidence 0.75
        4. Fuzzy string match     → confidence = fuzzy score  (≥ *fuzzy_threshold*)
        5. Unique containment     → confidence 0.65
        6. Embedding similarity   → confidence = emb score    (≥ *embedding_threshold*)
        7. Title-case fallback    → confidence = max(fuzzy, emb, 0.50)
    """

    #: Candidates returned per mention, and in total, by :meth:`expand`.
    #: Carried over from the seed expansion in the benchmark driver that this
    #: replaces, so that swapping one for the other changes the matching rule
    #: and not the breadth. They are the reason a mention that names something
    #: very generic cannot flood a query with the whole graph.
    EXPANSION_PER_MENTION = 3
    EXPANSION_TOTAL = 6

    #: Confidence of a containment match. Below a normalised match (0.75),
    #: because one name sitting inside another is evidence that they name the
    #: same thing and not proof of it; above the title-case fallback (0.50),
    #: because the fallback is what happens when nothing matched at all.
    CONTAINMENT_CONFIDENCE = 0.65

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
        strict: bool = False,
    ) -> None:
        self._init_degradation(strict)
        self.registry = AliasRegistry()
        self._canonical_entities: set[str] = set()
        #: What a mention resolved to, and at what confidence, the first time
        #: it was asked. Every rung below except containment already cached
        #: its answer by registering an alias, which made the mention
        #: canonical — and the canonical rung reports 0.95, not the
        #: confidence of the rung that actually did the work. So the same
        #: mention scored one number the first time a run met it and another
        #: on every repeat, and a benchmark scored the same entity
        #: differently depending on which item it appeared in.
        self._resolution_cache: dict[str, "ResolutionResult"] = {}
        self.graph_engine = graph_engine
        self.fuzzy_threshold = fuzzy_threshold
        self.embedding_threshold = embedding_threshold
        self.embedding_similarity_fn = embedding_similarity_fn

        # Embedding model is loaded lazily on first call to _embed().
        self._embedding_model: Any | None = _SENTINEL
        self._embedding_cache: dict[str, list[float]] = {}

        if fuzz is None:
            # Known at construction, so declare it here rather than from
            # inside the per-candidate scoring loop.
            self._degrade(
                "rapidfuzz is not installed, so fuzzy matching uses "
                "difflib.SequenceMatcher, a different algorithm whose ratios "
                "are not comparable to rapidfuzz's. The fuzzy threshold "
                f"({self.fuzzy_threshold}) was not calibrated for it. Install "
                "it with `pip install rapidfuzz`"
            )

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
        # A caller registering an alias knows something this resolver did not
        # when it cached its answer, so the cached answer is stale.
        self._resolution_cache.pop(canonical_name, None)
        for alias in aliases:
            alias_name = str(alias).strip()
            self._resolution_cache.pop(alias_name, None)
            # A name that is an alias for something else is not a canonical
            # entity of its own. The unresolvable rungs answer with the
            # mention's own title case and register that, which made every
            # mention nobody could place a canonical entity — and the
            # exact-canonical rung is checked before the alias registry, so
            # once a mention had been guessed at, saying what it actually
            # meant could not change the answer. merge_entities has always
            # dropped the name it merged away; this is the same rule applied
            # wherever an alias is registered.
            if alias_name != canonical_name:
                self._canonical_entities.discard(alias_name)

    def _remember(self, mention: str, result: "ResolutionResult") -> "ResolutionResult":
        """Cache a resolution under the mention that produced it.

        Registering the alias is what the rungs below already did. Recording
        the confidence beside it is what stops the repeat of a mention being
        scored 0.95 by the canonical rung instead of by the rung that
        resolved it.
        """
        self.register_alias(mention, result.canonical)
        self._resolution_cache[mention] = result
        return result

    # ------------------------------------------------------------------
    # Core resolution — single entity
    # ------------------------------------------------------------------

    def resolve_with_score(self, entity: str) -> ResolutionResult:
        """Full resolution chain returning canonical name + confidence."""
        name = str(entity).strip()
        normalized = self._normalize(name)
        if not normalized:
            return ResolutionResult(canonical=name or "", confidence=0.0)

        # 0. What this mention resolved to last time. Ahead of every rung
        # below, because the rungs that cache register the mention as an
        # alias and the canonical rung would then answer 0.95 for it.
        remembered = self._resolution_cache.get(name)
        if remembered is not None:
            return remembered

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
                return self._remember(
                    name, ResolutionResult(canonical=known, confidence=0.75)
                )

        # 4-6. Fuzzy → embedding → fallback (need candidates)
        candidates = self._graph_candidates()
        if not candidates:
            canonical = self._title_case_entity(name)
            return self._remember(
                name, ResolutionResult(canonical=canonical, confidence=0.50)
            )

        best_fuzzy_candidate, best_fuzzy = self._best_fuzzy(normalized, candidates)
        if best_fuzzy_candidate and best_fuzzy >= self.fuzzy_threshold:
            return self._remember(
                name, ResolutionResult(best_fuzzy_candidate, confidence=best_fuzzy)
            )

        contained = self._unique_container(name, candidates)
        if contained:
            # Deliberately not registered as an alias. Every other rung caches
            # its answer, but this one is conditional on the candidate set: a
            # mention with exactly one container today may have two once more
            # text is ingested, and a cached alias would hide that the answer
            # had become ambiguous.
            return ResolutionResult(contained, confidence=self.CONTAINMENT_CONFIDENCE)

        best_emb_candidate, best_emb = self._best_embedding(name, candidates)
        if best_emb_candidate and best_emb >= self.embedding_threshold:
            return self._remember(
                name, ResolutionResult(best_emb_candidate, confidence=best_emb)
            )

        canonical = self._title_case_entity(name)
        return self._remember(
            name,
            ResolutionResult(
                canonical=canonical,
                confidence=max(best_fuzzy, best_emb, 0.50),
            ),
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
        """Detect entity mentions in *query* using candidates, registry,
        and NER heuristics."""
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
        for phrase in re.findall(r"\b(?:[A-Z][\w-]*)(?:\s+[A-Z][\w-]*)*\b", query):
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
    # Fan-out: one mention to several entities
    # ------------------------------------------------------------------

    def expand(
        self,
        mentions: Iterable[str],
        per_mention: int | None = None,
        total: int | None = None,
    ) -> list[str]:
        """Map query mentions onto the graph entities that could carry them.

        :meth:`resolve` answers "which entity is this name?", and has to pick
        one, so it refuses a mention with several containers rather than
        guessing. Retrieval asks a different question — "where in the graph
        should this query start?" — and there several starting points are
        better than none. A question mentioning "Kansas City" against a graph
        holding both "Kansas City jazz" and "Kansas City Chiefs" should seed
        both and let path scoring choose.

        Ordered by how much of the candidate the mention covers, so the
        tightest fit comes first: "RESP" covers half of "RESP protocol" and a
        tenth of a clause that merely contains it. Ties break on the name, so
        a run is reproducible.

        A mention that expands to nothing keeps its resolved form, which is
        what :meth:`resolve` would have returned on its own. Expansion adds
        starting points; it never removes the one already there.
        """
        total = self.EXPANSION_TOTAL if total is None else total

        expanded: list[str] = []
        seen: set[str] = set()
        for group in self.expand_groups(mentions, per_mention=per_mention):
            for match in group:
                if match not in seen:
                    seen.add(match)
                    expanded.append(match)
        return expanded[:total]

    def expand_groups(
        self, mentions: Iterable[str], per_mention: int | None = None
    ) -> list[list[str]]:
        """:meth:`expand`, but keeping each mention's candidates together.

        Comparison retrieval bridges between two *different* mentions, so it
        cannot use the flattened list: with mentions "Alpha" and "Beta" over a
        graph also holding "Alpha Group", flattening puts two expansions of the
        first mention in the first two positions, and the bridge compares Alpha
        against Alpha Group while ignoring Beta entirely.
        """
        per_mention = self.EXPANSION_PER_MENTION if per_mention is None else per_mention
        candidates = self._graph_candidates()
        known = set(candidates)

        groups: list[list[str]] = []
        for mention in mentions:
            group = self._containers_by_fit(mention, candidates)[:per_mention]

            # The resolved form is kept, never traded away for a container. An
            # alias registered as "NYC" -> "New York City" is a stronger
            # statement about what the mention names than the fact that some
            # entity happens to hold "NYC" inside it; dropping it sent
            # retrieval to "NYC Department" and nowhere else. It leads when it
            # names a real entity. When it does not — a title-cased fallback
            # for a mention nothing matched — it is worth carrying only if
            # there is nothing else, so the mention still reaches the caller.
            resolved = self.resolve(mention)
            if resolved in known:
                group = [resolved] + [c for c in group if c != resolved]
            elif not group and resolved:
                group = [resolved]

            # The cap bounds the whole group, resolved form included: it exists
            # so that one generic mention cannot flood a query with entities,
            # and that reason does not care which rung produced them.
            if group:
                groups.append(group[:per_mention])
        return groups

    @classmethod
    def _containers_by_fit(cls, name: str, candidates: Iterable[str]) -> list[str]:
        """Candidates holding *name* as a phrase, tightest fit first.

        An exact match sorts first by construction: it covers the whole of
        itself, which no longer candidate can.
        """
        mention = cls._tokens(name)
        if not mention or all(token in _UNINFORMATIVE_TOKENS for token in mention):
            return []

        scored: list[tuple[float, str, str]] = []
        for candidate in candidates:
            tokens = cls._tokens(candidate)
            if not tokens or not cls._contains_phrase(tokens, mention):
                continue
            # Descending fit, then ascending name: negate the fit so one sort
            # key orders both.
            scored.append((-len(mention) / len(tokens), candidate, candidate))
        scored.sort()
        return [candidate for _, _, candidate in scored]

    # ------------------------------------------------------------------
    # Private: containment matching
    # ------------------------------------------------------------------

    @classmethod
    def _unique_container(cls, name: str, candidates: Iterable[str]) -> str | None:
        """Return the one candidate that contains *name*, or None.

        A source that names an entity both ways — "RESP" in one sentence and
        "RESP protocol" in the next — leaves a query mention that matches no
        node exactly and scores far below the fuzzy threshold, because the two
        strings differ by most of their characters. Fuzzy matching measures
        edit distance; this measures whether one name sits inside the other as
        a whole phrase.

        Ambiguity is refused rather than guessed. "mosque" is inside many
        names in a graph of buildings, and there is no basis here for choosing
        among them, so a mention with more than one container resolves to
        nothing and the caller keeps its original string. That refusal is what
        keeps the rule safe as a graph grows: the more text ingested, the more
        likely a loose mention is ambiguous, and ambiguity declines.
        """
        mention = cls._tokens(name)
        containers = [
            candidate
            for candidate in cls._containers_by_fit(name, candidates)
            if cls._tokens(candidate) != mention
        ]
        return containers[0] if len(containers) == 1 else None

    @classmethod
    def _tokens(cls, text: str) -> tuple[str, ...]:
        return tuple(re.findall(r"\w+", cls._normalize(text)))

    @staticmethod
    def _contains_phrase(haystack: tuple[str, ...], needle: tuple[str, ...]) -> bool:
        """True when *needle* appears in *haystack* as consecutive tokens.

        Consecutive, not as a subset: "Kansas City" is inside "Kansas City
        jazz" and not inside "Kansas has a city".
        """
        span = len(needle)
        if not span or span > len(haystack):
            return False
        return any(
            haystack[i : i + span] == needle for i in range(len(haystack) - span + 1)
        )

    # ------------------------------------------------------------------
    # Private: fuzzy matching
    # ------------------------------------------------------------------

    def _fuzzy_score(self, left: str, right: str) -> float:
        # The rapidfuzz degradation is declared once in __init__, not here:
        # this runs once per candidate entity.
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
            self._degrade(
                "sentence-transformers is not installed, so entity similarity "
                "is computed from hashed token counts rather than embeddings, "
                f"and the embedding threshold ({self.embedding_threshold}) was "
                "not calibrated for that. Install it with "
                "`pip install sentence-transformers`"
            )
            return None

        pin_problem = revision_pin_problem(
            _EMBEDDING_MODEL_NAME, _EMBEDDING_MODEL_REVISION
        )
        if pin_problem is not None:  # pragma: no cover - the pin is a constant
            self._degrade(pin_problem)
            return None

        cached = _MODEL_CACHE.get(_EMBEDDING_MODEL_NAME)
        if cached is not None:
            return cached

        previous_failure = _FAILED_MODEL_LOADS.get(_EMBEDDING_MODEL_NAME)
        if previous_failure is not None:
            self._degrade(previous_failure)
            return None

        try:
            model = SentenceTransformer(
                _EMBEDDING_MODEL_NAME, revision=_EMBEDDING_MODEL_REVISION
            )
        except Exception as exc:  # noqa: BLE001 - download/runtime failures vary
            reason = (
                f"sentence-transformers could not load "
                f"{_EMBEDDING_MODEL_NAME!r} ({type(exc).__name__}: {exc}), so "
                "entity similarity is computed from hashed token counts "
                "rather than embeddings"
            )
            _FAILED_MODEL_LOADS[_EMBEDDING_MODEL_NAME] = reason
            self._degrade(reason)
            return None

        _MODEL_CACHE[_EMBEDDING_MODEL_NAME] = model
        record_loaded_model(
            "EntityResolver",
            _EMBEDDING_MODEL_NAME,
            f"{_EMBEDDING_MODEL_NAME}@{_EMBEDDING_MODEL_REVISION}",
        )
        return model

    def _embed(self, text: str) -> list[float]:
        key = str(text)
        if key in self._embedding_cache:
            return self._embedding_cache[key]

        # Lazy model load on first call.
        if self._embedding_model is _SENTINEL:
            self._embedding_model = self._load_embedding_model()

        if self._embedding_model is not None and np is not None:
            vector = self._embedding_model.encode([text], normalize_embeddings=True)[0]
            out = vector.tolist()
            self._embedding_cache[key] = out
            return out

        if self._embedding_model is not None and np is None:
            self._degrade(
                "numpy is not installed, so the loaded embedding model cannot "
                "be used and entity similarity falls back to hashed token "
                "counts. Install it with `pip install numpy`"
            )

        # Bag-of-hashes fallback when SentenceTransformer is unavailable.
        # Hashing is SHA256 rather than the builtin hash(): builtin string
        # hashing is salted per process, which made resolution differ between
        # two runs of the same commit on the same data.
        vec = [0.0] * _FALLBACK_DIM
        for token in re.findall(r"\w+", self._normalize(text)):
            digest = hashlib.sha256(token.encode("utf-8")).hexdigest()
            vec[int(digest, 16) % _FALLBACK_DIM] += 1.0
        norm = math.sqrt(sum(v * v for v in vec)) or 1.0
        out = [v / norm for v in vec]
        self._embedding_cache[key] = out
        return out

    def _embedding_similarity(self, left: str, right: str) -> float:
        if self.embedding_similarity_fn is not None:
            return self.embedding_similarity_fn(left, right)
        lvec, rvec = self._embed(left), self._embed(right)
        denom = (
            math.sqrt(sum(v * v for v in lvec)) * math.sqrt(sum(v * v for v in rvec))
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
