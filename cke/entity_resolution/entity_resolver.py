"""Deterministic entity mention detection and canonical resolution."""

from __future__ import annotations

import re
from typing import TYPE_CHECKING, Iterable

from cke.entity_resolution.alias_registry import AliasRegistry

if TYPE_CHECKING:
    from cke.pipeline.types import ResolvedEntity


class EntityResolver:
    """Resolve query mentions into structured canonical entities."""

    def __init__(self, aliases: dict[str, str] | None = None) -> None:
        self.registry = AliasRegistry()
        self._canonical_entities: set[str] = set()
        if aliases:
            canonical_to_aliases: dict[str, list[str]] = {}
            for alias, canonical in aliases.items():
                canonical_to_aliases.setdefault(canonical, []).append(alias)
            for canonical, values in canonical_to_aliases.items():
                self.register_aliases(canonical, values)

    def register_alias(self, alias: str, canonical: str) -> None:
        self.register_aliases(canonical, [alias])

    def register_aliases(self, canonical: str, aliases: list[str]) -> None:
        canonical_name = str(canonical).strip()
        if not canonical_name:
            return
        self._canonical_entities.add(canonical_name)
        self.registry.add(canonical_name, aliases)

    @staticmethod
    def _canonical_key(text: str) -> str:
        normalized = AliasRegistry.normalize(text).replace("_", " ").replace("-", " ")
        tokens = [
            tok
            for tok in re.findall(r"[a-z0-9]+", normalized)
            if tok not in {"db", "database", "server"}
        ]
        return " ".join(tokens)

    def resolve_entity(self, name: str) -> str:
        resolved = self.registry.resolve(name)
        if resolved:
            return resolved
        normalized = AliasRegistry.normalize(name)
        key = self._canonical_key(name)
        for canonical in self._canonical_entities:
            if AliasRegistry.normalize(canonical) == normalized or self._canonical_key(canonical) == key:
                return canonical
        canonical = self._title_case_entity(name)
        self.register_alias(name, canonical)
        return canonical

    def detect_mentions(
        self,
        query: str,
        candidate_entities: Iterable[str] | None = None,
    ) -> list[str]:
        q = query or ""
        lowered = q.lower()
        mentions: list[str] = []

        for canonical in sorted(set(candidate_entities or []), key=len, reverse=True):
            if canonical and canonical.lower() in lowered:
                mentions.append(canonical)

        for canonical, aliases in self.registry.canonical_to_aliases.items():
            for alias in aliases:
                if alias and alias.lower() in lowered:
                    mentions.append(alias)
                    if canonical.lower() in lowered:
                        mentions.append(canonical)

        name_chunks = re.findall(
            r"\b(?:[A-Z][a-z0-9'/-]+(?:\s+[A-Z][a-z0-9'/-]+)+)\b",
            q,
        )
        mentions.extend(name_chunks)

        if not mentions:
            mentions.extend(re.findall(r"\b[A-Z][a-zA-Z0-9_/-]*\b", q))

        deduped: list[str] = []
        seen: set[str] = set()
        for mention in mentions:
            key = AliasRegistry.normalize(mention)
            if key and key not in seen:
                deduped.append(mention)
                seen.add(key)
        return deduped

    def resolve_mentions(
        self,
        query: str,
        candidate_entities: Iterable[str] | None = None,
    ) -> list["ResolvedEntity"]:
        mentions = self.detect_mentions(query, candidate_entities)
        from cke.pipeline.types import ResolvedEntity

        resolved_entities: list[ResolvedEntity] = []

        for mention in mentions:
            confidence = 0.50
            aliases_matched: list[str] = []
            canonical = None

            if mention in self._canonical_entities:
                canonical = mention
                confidence = 0.95
                aliases_matched = self.registry.aliases_for(canonical)
            else:
                direct = self.registry.resolve(mention)
                if direct:
                    canonical = direct
                    confidence = 0.90
                    aliases_matched = [mention]
                else:
                    norm = AliasRegistry.normalize(mention)
                    key = self._canonical_key(mention)
                    for known in self._canonical_entities:
                        if AliasRegistry.normalize(known) == norm or self._canonical_key(known) == key:
                            canonical = known
                            confidence = 0.75
                            aliases_matched = [mention]
                            break

            if canonical is None:
                canonical = self.resolve_entity(mention)
                aliases_matched = [mention]

            resolved_entities.append(
                ResolvedEntity(
                    surface_form=mention,
                    canonical_name=canonical,
                    entity_id=AliasRegistry.normalize(canonical),
                    link_confidence=confidence,
                    aliases_matched=sorted(set(aliases_matched)),
                )
            )

        return resolved_entities

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

    @staticmethod
    def _title_case_entity(text: str) -> str:
        clean = re.sub(r"\s+", " ", text.strip())
        if clean.isupper() or len(clean) <= 5:
            return clean
        return " ".join(part.capitalize() for part in clean.split())
