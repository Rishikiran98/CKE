"""Deterministic canonical/alias registry for entity resolution."""

from __future__ import annotations

import re


class AliasRegistry:
    """Bidirectional alias mapping with deterministic normalization."""

    def __init__(self) -> None:
        self.alias_to_canonical: dict[str, str] = {}
        self.canonical_to_aliases: dict[str, set[str]] = {}
        self._normalized_canonical: dict[str, str] = {}

    @staticmethod
    def normalize(name: str) -> str:
        lowered = str(name).lower().strip()
        lowered = lowered.replace(".", "")
        lowered = re.sub(r"[^\w\s-]", " ", lowered)
        lowered = re.sub(r"\s+", " ", lowered)
        return lowered.strip(" -_")

    def add(self, canonical: str, aliases: list[str]) -> None:
        canonical_name = str(canonical).strip()
        if not canonical_name:
            return

        norm_canonical = self.normalize(canonical_name)
        self.alias_to_canonical[norm_canonical] = canonical_name
        self.canonical_to_aliases.setdefault(canonical_name, set()).add(canonical_name)
        self._normalized_canonical[norm_canonical] = canonical_name

        for alias in aliases:
            alias_name = str(alias).strip()
            if not alias_name:
                continue
            norm_alias = self.normalize(alias_name)
            self.alias_to_canonical[norm_alias] = canonical_name
            self.canonical_to_aliases[canonical_name].add(alias_name)

    def resolve(self, name: str) -> str | None:
        normalized = self.normalize(name)
        if normalized in self.alias_to_canonical:
            return self.alias_to_canonical[normalized]
        return self._normalized_canonical.get(normalized)

    def aliases_for(self, canonical: str) -> list[str]:
        values = self.canonical_to_aliases.get(str(canonical).strip(), set())
        return sorted(values)

