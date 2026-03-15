"""Mapping extracted relation text to canonical ontology labels."""

from __future__ import annotations

from difflib import SequenceMatcher
from pathlib import Path

try:
    import yaml
except Exception:  # pragma: no cover
    yaml = None


class RelationMapper:
    """Map free-text relation labels to canonical relation ontology names."""

    def __init__(self, schema_path: str | None = None, threshold: float = 0.78) -> None:
        self.threshold = threshold
        self.schema_path = schema_path or str(
            Path(__file__).with_name("relations.yaml")
        )
        self.relations = self._load_relations()
        self.alias_to_relation = self._build_alias_index(self.relations)

    def map(self, relation: str) -> str:
        normalized = self._normalize(relation)
        if not normalized:
            return str(relation).strip()
        if normalized in self.alias_to_relation:
            return self.alias_to_relation[normalized]

        best_alias = None
        best_score = 0.0
        for alias in self.alias_to_relation:
            score = SequenceMatcher(a=normalized, b=alias).ratio()
            if score > best_score:
                best_alias = alias
                best_score = score

        if best_alias and best_score >= self.threshold:
            return self.alias_to_relation[best_alias]
        return normalized

    @staticmethod
    def _normalize(label: str) -> str:
        return "_".join(str(label).lower().replace("-", " ").split())

    def _load_relations(self) -> dict[str, dict[str, list[str]]]:
        if yaml is None:
            return {
                "directed": {"aliases": ["directed", "director_of", "was_director_of"]},
                "acted_in": {"aliases": ["starred_in", "acted_in", "featured_in"]},
            }
        with open(self.schema_path, "r", encoding="utf-8") as handle:
            payload = yaml.safe_load(handle) or {}
        return payload.get("relations", {})

    def _build_alias_index(
        self, relations: dict[str, dict[str, list[str]]]
    ) -> dict[str, str]:
        alias_map: dict[str, str] = {}
        for canonical, config in relations.items():
            normalized_canonical = self._normalize(canonical)
            alias_map[normalized_canonical] = normalized_canonical
            for alias in config.get("aliases", []):
                alias_map[self._normalize(alias)] = normalized_canonical
        return alias_map
