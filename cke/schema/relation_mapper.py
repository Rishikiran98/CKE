"""Mapping extracted relation text to canonical ontology labels.

Without PyYAML the ontology cannot be read and collapses to two hardcoded
relations, so nearly every relation passes through unmapped. That is a large
change in behaviour from a missing dependency, and it is declared rather than
absorbed.
"""

from __future__ import annotations

from difflib import SequenceMatcher
from pathlib import Path

from cke.diagnostics import DegradationMixin

try:
    import yaml
except ImportError:  # pragma: no cover - optional runtime dependency
    yaml = None


#: Used only when PyYAML is absent. Two relations out of the full ontology.
_MINIMAL_RELATIONS: dict[str, dict[str, list[str]]] = {
    "directed": {"aliases": ["directed", "director_of", "was_director_of"]},
    "acted_in": {"aliases": ["starred_in", "acted_in", "featured_in"]},
}


class RelationMapper(DegradationMixin):
    """Map free-text relation labels to canonical relation ontology names.

    Args:
        schema_path: path to the relation ontology YAML.
        threshold: minimum fuzzy-match ratio to accept an alias.
        strict: when True, raise rather than fall back to the minimal ontology.
    """

    def __init__(
        self,
        schema_path: str | None = None,
        threshold: float = 0.78,
        strict: bool = False,
    ) -> None:
        self._init_degradation(strict)
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
            self._degrade(
                "PyYAML is not installed, so the relation ontology at "
                f"{self.schema_path} cannot be read and only "
                f"{len(_MINIMAL_RELATIONS)} relations "
                f"({', '.join(sorted(_MINIMAL_RELATIONS))}) are recognised. "
                "Every other relation passes through unmapped. Install it "
                "with `pip install PyYAML`"
            )
            return dict(_MINIMAL_RELATIONS)
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
