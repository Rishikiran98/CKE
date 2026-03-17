"""Query decomposition utilities for multi-hop graph retrieval."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(slots=True)
class QueryStep:
    """A decomposition unit representing an entity or relation hop."""

    step_type: str
    value: str
    confidence: float = 1.0


@dataclass(slots=True)
class DecomposedQuery:
    """Structured representation of a decomposed query."""

    original_query: str
    steps: list[QueryStep] = field(default_factory=list)
    operator_hint: str | None = None
    target_relations: list[str] = field(default_factory=list)


class QueryDecomposer:
    """Heuristic decomposer that derives relation chains from natural language."""

    _RELATION_PATTERNS = [
        ("starred", "starred_in"),
        ("nationality", "nationality"),
        ("citizenship", "nationality"),
        ("directed", "directed_by"),
        ("director", "directed_by"),
        ("written", "written_by"),
        ("author", "written_by"),
        ("born", "birth_date"),
        ("birth", "birth_date"),
        ("located", "located_in"),
        ("release year", "release_year"),
        ("released", "release_year"),
        ("member of", "member_of"),
        ("part of", "member_of"),
        ("children", "child"),
        ("child", "child"),
        ("employer", "employer"),
    ]

    def decompose(
        self, query: str, entities: list[str] | None = None
    ) -> DecomposedQuery:
        normalized = (query or "").strip()
        entities = entities or []

        steps: list[QueryStep] = []
        seen: set[tuple[str, str]] = set()

        for entity in entities:
            key = ("entity", entity)
            if key not in seen:
                steps.append(
                    QueryStep(step_type="entity", value=entity, confidence=0.95)
                )
                seen.add(key)

        lower_query = normalized.lower()
        inferred_relations = self._infer_target_relations(lower_query)
        for relation in inferred_relations:
            key = ("relation", relation)
            if key not in seen:
                steps.append(
                    QueryStep(step_type="relation", value=relation, confidence=0.86)
                )
                seen.add(key)

        return DecomposedQuery(
            original_query=normalized,
            steps=steps,
            operator_hint=self._detect_operator_hint(lower_query),
            target_relations=inferred_relations,
        )

    def _detect_operator_hint(self, query: str) -> str | None:
        q = f" {query} "
        if " how many" in q:
            return "count"
        if any(marker in q for marker in [" same ", " equal ", " identical "]):
            return "equality"
        if any(
            marker in q
            for marker in [
                " before ",
                " after ",
                " older ",
                " younger ",
                " later ",
                " earlier ",
            ]
        ):
            return "temporal_compare"
        if any(
            marker in q
            for marker in [
                " greater ",
                " less ",
                " more than ",
                " fewer ",
                " higher ",
                " lower ",
            ]
        ):
            return "numeric_compare"
        if any(marker in q for marker in [" member of ", " part of ", " belong to "]):
            return "containment"
        if (
            query.startswith("did ")
            or query.startswith("is ")
            or query.startswith("has ")
            or query.startswith("was ")
        ):
            return "existence"
        return None

    def _infer_target_relations(self, query: str) -> list[str]:
        found: list[str] = []
        for marker, relation in self._RELATION_PATTERNS:
            if marker in query and relation not in found:
                found.append(relation)
        return found
