"""Select deterministic operator paths conservatively."""

from __future__ import annotations

from cke.pipeline.types import ResolvedEntity
from cke.router.query_plan import QueryPlan


class OperatorSelector:
    """Conservative deterministic operator route selection."""

    def select(
        self,
        query: str,
        query_plan: QueryPlan,
        resolved_entities: list[ResolvedEntity],
        statements,
    ) -> str | None:
        hint = (getattr(query_plan, "operator_hint", None) or "").strip().lower()
        if not hint:
            return None
        if not statements:
            return None

        entity_count = len(resolved_entities)

        if hint in {"equality", "comparison"} and entity_count >= 2:
            return "equality"
        if hint in {"count"}:
            return "count"
        if hint in {"existence"}:
            return "existence"
        if hint in {"containment", "membership"} and entity_count >= 2:
            return "containment"
        if hint in {"temporal_compare", "numeric_compare"} and entity_count >= 2:
            return hint
        return None
