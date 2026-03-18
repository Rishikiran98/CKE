"""Generate shallow candidate paths from a local subgraph."""

from __future__ import annotations

from cke.entity_resolution.alias_registry import AliasRegistry
from cke.retrieval.path_types import CandidatePath, LocalSubgraph
from cke.router.query_plan import QueryPlan


class CandidatePathGenerator:
    """Create 1-hop and 2-hop path candidates from shallow evidence graphs."""

    _BRIDGE_MARKERS = (
        "through",
        "via",
        "connected",
        "between",
        "what links",
        "which person",
        "who",
        "the woman who",
        "the film directed by the person who",
        "associated with the character portrayed by",
    )

    def generate(
        self,
        subgraph: LocalSubgraph,
        query: str,
        query_plan: QueryPlan,
    ) -> list[CandidatePath]:
        paths: list[CandidatePath] = []
        for edge in subgraph.edges:
            paths.append(
                CandidatePath(
                    statements=[edge],
                    path_score=0.0,
                    entity_overlap_score=0.0,
                    relation_match_score=0.0,
                    trust_score=0.0,
                    summary=self._summarize([edge]),
                )
            )

        for left in subgraph.edges:
            for right in subgraph.edges:
                if left is right:
                    continue
                if not self._is_chain(left, right):
                    continue
                path = [left, right]
                if self._should_keep_path(path, query, query_plan, subgraph):
                    paths.append(
                        CandidatePath(
                            statements=path,
                            path_score=0.0,
                            entity_overlap_score=0.0,
                            relation_match_score=0.0,
                            trust_score=0.0,
                            summary=self._summarize(path),
                        )
                    )

        return self._dedupe(paths)

    def _is_chain(self, left, right) -> bool:
        left_targets = {
            AliasRegistry.normalize(left.object),
            AliasRegistry.normalize(left.canonical_object_id or ""),
        }
        right_sources = {
            AliasRegistry.normalize(right.subject),
            AliasRegistry.normalize(right.canonical_subject_id or ""),
        }
        return bool({value for value in left_targets if value} & right_sources)

    def _should_keep_path(
        self,
        path,
        query: str,
        query_plan: QueryPlan,
        subgraph: LocalSubgraph,
    ) -> bool:
        if len(path) == 1:
            return True

        if not self._bridge_query(query, query_plan):
            return True

        seed_norms = {
            AliasRegistry.normalize(entity)
            for entity in subgraph.seed_entities
            if entity
        }
        if len(seed_norms) < 2:
            return True

        path_entities = set()
        for statement in path:
            path_entities.update(
                [
                    AliasRegistry.normalize(statement.subject),
                    AliasRegistry.normalize(statement.object),
                    AliasRegistry.normalize(statement.canonical_subject_id or ""),
                    AliasRegistry.normalize(statement.canonical_object_id or ""),
                ]
            )
        return len({entity for entity in path_entities if entity} & seed_norms) >= 2

    def _bridge_query(self, query: str, query_plan: QueryPlan) -> bool:
        if getattr(query_plan, "multi_hop_hint", False) or getattr(
            query_plan, "bridge_entities_expected", False
        ):
            return True
        lowered = query.lower()
        return any(marker in lowered for marker in self._BRIDGE_MARKERS)

    def _dedupe(self, paths: list[CandidatePath]) -> list[CandidatePath]:
        deduped: list[CandidatePath] = []
        seen: set[tuple[tuple[str, str, str], ...]] = set()
        for path in paths:
            key = tuple(statement.key() for statement in path.statements)
            if key in seen:
                continue
            seen.add(key)
            deduped.append(path)
        return deduped

    def _summarize(self, statements) -> str:
        return " -> ".join(statement.as_text() for statement in statements)
