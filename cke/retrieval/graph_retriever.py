"""Graph retrieval engine with BFS and beam search traversal."""

from __future__ import annotations

from collections import deque
from heapq import nlargest

from cke.graph.entity_resolver import EntityResolver
from cke.graph_engine.graph_engine import KnowledgeGraphEngine
from cke.models import Statement
from cke.router.query_plan import QueryPlan


class GraphRetriever:
    """Retrieve sparse evidence paths from the graph for a query plan."""

    def __init__(
        self,
        graph_engine: KnowledgeGraphEngine,
        entity_resolver: EntityResolver | None = None,
    ) -> None:
        self.graph_engine = graph_engine
        self.entity_resolver = entity_resolver or EntityResolver()
        for entity in self.graph_engine.all_entities():
            self.entity_resolver.register_alias(entity, entity)

    def retrieve(
        self,
        query_plan: QueryPlan,
        mode: str = "bfs",
        max_nodes: int = 200,
        beam_width: int = 4,
    ) -> dict:
        seeds = [
            self.entity_resolver.resolve(name) for name in query_plan.seed_entities
        ]
        if mode == "beam":
            paths = self._beam_search(
                seeds=seeds,
                max_depth=query_plan.max_depth,
                beam_width=beam_width,
                max_nodes=max_nodes,
            )
        else:
            paths = self._bfs_traversal(
                seeds=seeds,
                max_depth=query_plan.max_depth,
                max_nodes=max_nodes,
            )

        scored_paths = [(path, self._score_path(path)) for path in paths if path]
        scored_paths.sort(key=lambda item: item[1], reverse=True)
        evidence = self._select_evidence(
            scored_paths, max_results=query_plan.max_results
        )

        return {
            "evidence": evidence,
            "paths": [
                {
                    "score": score,
                    "assertions": [
                        {
                            "subject": st.subject,
                            "relation": st.relation,
                            "object": st.object,
                            "trust_score": st.confidence,
                        }
                        for st in path
                    ],
                }
                for path, score in scored_paths
            ],
            "entities": sorted(
                {edge["subject"] for edge in evidence}
                | {edge["object"] for edge in evidence}
            ),
        }

    def _bfs_traversal(
        self,
        seeds: list[str],
        max_depth: int,
        max_nodes: int,
    ) -> list[list[Statement]]:
        paths: list[list[Statement]] = []
        visited_depth: dict[str, int] = {}
        node_visits = 0

        queue = deque((seed, [], 0) for seed in seeds)
        while queue and node_visits < max_nodes:
            node, path, depth = queue.popleft()
            if depth > max_depth:
                continue
            previous_depth = visited_depth.get(node)
            if previous_depth is not None and depth > previous_depth:
                continue
            visited_depth[node] = depth
            node_visits += 1

            for edge in self.graph_engine.get_neighbors(node):
                next_path = path + [edge]
                paths.append(next_path)
                if depth + 1 <= max_depth and edge.object != node:
                    queue.append((edge.object, next_path, depth + 1))

        return paths

    def _beam_search(
        self,
        seeds: list[str],
        max_depth: int,
        beam_width: int,
        max_nodes: int,
    ) -> list[list[Statement]]:
        partial: list[tuple[str, list[Statement]]] = [(seed, []) for seed in seeds]
        complete_paths: list[list[Statement]] = []
        visited = 0

        for _depth in range(max_depth):
            expanded: list[tuple[str, list[Statement], float]] = []
            for node, path in partial:
                if visited >= max_nodes:
                    break
                visited += 1
                for edge in self.graph_engine.get_neighbors(node):
                    candidate = path + [edge]
                    expanded.append(
                        (edge.object, candidate, self._score_path(candidate))
                    )
                    complete_paths.append(candidate)
            if not expanded:
                break
            top = nlargest(beam_width, expanded, key=lambda item: item[2])
            partial = [(node, path) for node, path, _ in top]

        return complete_paths

    def _score_path(self, path: list[Statement]) -> float:
        if not path:
            return 0.0
        base_score = sum(float(edge.confidence) for edge in path) / len(path)

        entities = [path[0].subject] + [edge.object for edge in path]
        repeated_count = len(entities) - len(set(entities))
        repeat_penalty = 0.12 * repeated_count
        length_penalty = 0.05 * max(0, len(path) - 1)
        return max(0.0, base_score - repeat_penalty - length_penalty)

    def _select_evidence(
        self,
        scored_paths: list[tuple[list[Statement], float]],
        max_results: int,
    ) -> list[dict]:
        min_results = min(8, max_results)
        best_by_key: dict[tuple[str, str, str], tuple[float, Statement]] = {}

        for path, score in scored_paths:
            for edge in path:
                key = edge.key()
                weighted = score * float(edge.confidence)
                if key not in best_by_key or weighted > best_by_key[key][0]:
                    best_by_key[key] = (weighted, edge)

        ranked = sorted(best_by_key.values(), key=lambda item: item[0], reverse=True)

        selected: list[dict] = []
        covered_entities: set[str] = set()
        for _, edge in ranked:
            if len(selected) >= max_results:
                break
            has_new = (
                edge.subject not in covered_entities
                or edge.object not in covered_entities
            )
            if not has_new and len(selected) >= min_results:
                continue
            selected.append(
                {
                    "subject": edge.subject,
                    "relation": edge.relation,
                    "object": edge.object,
                    "trust_score": float(edge.confidence),
                }
            )
            covered_entities.update([edge.subject, edge.object])

        return selected
