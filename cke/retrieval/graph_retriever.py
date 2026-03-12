"""Graph retrieval engine with BFS and beam search traversal."""

from __future__ import annotations

from collections import deque
from heapq import nlargest
from itertools import count

from cke.graph.entity_resolver import EntityResolver
from cke.graph_engine.graph_engine import KnowledgeGraphEngine
from cke.models import Statement
from cke.observability.system_monitor import SystemMonitor
from cke.router.query_plan import QueryPlan


class GraphRetriever:
    """Retrieve sparse evidence paths from the graph for a query plan."""

    def __init__(
        self,
        graph_engine: KnowledgeGraphEngine,
        entity_resolver: EntityResolver | None = None,
        monitor: SystemMonitor | None = None,
    ) -> None:
        self.graph_engine = graph_engine
        self.entity_resolver = entity_resolver or EntityResolver()
        self.monitor = monitor
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
        intent = (query_plan.intent or "").lower()

        if intent == "definition":
            scored_paths = self._neighborhood_mode(
                seeds=seeds,
                max_results=query_plan.max_results,
            )
        elif intent == "multi-hop":
            scored_paths = self._path_mode(
                seeds=seeds,
                max_depth=query_plan.max_depth,
                mode=mode,
                max_nodes=max_nodes,
                beam_width=beam_width,
            )
        elif intent == "comparison":
            scored_paths = self._bridge_mode(
                seeds=seeds,
                max_depth=query_plan.max_depth,
            )
        else:
            scored_paths = self._path_mode(
                seeds=seeds,
                max_depth=query_plan.max_depth,
                mode=mode,
                max_nodes=max_nodes,
                beam_width=beam_width,
            )

        if self.monitor:
            traversed_nodes = sum(len(path) for path, _ in scored_paths)
            self.monitor.record_retrieval(
                steps=len(scored_paths),
                nodes_traversed=traversed_nodes,
            )

        evidence = self._select_evidence(
            scored_paths, max_results=query_plan.max_results
        )

        return {
            "evidence": evidence,
            "paths": [
                {
                    "score": score,
                    "assertions": [self._statement_payload(st) for st in path],
                }
                for path, score in scored_paths
            ],
            "entities": sorted(
                {edge["subject"] for edge in evidence}
                | {edge["object"] for edge in evidence}
            ),
        }

    def _path_mode(
        self,
        seeds: list[str],
        max_depth: int,
        mode: str,
        max_nodes: int,
        beam_width: int,
    ) -> list[tuple[list[Statement], float]]:
        if mode == "beam":
            paths = self._beam_search(
                seeds=seeds,
                max_depth=max_depth,
                beam_width=beam_width,
                max_nodes=max_nodes,
            )
        else:
            paths = self._bfs_traversal(
                seeds=seeds,
                max_depth=max_depth,
                max_nodes=max_nodes,
            )
        scored_paths = [(path, self._score_path(path)) for path in paths if path]
        scored_paths.sort(key=lambda item: item[1], reverse=True)
        return scored_paths

    def _neighborhood_mode(
        self,
        seeds: list[str],
        max_results: int,
    ) -> list[tuple[list[Statement], float]]:
        scored: list[tuple[list[Statement], float]] = []
        expanded = 0
        for seed in seeds:
            neighbors = self.graph_engine.get_neighbors(seed)
            expanded += len(neighbors)
            for edge in neighbors:
                scored.append(([edge], self._score_neighbor(edge)))

        if self.monitor:
            self.monitor.record_neighborhood_expansion(expanded)

        scored.sort(key=lambda item: item[1], reverse=True)
        return scored[: max(1, min(max_results, 12))]

    def _bridge_mode(
        self,
        seeds: list[str],
        max_depth: int,
    ) -> list[tuple[list[Statement], float]]:
        if len(seeds) < 2:
            return []

        left = seeds[0]
        right = seeds[1]

        left_paths = self._paths_from_seed(left, max_depth=max_depth)
        right_paths = self._paths_from_seed(right, max_depth=max_depth)

        right_by_node: dict[str, list[list[Statement]]] = {}
        for path in right_paths:
            bridge_node = path[-1].object
            right_by_node.setdefault(bridge_node, []).append(path)

        candidates: list[tuple[list[Statement], float]] = []
        bridge_nodes_found: set[str] = set()
        for left_path in left_paths:
            bridge_node = left_path[-1].object
            for right_path in right_by_node.get(bridge_node, []):
                bridge_nodes_found.add(bridge_node)
                candidate = left_path + self._invert_path(right_path)
                candidates.append((candidate, self._score_bridge(candidate)))

        if self.monitor:
            self.monitor.record_bridge_nodes(len(bridge_nodes_found))

        candidates.sort(key=lambda item: item[1], reverse=True)
        return candidates

    def _paths_from_seed(self, seed: str, max_depth: int) -> list[list[Statement]]:
        queue = deque([(seed, [], 0)])
        paths: list[list[Statement]] = []
        visited: set[tuple[str, int]] = set()

        while queue:
            node, path, depth = queue.popleft()
            if depth >= max_depth:
                continue
            marker = (node, depth)
            if marker in visited:
                continue
            visited.add(marker)

            for edge in self.graph_engine.get_neighbors(node):
                next_path = path + [edge]
                paths.append(next_path)
                queue.append((edge.object, next_path, depth + 1))

        return paths

    def _invert_path(self, path: list[Statement]) -> list[Statement]:
        return [
            Statement(
                subject=edge.object,
                relation=f"inverse_{edge.relation}",
                object=edge.subject,
                context=edge.context,
                confidence=edge.confidence,
                source=edge.source,
                timestamp=edge.timestamp,
            )
            for edge in reversed(path)
        ]

    def _score_neighbor(self, edge: Statement) -> float:
        relation_relevance = (
            1.0 if edge.relation not in {"related_to", "inverse_related_to"} else 0.7
        )
        novelty = 1.0
        return float(edge.confidence) * relation_relevance * novelty

    def _score_bridge(self, path: list[Statement]) -> float:
        if not path:
            return 0.0
        path_trust = sum(float(edge.confidence) for edge in path) / len(path)
        bridge_nodes = {edge.object for edge in path[:-1]} & {
            edge.subject for edge in path[1:]
        }
        novelty = 1.0 / max(1, len(bridge_nodes))
        return path_trust * novelty

    def _statement_payload(
        self, st: Statement, index: int | None = None
    ) -> dict[str, str | float]:
        statement_id = (
            "::".join(st.key()) if index is None else f"{index}:{'::'.join(st.key())}"
        )
        return {
            "id": statement_id,
            "subject": st.subject,
            "relation": st.relation,
            "object": st.object,
            "trust": float(st.confidence),
            "trust_score": float(st.confidence),
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
        cap = min(max_results, 12)
        min_results = min(8, cap)
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
        id_counter = count(1)
        for _, edge in ranked:
            if len(selected) >= cap:
                break
            has_new = (
                edge.subject not in covered_entities
                or edge.object not in covered_entities
            )
            if not has_new and len(selected) >= min_results:
                continue
            selected.append(self._statement_payload(edge, next(id_counter)))
            covered_entities.update([edge.subject, edge.object])

        return selected
