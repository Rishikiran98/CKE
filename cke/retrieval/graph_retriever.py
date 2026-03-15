"""Graph retrieval engine with BFS, beam, and A* traversal plus path ranking."""

from __future__ import annotations

from collections import deque
import re
from heapq import heappop, heappush, nlargest
from itertools import count

from cke.graph.entity_resolver import EntityResolver
from cke.graph_engine.graph_engine import KnowledgeGraphEngine
from cke.models import Statement
from cke.observability.system_monitor import SystemMonitor
from cke.retrieval.evidence_graph import EvidenceGraph
from cke.retrieval.path_ranking import (
    PathFeatures,
    PathRankingModel,
    relation_match_score,
)
from cke.router.query_plan import QueryPlan


class GraphRetriever:
    """Retrieve sparse evidence paths from the graph for a query plan."""

    def __init__(
        self,
        graph_engine: KnowledgeGraphEngine,
        entity_resolver: EntityResolver | None = None,
        monitor: SystemMonitor | None = None,
        path_ranker: PathRankingModel | None = None,
    ) -> None:
        self.graph_engine = graph_engine
        self.entity_resolver = entity_resolver or EntityResolver()
        self.monitor = monitor
        self.path_ranker = path_ranker or PathRankingModel()
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
                query_text=query_plan.query_text,
                seeds_raw=query_plan.seed_entities,
                decomposition=query_plan.decomposition,
            )
        elif intent == "comparison":
            scored_paths = self._bridge_mode(
                seeds=seeds,
                max_depth=query_plan.max_depth,
                query_text=query_plan.query_text,
                seed_entities=query_plan.seed_entities,
                decomposition=query_plan.decomposition,
            )
        else:
            scored_paths = self._path_mode(
                seeds=seeds,
                max_depth=query_plan.max_depth,
                mode=mode,
                max_nodes=max_nodes,
                beam_width=beam_width,
                query_text=query_plan.query_text,
                seed_entities=query_plan.seed_entities,
                decomposition=query_plan.decomposition,
            )

        if self.monitor:
            traversed_nodes = sum(len(path) for path, _ in scored_paths)
            self.monitor.record_retrieval(
                steps=len(scored_paths), nodes_traversed=traversed_nodes
            )

        evidence = self._select_evidence(
            scored_paths, max_results=query_plan.max_results
        )
        evidence_graph = self._build_evidence_graph(scored_paths)

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
            "evidence_graph": evidence_graph.as_dict(),
        }

    def _path_mode(
        self,
        seeds,
        max_depth,
        mode,
        max_nodes,
        beam_width,
        query_text,
        seed_entities,
        decomposition,
    ):
        if mode == "beam":
            paths = self._beam_search(
                seeds, max_depth, beam_width, max_nodes, query_text, seed_entities
            )
        elif mode == "astar":
            paths = self._astar_search(
                seeds, max_depth, max_nodes, query_text, seed_entities
            )
        else:
            paths = self._bfs_traversal(
                seeds=seeds, max_depth=max_depth, max_nodes=max_nodes
            )
        scored_paths = [
            (
                path,
                self._rank_path(
                    path,
                    query_text=query_text,
                    seed_entities=seed_entities,
                    decomposition=decomposition,
                ),
            )
            for path in paths
            if path
        ]
        scored_paths.sort(key=lambda item: item[1], reverse=True)
        return scored_paths

    def _neighborhood_mode(
        self, seeds, max_results, query_text, seeds_raw, decomposition
    ):
        scored = []
        expanded = 0
        for seed in seeds:
            neighbors = self.graph_engine.get_neighbors(seed)
            expanded += len(neighbors)
            for edge in neighbors:
                scored.append(
                    (
                        [edge],
                        self._rank_path([edge], query_text, seeds_raw, decomposition),
                    )
                )
        if self.monitor:
            self.monitor.record_neighborhood_expansion(expanded)
        scored.sort(key=lambda item: item[1], reverse=True)
        return scored[: max(1, min(max_results, 12))]

    def _bridge_mode(self, seeds, max_depth, query_text, seed_entities, decomposition):
        if len(seeds) < 2:
            return []
        left_paths = self._paths_from_seed(seeds[0], max_depth=max_depth)
        right_paths = self._paths_from_seed(seeds[1], max_depth=max_depth)
        right_by_node = {}
        for path in right_paths:
            right_by_node.setdefault(path[-1].object, []).append(path)

        candidates = []
        bridge_nodes_found = set()
        for left_path in left_paths:
            bridge_node = left_path[-1].object
            for right_path in right_by_node.get(bridge_node, []):
                bridge_nodes_found.add(bridge_node)
                candidate = left_path + self._invert_path(right_path)
                candidates.append(
                    (
                        candidate,
                        self._rank_path(
                            candidate, query_text, seed_entities, decomposition
                        ),
                    )
                )
        if self.monitor:
            self.monitor.record_bridge_nodes(len(bridge_nodes_found))
        candidates.sort(key=lambda item: item[1], reverse=True)
        return candidates

    def _paths_from_seed(self, seed: str, max_depth: int) -> list[list[Statement]]:
        queue = deque([(seed, [], 0)])
        paths = []
        visited = set()
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

    def _invert_path(self, path):
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

    @staticmethod
    def _tokenize(text: str) -> set[str]:
        return set(re.findall(r"[a-z0-9]+", (text or "").lower()))

    def _edge_relevance(
        self, edge: Statement, query_tokens: set[str], seed_tokens: set[str]
    ) -> float:
        edge_tokens = (
            self._tokenize(edge.subject)
            | self._tokenize(edge.relation)
            | self._tokenize(edge.object)
        )
        if not edge_tokens:
            return 0.0
        token_overlap = len(edge_tokens & query_tokens) / len(edge_tokens)
        entity_overlap = len(
            (self._tokenize(edge.subject) | self._tokenize(edge.object)) & seed_tokens
        ) / max(1, len(seed_tokens))
        relation_relevance = (
            1.0 if edge.relation not in {"related_to", "inverse_related_to"} else 0.7
        )
        return 0.45 * token_overlap + 0.35 * entity_overlap + 0.20 * relation_relevance

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
        self, seeds: list[str], max_depth: int, max_nodes: int
    ) -> list[list[Statement]]:
        paths = []
        visited_depth = {}
        node_visits = 0
        queue = deque((seed, [], 0) for seed in seeds)
        while queue and node_visits < max_nodes:
            node, path, depth = queue.popleft()
            if depth > max_depth:
                continue
            prev = visited_depth.get(node)
            if prev is not None and depth > prev:
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
        self, seeds, max_depth, beam_width, max_nodes, query_text, seed_entities
    ):
        partial = [(seed, []) for seed in seeds]
        complete_paths = []
        visited = 0
        for _depth in range(max_depth):
            expanded = []
            for node, path in partial:
                if visited >= max_nodes:
                    break
                visited += 1
                for edge in self.graph_engine.get_neighbors(node):
                    candidate = path + [edge]
                    expanded.append(
                        (
                            edge.object,
                            candidate,
                            self._score_path(candidate, query_text, seed_entities),
                        )
                    )
                    complete_paths.append(candidate)
            if not expanded:
                break
            top = nlargest(beam_width, expanded, key=lambda item: item[2])
            partial = [(node, path) for node, path, _ in top]
        return complete_paths

    def _astar_search(self, seeds, max_depth, max_nodes, query_text, seed_entities):
        query_tokens = self._tokenize(query_text)
        seed_tokens = set().union(
            *(self._tokenize(seed) for seed in (seed_entities or []))
        )
        frontier = []
        seen_best = {}
        tick = count()
        for seed in seeds:
            heappush(frontier, (0.0, next(tick), seed, [], 0))

        results = []
        expanded = 0
        while frontier and expanded < max_nodes:
            priority, _idx, node, path, depth = heappop(frontier)
            cost = -priority
            sig = (node, depth)
            if sig in seen_best and cost <= seen_best[sig]:
                continue
            seen_best[sig] = cost
            if path:
                results.append(path)
            if depth >= max_depth:
                continue
            expanded += 1
            for edge in self.graph_engine.get_neighbors(node):
                candidate = path + [edge]
                g = self._score_path(candidate, query_text, seed_entities)
                h = self._heuristic(edge.object, query_tokens, seed_tokens)
                f = g + h
                heappush(frontier, (-f, next(tick), edge.object, candidate, depth + 1))
        return results

    def _heuristic(
        self, node: str, query_tokens: set[str], seed_tokens: set[str]
    ) -> float:
        node_tokens = self._tokenize(node)
        if not node_tokens:
            return 0.0
        overlap = len(node_tokens & (query_tokens | seed_tokens)) / len(node_tokens)
        return 0.5 * overlap

    def _score_path(
        self,
        path: list[Statement],
        query_text: str = "",
        seed_entities: list[str] | None = None,
    ) -> float:
        if not path:
            return 0.0
        base_score = sum(float(edge.confidence) for edge in path) / len(path)
        entities = [path[0].subject] + [edge.object for edge in path]
        repeated_count = len(entities) - len(set(entities))
        repeat_penalty = 0.12 * repeated_count
        length_penalty = 0.05 * max(0, len(path) - 1)
        query_tokens = self._tokenize(query_text)
        seed_entities = seed_entities or []
        seed_tokens = set().union(*(self._tokenize(seed) for seed in seed_entities))
        if not query_tokens and seed_tokens:
            query_tokens = set(seed_tokens)
        relevance_scores = [
            self._edge_relevance(edge, query_tokens, seed_tokens) for edge in path
        ]
        relevance_bonus = sum(relevance_scores) / len(relevance_scores)
        return max(
            0.0, base_score + 0.35 * relevance_bonus - repeat_penalty - length_penalty
        )

    def _rank_path(self, path, query_text, seed_entities, decomposition):
        score = self._score_path(
            path, query_text=query_text, seed_entities=seed_entities
        )
        query_tokens = self._tokenize(query_text)
        path_tokens = (
            set().union(
                *(
                    self._tokenize(edge.subject)
                    | self._tokenize(edge.relation)
                    | self._tokenize(edge.object)
                    for edge in path
                )
            )
            if path
            else set()
        )
        token_overlap = len(path_tokens & query_tokens) / max(1, len(query_tokens))
        trust = sum(float(edge.confidence) for edge in path) / max(1, len(path))
        path_length = 1.0 / max(1, len(path))
        rel_match = relation_match_score(path, decomposition or [])
        features = PathFeatures(
            token_overlap=token_overlap,
            trust=trust,
            path_length=path_length,
            relation_match=rel_match,
        )
        rank_score = self.path_ranker.rank_score(features)
        return 0.55 * score + 0.45 * rank_score

    def _select_evidence(self, scored_paths, max_results):
        cap = min(max_results, 12)
        min_results = min(8, cap)
        best_by_key = {}
        for path, score in scored_paths:
            for edge in path:
                key = edge.key()
                if key not in best_by_key or score > best_by_key[key][0]:
                    best_by_key[key] = (score, edge)
        ranked = sorted(best_by_key.values(), key=lambda item: item[0], reverse=True)
        selected = []
        covered_entities = set()
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

    def _build_evidence_graph(
        self, scored_paths: list[tuple[list[Statement], float]]
    ) -> EvidenceGraph:
        top_paths = scored_paths[: min(10, len(scored_paths))]
        node_conf: dict[str, list[float]] = {}
        rendered_paths = []
        for path, score in top_paths:
            rendered = [self._statement_payload(edge) for edge in path]
            rendered_paths.append({"score": score, "assertions": rendered})
            for edge in path:
                node_conf.setdefault(edge.subject, []).append(float(edge.confidence))
                node_conf.setdefault(edge.object, []).append(float(edge.confidence))
        nodes = [
            {"id": node, "confidence": sum(vals) / len(vals)}
            for node, vals in sorted(node_conf.items())
        ]
        confidence = sum(score for _, score in top_paths) / max(1, len(top_paths))
        return EvidenceGraph(paths=rendered_paths, nodes=nodes, confidence=confidence)
