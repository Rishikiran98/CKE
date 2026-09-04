"""Graph retrieval engine with BFS, beam, and A* traversal plus path ranking."""

from __future__ import annotations

from collections import deque
import re
from heapq import heappop, heappush, nlargest
from itertools import count

from cke.diagnostics import DegradationMixin, require_strict_component
from cke.entity_resolution.entity_resolver import EntityResolver
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


#: Below this many selected statements, the diversity filter is not applied:
#: a small evidence set should not be thinned further for redundancy.
_DIVERSITY_FLOOR = 8


class GraphRetriever(DegradationMixin):
    """Retrieve sparse evidence paths from the graph for a query plan."""

    def __init__(
        self,
        graph_engine: KnowledgeGraphEngine,
        entity_resolver: EntityResolver | None = None,
        monitor: SystemMonitor | None = None,
        path_ranker: PathRankingModel | None = None,
        strict: bool = False,
    ) -> None:
        self._init_degradation(strict)
        self.graph_engine = graph_engine
        require_strict_component(
            type(self).__name__, entity_resolver, "entity resolver", self.strict
        )
        self.entity_resolver = entity_resolver or EntityResolver(strict=strict)
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
        # Fan-out, not one canonical per mention: a query mentioning "Kansas
        # City" against a graph holding "Kansas City jazz" and "Kansas City
        # Chiefs" should start from both and let path scoring choose. resolve()
        # has to pick one and so refuses that mention outright.
        #
        # Kept grouped by mention as well as flattened, because comparison
        # bridges between two different mentions and cannot read them off a
        # flat list: expanding the first mention to two entities would leave
        # the bridge comparing that mention against itself.
        seed_groups = self.entity_resolver.expand_groups(query_plan.seed_entities)
        seeds = [seed for group in seed_groups for seed in group]
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
                seed_groups=seed_groups,
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

    @staticmethod
    def _other_endpoint(edge: Statement, node: str) -> str:
        """Return the endpoint of *edge* that is not *node*.

        Traversal follows an edge from either end, so the next node is whichever
        endpoint we did not arrive from. The edge's own direction is untouched.
        """
        if edge.subject.strip().lower() == str(node).strip().lower():
            return edge.object
        return edge.subject

    def _forward_edges(self, node: str, path: list[Statement]) -> list[Statement]:
        """Incident edges at *node*, excluding the one just traversed.

        Traversal follows edges in both directions, so at every non-root node
        the edge we arrived on is also a candidate leading straight back. A
        narrow beam would spend its whole width re-selecting it and never
        expand a lower-scoring continuation.
        """
        incident = self.graph_engine.get_incident(node)
        if not path:
            return incident
        arrived_on = path[-1].key()
        return [edge for edge in incident if edge.key() != arrived_on]

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
        # An edge between two seeds is incident to both, so it arrives twice.
        # Truncating before deduplication let one edge consume several budget
        # slots and push out unique lower-ranked ones.
        seen_edges: set = set()
        for seed in seeds:
            neighbors = self.graph_engine.get_incident(seed)
            expanded += len(neighbors)
            for edge in neighbors:
                key = edge.key()
                if key in seen_edges:
                    continue
                seen_edges.add(key)
                scored.append(
                    (
                        [edge],
                        self._rank_path([edge], query_text, seeds_raw, decomposition),
                    )
                )
        if self.monitor:
            self.monitor.record_neighborhood_expansion(expanded)
        scored.sort(key=lambda item: item[1], reverse=True)
        return scored[: max(1, max_results)]

    def _bridge_mode(
        self, seed_groups, max_depth, query_text, seed_entities, decomposition
    ):
        """Find paths joining the first mention to the second.

        Takes candidates grouped by mention, not a flat seed list: a mention
        expanding to several entities must not be compared against itself.
        Every entity of the first mention is bridged against every entity of
        the second, so fan-out widens the search rather than displacing the
        thing being compared to.
        """
        if len(seed_groups) < 2:
            return []

        right_by_node: dict[str, list[list[Statement]]] = {}
        for right_seed in seed_groups[1]:
            for path, reached in self._paths_from_seed(right_seed, max_depth=max_depth):
                right_by_node.setdefault(reached, []).append(path)

        candidates = []
        bridge_nodes_found = set()
        for left_seed in seed_groups[0]:
            for left_path, bridge_node in self._paths_from_seed(
                left_seed, max_depth=max_depth
            ):
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

    def _paths_from_seed(
        self, seed: str, max_depth: int
    ) -> list[tuple[list[Statement], str]]:
        """Walk out from *seed*, returning each path with the node it reached.

        The reached node is returned explicitly because a path may end by
        following an edge backwards: for ``X -> A`` walked from ``A``, the walk
        reaches ``X`` while ``path[-1].object`` is still ``A``. Callers that
        need the endpoint, such as bridge matching, must use this rather than
        reading it off the last edge.
        """
        queue = deque([(seed, [], 0)])
        paths: list[tuple[list[Statement], str]] = []
        visited = set()
        while queue:
            node, path, depth = queue.popleft()
            if depth >= max_depth:
                continue
            marker = (node, depth)
            if marker in visited:
                continue
            visited.add(marker)
            for edge in self._forward_edges(node, path):
                next_node = self._other_endpoint(edge, node)
                next_path = path + [edge]
                paths.append((next_path, next_node))
                queue.append((next_node, next_path, depth + 1))
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
            # Which document this came from, when the caller recorded one.
            # Without it a retrieved statement cannot be checked against the
            # dataset's supporting facts, so the graph arm's recall could not
            # be measured at all while the dense arm's could.
            "source": st.source,
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
            for edge in self._forward_edges(node, path):
                next_path = path + [edge]
                paths.append(next_path)
                next_node = self._other_endpoint(edge, node)
                if depth + 1 <= max_depth and next_node != node:
                    queue.append((next_node, next_path, depth + 1))
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
                for edge in self._forward_edges(node, path):
                    candidate = path + [edge]
                    expanded.append(
                        (
                            self._other_endpoint(edge, node),
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
            for edge in self._forward_edges(node, path):
                candidate = path + [edge]
                next_node = self._other_endpoint(edge, node)
                g = self._score_path(candidate, query_text, seed_entities)
                h = self._heuristic(next_node, query_tokens, seed_tokens)
                f = g + h
                heappush(frontier, (-f, next(tick), next_node, candidate, depth + 1))
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
        # max_results used to be clamped to 12 here and in _neighborhood_mode,
        # so any retrieval budget above 12 silently produced the same evidence
        # set. The budget the caller asked for is now the budget applied.
        cap = max(1, int(max_results))
        min_results = min(_DIVERSITY_FLOOR, cap)
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
