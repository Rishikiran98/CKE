"""Query router responsible for query planning and adaptive routing."""

from __future__ import annotations

import re
from typing import Iterable

from cke.graph.domain_classifier import DomainClassifier
from cke.graph.domain_registry import DomainRegistry
from cke.graph_engine.graph_engine import KnowledgeGraphEngine
from cke.router.entity_linker import EntityLinker
from cke.router.intent_classifier import IntentClassifier
from cke.router.query_decomposer import QueryDecomposer
from cke.router.query_plan import QueryPlan
from cke.trust.confidence_model import ConfidenceModel, ReasoningConfidenceSignals


class QueryRouter:
    def __init__(
        self,
        graph_engine: KnowledgeGraphEngine | None = None,
        domain_classifier: DomainClassifier | None = None,
        domain_registry: DomainRegistry | None = None,
    ):
        self.graph_engine = graph_engine
        self.entity_linker = EntityLinker(graph_engine)
        self.intent_classifier = IntentClassifier()
        self.domain_classifier = domain_classifier or DomainClassifier()
        self.domain_registry = domain_registry
        self.query_decomposer = QueryDecomposer()
        self.confidence_model = ConfidenceModel()

    def detect_entities(
        self,
        query: str,
        candidate_entities: Iterable[str] | None = None,
    ):
        linked = self.entity_linker.extract_entities(query)
        if linked:
            return linked

        matched_entities: list[str] = []
        if candidate_entities:
            q = query.lower()
            matched_entities = [e for e in candidate_entities if e.lower() in q]
            if matched_entities:
                return sorted(set(matched_entities))

        name_chunks = re.findall(
            r"\b(?:[A-Z][a-z0-9'/-]+(?:\s+[A-Z][a-z0-9'/-]+)+)\b",
            query,
        )
        if name_chunks:
            return sorted(set(name_chunks))

        return sorted(set(re.findall(r"\b[A-Z][a-zA-Z0-9_/-]*\b", query)))

    def classify_domain(self, query: str):
        q = query.lower()
        domains = []

        if any(k in q for k in ["redis", "database", "sql", "nosql", "pubsub"]):
            domains.append("databases")
        if any(k in q for k in ["api", "service", "endpoint", "http"]):
            domains.append("systems")
        if any(k in q for k in ["model", "llm", "embedding", "inference"]):
            domains.append("ml")

        return domains or ["general"]

    def route(self, query: str, max_depth=None, max_results=12):
        graph_entities = self.graph_engine.all_entities() if self.graph_engine else None
        entities = self.detect_entities(query, graph_entities)
        intent = self.intent_classifier.classify(query)
        domains = self.classify_domain(query)

        max_depth = self._adaptive_depth(
            query=query,
            intent=intent,
            entities=entities,
            domains=domains,
            requested_depth=max_depth,
        )

        decomposition = self.query_decomposer.decompose(query, entities)
        operator_hint = decomposition.operator_hint or self._detect_operator_hint(query)
        confidence_score = self._estimate_confidence(
            intent, entities, decomposition.steps
        )
        reasoning_route, route_confidence = self._select_route(
            query=query,
            intent=intent,
            entities=entities,
            operator_hint=operator_hint,
            decomposition=decomposition,
            base_confidence=confidence_score,
        )

        return QueryPlan(
            query_text=query,
            seed_entities=entities,
            decomposition=[
                {
                    "type": step.step_type,
                    "value": step.value,
                    "confidence": step.confidence,
                }
                for step in decomposition.steps
            ],
            domains=domains,
            intent=intent,
            max_depth=max_depth,
            max_results=max_results,
            confidence_score=confidence_score,
            route_confidence=route_confidence,
            reasoning_route=reasoning_route,
            operator_hint=operator_hint,
            target_relations=list(decomposition.target_relations),
            multi_hop_hint=decomposition.multi_hop_hint,
            bridge_entities_expected=decomposition.bridge_entities_expected,
        )

    def _detect_operator_hint(self, query: str) -> str | None:
        lowered = f" {query.lower()} "
        if " how many" in lowered:
            return "count"
        if any(token in lowered for token in [" same ", " equal ", " identical "]):
            return "equality"
        if any(
            token in lowered
            for token in [
                " older ",
                " younger ",
                " before ",
                " after ",
                " earlier ",
                " later ",
                " latest ",
                " first ",
            ]
        ):
            return "temporal_compare"
        if any(
            token in lowered
            for token in [
                " greater ",
                " less ",
                " more than ",
                " fewer ",
                " higher ",
                " lower ",
            ]
        ):
            return "numeric_compare"
        if any(
            token in lowered for token in [" member of ", " part of ", " belong to "]
        ):
            return "containment"
        if lowered.strip().startswith(("did ", "is ", "has ", "was ")):
            return "existence"
        return None

    def _estimate_confidence(
        self, intent: str, entities: list[str], steps: list[object]
    ) -> float:
        relation_steps = [
            step for step in steps if getattr(step, "step_type", "") == "relation"
        ]
        signals = ReasoningConfidenceSignals(
            fact_completeness=min(1.0, len(entities) / 2),
            graph_coherence=0.9 if intent in {"factoid", "comparison"} else 0.7,
            operator_validity=0.85 if relation_steps else 0.6,
            retrieval_relevance=min(1.0, len(steps) / 4),
        )
        return self.confidence_model.reasoning_confidence(signals)

    @staticmethod
    def _reasoning_route(confidence_score: float) -> str:
        if confidence_score > 0.8:
            return "answer_immediately"
        if confidence_score > 0.6:
            return "graph_traversal"
        return "advanced_reasoner"

    def _select_route(
        self,
        query: str,
        intent: str,
        entities: list[str],
        operator_hint: str | None,
        decomposition,
        base_confidence: float,
    ) -> tuple[str, float]:
        relation_count = len(getattr(decomposition, "target_relations", []))
        entity_count = len(entities)
        route = self._reasoning_route(base_confidence)
        route_confidence = base_confidence
        normalized_query = query.lower()

        if (
            operator_hint
            and entity_count >= 1
            and (relation_count >= 1 or "how many" in normalized_query)
        ):
            return "answer_immediately", max(route_confidence, 0.85)

        if getattr(decomposition, "multi_hop_hint", False) and entity_count >= 2:
            return "advanced_reasoner", max(route_confidence, 0.8)

        if entity_count == 1 and relation_count == 1:
            return "graph_traversal", max(route_confidence, 0.78)

        if intent == "comparison" and entity_count >= 2:
            return "advanced_reasoner", max(route_confidence, 0.72)

        if relation_count == 0 and entity_count == 0:
            return route, min(route_confidence, 0.35)

        return route, route_confidence

    def _adaptive_depth(
        self,
        query: str,
        intent: str,
        entities: list[str],
        domains: list[str],
        requested_depth: int | None,
    ) -> int:
        base_depth = 3 if intent == "multi-hop" else 2
        if intent == "comparison":
            base_depth = 3

        if len(entities) >= 2:
            base_depth = max(base_depth, 3)

        lower_query = query.lower()
        bridge_markers = [
            " connected ",
            "connection",
            "path",
            "chain",
            "through",
            "via",
            "lead to",
            "result in",
            "because",
        ]
        bridge_hits = sum(
            1 for marker in bridge_markers if marker in f" {lower_query} "
        )

        clause_markers = [" and ", " then ", " after ", " before ", " while "]
        clause_hits = sum(
            1 for marker in clause_markers if marker in f" {lower_query} "
        )

        token_count = len(re.findall(r"[a-z0-9]+", lower_query))
        complexity_bonus = 0
        if bridge_hits >= 2:
            complexity_bonus += 1
        if clause_hits >= 2:
            complexity_bonus += 1
        if token_count > 18:
            complexity_bonus += 1
        if len(domains) > 1:
            complexity_bonus += 1

        adaptive_depth = base_depth + complexity_bonus

        if requested_depth is None:
            return max(2, min(8, adaptive_depth))

        if complexity_bonus == 0:
            return max(2, min(8, min(requested_depth, adaptive_depth + 1)))
        return max(2, min(8, max(requested_depth, adaptive_depth)))

    def routing_policy_for_query(self, query: str):
        domain = self.domain_classifier.classify_entity(query)
        state = (
            self.domain_registry.get_domain_state(domain)
            if self.domain_registry
            else "evolving"
        )

        policy = {
            "domain": domain,
            "state": state,
            "prefer_cached": False,
            "retrieval_depth_delta": 0,
            "prioritize_recent": False,
        }

        if state == "stable":
            policy["prefer_cached"] = True
        elif state == "evolving":
            policy["retrieval_depth_delta"] = 1
        elif state == "volatile":
            policy["prioritize_recent"] = True
            policy["retrieval_depth_delta"] = 1

        return policy
