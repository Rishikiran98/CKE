"""Query router that creates graph retrieval plans."""

from cke.graph_engine.graph_engine import KnowledgeGraphEngine
from cke.router.entity_linker import EntityLinker
from cke.router.intent_classifier import IntentClassifier
from cke.router.query_plan import QueryPlan


class QueryRouter:
    """Build query plans by entity linking, intent detection, and domain rules."""

    def __init__(self, graph_engine: KnowledgeGraphEngine | None = None) -> None:
        self.graph_engine = graph_engine
        self.entity_linker = EntityLinker(graph_engine)
        self.intent_classifier = IntentClassifier()

    def detect_entities(self, query: str, candidate_entities: list[str]) -> list[str]:
        """Backward-compatible entity detection API."""
        linked = self.entity_linker.extract_entities(query)
        if linked:
            return linked
        query_lower = query.lower()
        return sorted(
            {entity for entity in candidate_entities if entity.lower() in query_lower}
        )

    def classify_domain(self, query: str) -> list[str]:
        q = query.lower()
        domains: list[str] = []
        if any(k in q for k in ["redis", "database", "sql", "nosql", "pubsub"]):
            domains.append("databases")
        if any(k in q for k in ["api", "service", "endpoint", "http"]):
            domains.append("systems")
        if any(k in q for k in ["model", "llm", "embedding", "inference"]):
            domains.append("ml")
        return domains or ["general"]

    def route(
        self, query: str, max_depth: int | None = None, max_results: int = 12
    ) -> QueryPlan:
        entities = self.entity_linker.extract_entities(query)
        intent = self.intent_classifier.classify(query)
        domains = self.classify_domain(query)

        if max_depth is None:
            max_depth = 3 if intent == "multi-hop" else 2

        return QueryPlan(
            seed_entities=entities,
            domains=domains,
            intent=intent,
            max_depth=max_depth,
            max_results=max_results,
        )
"""Query router with adaptive domain-state-aware routing policies."""

from __future__ import annotations

import re
from typing import Iterable

from cke.graph.domain_classifier import DomainClassifier
from cke.graph.domain_registry import DomainRegistry


class QueryRouter:
    """Detect entities and suggest retrieval strategy from domain state."""

    def __init__(
        self,
        domain_classifier: DomainClassifier | None = None,
        domain_registry: DomainRegistry | None = None,
    ) -> None:
        self.domain_classifier = domain_classifier or DomainClassifier()
        self.domain_registry = domain_registry

    def detect_entities(
        self, query: str, candidate_entities: Iterable[str]
    ) -> list[str]:
        query_lower = query.lower()
        matches = [
            entity for entity in candidate_entities if entity.lower() in query_lower
        ]
        if matches:
            return sorted(set(matches))

        token_candidates = re.findall(r"\b[A-Z][a-zA-Z0-9_/-]*\b", query)
        return sorted(set(token_candidates))

    def routing_policy_for_query(self, query: str) -> dict[str, str | int | bool]:
        domain = self.domain_classifier.classify_entity(query)
        state = (
            self.domain_registry.get_domain_state(domain)
            if self.domain_registry is not None
            else "evolving"
        ) or "evolving"

        policy: dict[str, str | int | bool] = {
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
