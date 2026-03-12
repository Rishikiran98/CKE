"""Query router responsible for query planning and adaptive routing."""

from __future__ import annotations

import re
from typing import Iterable

from cke.graph_engine.graph_engine import KnowledgeGraphEngine
from cke.graph.domain_classifier import DomainClassifier
from cke.graph.domain_registry import DomainRegistry
from cke.router.entity_linker import EntityLinker
from cke.router.intent_classifier import IntentClassifier
from cke.router.query_plan import QueryPlan


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

    def detect_entities(
        self,
        query: str,
        candidate_entities: Iterable[str] | None = None,
    ):
        linked = self.entity_linker.extract_entities(query)
        if linked:
            return linked

        if candidate_entities:
            q = query.lower()
            matches = [e for e in candidate_entities if e.lower() in q]
            if matches:
                return sorted(set(matches))

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
