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
