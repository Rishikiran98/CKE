"""Query router that maps questions to graph entities."""

from __future__ import annotations

import re
from typing import Iterable, List


class QueryRouter:
    """Detect entities in natural language queries."""

    def detect_entities(self, query: str, candidate_entities: Iterable[str]) -> List[str]:
        query_lower = query.lower()
        matches = [entity for entity in candidate_entities if entity.lower() in query_lower]
        if matches:
            return sorted(set(matches))

        # fallback: extract title-like phrases as potential entities
        token_candidates = re.findall(r"\b[A-Z][a-zA-Z0-9_/-]*\b", query)
        return sorted(set(token_candidates))
