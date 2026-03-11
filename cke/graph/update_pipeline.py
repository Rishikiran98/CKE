"""Graph update pipeline with deduplication, trust scoring, and conflicts."""

from __future__ import annotations

from collections import defaultdict
from typing import Any

from cke.entity_resolution.entity_resolver import EntityResolver
from cke.graph.assertion import Assertion
from cke.graph.conflict_engine import ConflictEngine
from cke.graph.deduplicator import AssertionDeduplicator
from cke.graph.trust_engine import TrustEngine
from cke.graph_engine.graph_engine import KnowledgeGraphEngine


class GraphUpdatePipeline:
    """Apply assertion updates into the graph store."""

    def __init__(
        self,
        graph: KnowledgeGraphEngine,
        resolver: EntityResolver | None = None,
        trust_engine: TrustEngine | None = None,
        conflict_engine: ConflictEngine | None = None,
        deduplicator: AssertionDeduplicator | None = None,
    ) -> None:
        self.graph = graph
        self.resolver = resolver or EntityResolver()
        self.trust_engine = trust_engine or TrustEngine()
        self.conflict_engine = conflict_engine or ConflictEngine()
        self.deduplicator = deduplicator or AssertionDeduplicator(self.trust_engine)
        self.conflict_metadata: dict[str, list[dict[str, Any]]] = defaultdict(list)

    def _resolve_entity(self, name: str) -> str:
        return self.resolver.resolve_entity(name)

    def update_graph(self, assertions: list[Assertion]) -> list[Assertion]:
        """Update graph by resolving entities, deduplicating, and storing trust/conflicts.
        """
        resolved: list[Assertion] = []
        for assertion in assertions:
            assertion.subject = self._resolve_entity(assertion.subject)
            assertion.object = self._resolve_entity(assertion.object)
            self.trust_engine.compute_trust(assertion)
            resolved.append(assertion)

        merged = self.deduplicator.deduplicate(resolved)

        conflicts = self.conflict_engine.detect_conflicts(merged)
        secondary_ids: set[int] = set()
        for left, right in conflicts:
            winner, loser = self.conflict_engine.resolve_conflict(left, right)
            secondary_ids.add(id(loser))
            conflict_key = f"{winner.subject}::{winner.relation}"
            self.conflict_metadata[conflict_key].append(
                {
                    "winner": winner.object,
                    "loser": loser.object,
                    "winner_trust": winner.trust_score,
                    "loser_trust": loser.trust_score,
                }
            )

        for assertion in merged:
            context = {
                "qualifiers": assertion.qualifiers,
                "evidence_count": assertion.evidence_count,
                "trust_score": assertion.trust_score,
                "secondary": id(assertion) in secondary_ids,
                "evidence": [e.__dict__ for e in assertion.evidence],
            }
            self.graph.add_statement(
                assertion.subject,
                assertion.relation,
                assertion.object,
                context=context,
                confidence=assertion.trust_score,
                source=assertion.source,
                timestamp=str(assertion.timestamp),
            )

        return merged
