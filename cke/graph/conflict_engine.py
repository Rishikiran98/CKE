"""Conflict detection and resolution for assertions."""

from __future__ import annotations

from itertools import combinations

from cke.graph.assertion import Assertion


class ConflictEngine:
    """Detect conflicting assertions and resolve by trust."""

    @staticmethod
    def _qualifiers_overlap(a: dict, b: dict) -> bool:
        if not a or not b:
            return True
        shared = set(a).intersection(b)
        if not shared:
            return True
        return any(a[key] == b[key] for key in shared)

    def assertions_conflict(
        self, assertion_a: Assertion, assertion_b: Assertion
    ) -> bool:
        return (
            assertion_a.subject == assertion_b.subject
            and assertion_a.relation == assertion_b.relation
            and assertion_a.object != assertion_b.object
            and self._qualifiers_overlap(assertion_a.qualifiers, assertion_b.qualifiers)
        )

    def detect_conflicts(
        self, assertions: list[Assertion]
    ) -> list[tuple[Assertion, Assertion]]:
        conflicts: list[tuple[Assertion, Assertion]] = []
        for left, right in combinations(assertions, 2):
            if self.assertions_conflict(left, right):
                conflicts.append((left, right))
        return conflicts

    def resolve_conflict(
        self, assertion_a: Assertion, assertion_b: Assertion
    ) -> tuple[Assertion, Assertion]:
        """Return (winner, loser); loser is retained as secondary."""
        if assertion_a.trust_score >= assertion_b.trust_score:
            winner, loser = assertion_a, assertion_b
        else:
            winner, loser = assertion_b, assertion_a
        loser.secondary = True
        return winner, loser
