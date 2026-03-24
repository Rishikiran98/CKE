"""Conflict detection and resolution for assertions."""

from __future__ import annotations

from itertools import combinations

from cke.graph.assertion import Assertion


class ConflictEngine:
    """Detect conflicting assertions and resolve by trust."""

    @staticmethod
    def _temporal_ranges_overlap(a: dict, b: dict) -> bool:
        """Check if two temporal range dicts have overlapping time spans."""
        a_start = a.get("start")
        a_end = a.get("end")
        b_start = b.get("start")
        b_end = b.get("end")
        if a_end and b_start and str(a_end) <= str(b_start):
            return False
        if b_end and a_start and str(b_end) <= str(a_start):
            return False
        return True

    @staticmethod
    def _qualifier_values_overlap(key: str, val_a: object, val_b: object) -> bool:
        """Check if two qualifier values for the same key overlap."""
        if key == "temporal" and isinstance(val_a, dict) and isinstance(val_b, dict):
            return ConflictEngine._temporal_ranges_overlap(val_a, val_b)
        return val_a == val_b

    @staticmethod
    def _qualifiers_overlap(a: dict, b: dict) -> bool:
        if not a or not b:
            return True
        shared = set(a).intersection(b)
        if not shared:
            return True
        return any(
            ConflictEngine._qualifier_values_overlap(key, a[key], b[key])
            for key in shared
        )

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
        loser.qualifiers = {**loser.qualifiers, "contradiction_flag": True}
        winner.qualifiers = {**winner.qualifiers, "contradiction_flag": False}
        return winner, loser
