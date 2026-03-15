"""Path-based symbolic reasoning with rule inference and confidence scoring."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Iterable, List

from cke.models import Statement


@dataclass(slots=True)
class InferenceRule:
    """A lightweight binary inference rule."""

    name: str
    left_relation: str
    right_relation: str
    inferred_relation: str

    def apply(
        self, statements: List[Statement]
    ) -> List[tuple[Statement, tuple[Statement, Statement]]]:
        inferred: List[tuple[Statement, tuple[Statement, Statement]]] = []
        for left in statements:
            if left.relation != self.left_relation:
                continue
            for right in statements:
                if right.relation != self.right_relation:
                    continue
                if left.object != right.subject:
                    continue

                score = max(0.0, min(1.0, left.confidence * right.confidence))
                candidate = Statement(
                    subject=left.subject,
                    relation=self.inferred_relation,
                    object=right.object,
                    confidence=score,
                    context={"inferred": True, "rule": self.name},
                )
                inferred.append((candidate, (left, right)))
        return inferred


class PathReasoner:
    """Reason over subject→relation→object chains with probabilistic scoring."""

    def __init__(self, rules: List[InferenceRule] | None = None) -> None:
        self.rules = rules or [
            InferenceRule(
                name="located_in_transitivity",
                left_relation="located_in",
                right_relation="located_in",
                inferred_relation="located_in",
            )
        ]
        self._last_trace: List[str] = []

    def answer(self, query: str, context: Iterable[Statement]) -> str:
        evidence_graph = list(context)
        if not evidence_graph:
            self._last_trace = ["No evidence graph edges were provided."]
            return "I don't have enough graph context to answer that yet."

        inferred, rule_traces = self._apply_rules(evidence_graph)
        full_graph = evidence_graph + inferred

        subject = self._resolve_subject(query, full_graph)
        target_relation = self._resolve_target_relation(query)

        if subject is None:
            self._last_trace = ["Unable to identify a subject entity from query."]
            return "I don't have enough graph context to answer that yet."

        best_path = self._best_path(subject, target_relation, full_graph)
        if not best_path:
            self._last_trace = [
                f"No reasoning path found for subject '{subject}'",
                *rule_traces,
            ]
            return "Insufficient graph evidence."

        confidence = 1.0
        trace = []
        for edge in best_path:
            confidence *= edge.confidence
            trace.append(
                f"{edge.subject} -> {edge.relation} -> {edge.object} "
                f"(confidence={edge.confidence:.2f})"
            )
        trace.append(f"Path confidence = {confidence:.4f}")
        trace.extend(rule_traces)
        self._last_trace = trace

        return best_path[-1].object

    def format_reasoning_path(self, context: List[Statement] | None = None) -> str:
        if self._last_trace:
            return "\n".join(self._last_trace)
        if not context:
            return "No path found."
        return "\n".join(
            f"{st.subject} -> {st.relation} -> {st.object}" for st in context
        )

    def _apply_rules(
        self, statements: List[Statement]
    ) -> tuple[List[Statement], List[str]]:
        inferred: List[Statement] = []
        traces: List[str] = []
        existing = {st.key() for st in statements}

        for rule in self.rules:
            for candidate, premises in rule.apply(statements):
                if candidate.key() in existing:
                    continue
                existing.add(candidate.key())
                inferred.append(candidate)
                left, right = premises
                traces.append(
                    "Rule applied "
                    f"{rule.name}: {left.as_text()} + "
                    f"{right.as_text()} -> {candidate.as_text()}"
                )

        return inferred, traces

    def _resolve_subject(self, query: str, graph: List[Statement]) -> str | None:
        lowered = query.lower()
        matches = [st.subject for st in graph if st.subject.lower() in lowered]
        if matches:
            return max(matches, key=len)
        return graph[0].subject if graph else None

    def _resolve_target_relation(self, query: str) -> str | None:
        lowered = query.lower()
        if "nationality" in lowered:
            return "nationality"
        if "located in" in lowered or "located_in" in lowered:
            return "located_in"
        match = re.search(r"\b(\w+)\b\??$", lowered)
        if match:
            return match.group(1)
        return None

    def _best_path(
        self,
        subject: str,
        target_relation: str | None,
        graph: List[Statement],
        max_depth: int = 4,
    ) -> List[Statement]:
        frontier: list[tuple[str, List[Statement], float]] = [(subject, [], 1.0)]
        best: tuple[List[Statement], int, float] | None = None

        while frontier:
            node, path, score = frontier.pop(0)
            if len(path) >= max_depth:
                continue

            outgoing = [edge for edge in graph if edge.subject == node]
            for edge in outgoing:
                new_path = path + [edge]
                new_score = score * edge.confidence

                relation_match = (
                    target_relation is None or edge.relation == target_relation
                )
                if relation_match:
                    depth = len(new_path)
                    if (
                        best is None
                        or depth > best[1]
                        or (depth == best[1] and new_score > best[2])
                    ):
                        best = (new_path, depth, new_score)

                if len(new_path) < max_depth:
                    frontier.append((edge.object, new_path, new_score))

        return best[0] if best else []
