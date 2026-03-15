"""Path-based symbolic reasoning with rule inference and confidence scoring."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Iterable, List

from cke.models import Statement
from cke.reasoning.operators import equality
from cke.reasoning.reasoning_trace import ReasoningTrace, ReasoningTraceLogger


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
        self._trace_logger = ReasoningTraceLogger()

    def answer(self, query: str, context: Iterable[Statement]) -> str:
        evidence_graph = list(context)
        if not evidence_graph:
            self._last_trace = ["No evidence graph edges were provided."]
            self._emit_trace(query, [], [], [], 0.0, "", [])
            return "I don't have enough graph context to answer that yet."

        inferred, rule_traces = self._apply_rules(evidence_graph)
        full_graph = evidence_graph + inferred

        subject = self._resolve_subject(query, full_graph)
        target_relation = self._resolve_target_relation(query)

        if subject is None:
            self._last_trace = ["Unable to identify a subject entity from query."]
            self._emit_trace(query, [], full_graph, [], 0.0, "", [])
            return "I don't have enough graph context to answer that yet."

        best_path = self._best_path(subject, target_relation, full_graph)
        if not best_path:
            self._last_trace = [
                f"No reasoning path found for subject '{subject}'",
                *rule_traces,
            ]
            self._emit_trace(query, [subject], full_graph, [], 0.0, "", [])
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

        operators_used: list[str] = []
        final_answer = best_path[-1].object
        maybe_comparison = self._comparison_answer(query, full_graph)
        if maybe_comparison is not None:
            final_answer = maybe_comparison
            operators_used.append("equality")
            trace.append(
                "Deterministic operator equality applied for comparison query."
            )

        self._last_trace = trace
        self._emit_trace(
            query=query,
            entities=[subject],
            retrieved_facts=full_graph,
            graph_paths=[best_path],
            confidence_score=confidence,
            final_answer=final_answer,
            operators_used=operators_used,
        )
        return final_answer

    def format_reasoning_path(self, context: List[Statement] | None = None) -> str:
        if self._last_trace:
            return "\n".join(self._last_trace)
        if not context:
            return "No path found."
        return "\n".join(
            f"{st.subject} -> {st.relation} -> {st.object}" for st in context
        )

    def _emit_trace(
        self,
        query: str,
        entities: list[str],
        retrieved_facts: list[Statement],
        graph_paths: list[list[Statement]],
        confidence_score: float,
        final_answer: str,
        operators_used: list[str],
    ) -> None:
        trace = ReasoningTrace(
            query=query,
            entities=entities,
            retrieved_facts=[
                {
                    "subject": st.subject,
                    "relation": st.relation,
                    "object": st.object,
                    "confidence": st.confidence,
                }
                for st in retrieved_facts
            ],
            graph_paths=[
                [
                    {
                        "subject": edge.subject,
                        "relation": edge.relation,
                        "object": edge.object,
                        "confidence": edge.confidence,
                    }
                    for edge in path
                ]
                for path in graph_paths
            ],
            operators_used=operators_used,
            confidence_score=confidence_score,
            final_answer=final_answer,
        )
        self._trace_logger.write(trace, stage="path_reasoner")

    def _comparison_answer(self, query: str, graph: list[Statement]) -> str | None:
        lowered = query.lower()
        if "same nationality" not in lowered:
            return None
        entities = sorted({st.subject for st in graph if st.relation == "nationality"})
        if len(entities) < 2:
            return None
        left_nat = next(
            (
                st.object
                for st in graph
                if st.subject == entities[0] and st.relation == "nationality"
            ),
            None,
        )
        right_nat = next(
            (
                st.object
                for st in graph
                if st.subject == entities[1] and st.relation == "nationality"
            ),
            None,
        )
        if left_nat is None or right_nat is None:
            return None
        return "yes" if equality(left_nat, right_nat) else "no"

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

    def _edge_rank(self, edge: Statement, target_relation: str | None) -> float:
        relation_bonus = (
            1.0 if target_relation and edge.relation == target_relation else 0.6
        )
        return edge.confidence * relation_bonus

    def _best_path(
        self,
        subject: str,
        target_relation: str | None,
        graph: List[Statement],
        min_confidence: float = 0.85,
        info_gain_epsilon: float = 0.05,
    ) -> List[Statement]:
        frontier: list[tuple[str, List[Statement], float]] = [(subject, [], 1.0)]
        best: tuple[List[Statement], int, float] | None = None

        while frontier:
            node, path, score = frontier.pop(0)
            if len(path) > 8:
                continue
            outgoing = [edge for edge in graph if edge.subject == node]
            ranked_outgoing = sorted(
                outgoing,
                key=lambda edge: self._edge_rank(edge, target_relation),
                reverse=True,
            )[:3]

            for edge in ranked_outgoing:
                new_path = path + [edge]
                new_score = score * edge.confidence
                info_gain = abs(new_score - score)

                relation_match = (
                    target_relation is None or edge.relation == target_relation
                )
                if relation_match:
                    effective_score = new_score + (
                        0.25 if edge.context.get("inferred") else 0.0
                    )
                    candidate = (new_path, len(new_path), effective_score)
                    if (
                        best is None
                        or candidate[1] > best[1]
                        or (candidate[1] == best[1] and candidate[2] > best[2])
                    ):
                        best = candidate
                    if new_score >= min_confidence:
                        continue

                if info_gain < info_gain_epsilon:
                    continue

                frontier.append((edge.object, new_path, new_score))

        if best:
            return best[0]
        return []
