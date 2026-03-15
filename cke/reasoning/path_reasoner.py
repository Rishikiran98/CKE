"""Path-based symbolic reasoning with ranked traversal and verification."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Iterable, List

import numpy as np

from cke.models import Statement
from cke.reasoning.operators import equality
from cke.reasoning.pattern_memory import PatternMemory
from cke.reasoning.reasoner import TemplateReasoner
from cke.reasoning.reasoning_trace import ReasoningTrace, ReasoningTraceLogger
from cke.reasoning.verifier import ReasoningVerifier
from cke.retrieval.embedding_model import EmbeddingModel


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

    def __init__(
        self,
        rules: List[InferenceRule] | None = None,
        verifier: ReasoningVerifier | None = None,
        pattern_memory: PatternMemory | None = None,
        embedding_model: EmbeddingModel | None = None,
    ) -> None:
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
        self._verifier = verifier or ReasoningVerifier()
        self._pattern_memory = pattern_memory or PatternMemory()
        self._embedding_model = embedding_model or EmbeddingModel()
        self._advanced_reasoner = TemplateReasoner()

    def answer(self, query: str, context: Iterable[Statement]) -> str:
        evidence_graph = list(context)
        if not evidence_graph:
            self._last_trace = ["No evidence graph edges were provided."]
            self._emit_trace(query, [], [], [], 0.0, "", [])
            return "I don't have enough graph context to answer that yet."

        inferred, rule_traces = self._apply_rules(evidence_graph)
        full_graph = evidence_graph + inferred

        template_run = self._pattern_memory.execute(query, full_graph)
        if template_run is not None:
            operator_checks = list(template_run.operator_checks)
            verification = self._verifier.verify(
                query=query,
                context=full_graph,
                reasoning_path=template_run.path,
                answer=template_run.answer,
                confidence_score=template_run.confidence_score,
                required_facts=template_run.required_facts,
                operator_checks=operator_checks,
            )
            if verification.passed:
                self._last_trace = [
                    *template_run.trace,
                    *rule_traces,
                    "Template reasoning verification passed.",
                ]
                self._emit_trace(
                    query=query,
                    entities=list(template_run.entities),
                    retrieved_facts=full_graph,
                    graph_paths=[list(template_run.path)],
                    confidence_score=template_run.confidence_score,
                    final_answer=template_run.answer,
                    operators_used=list(template_run.operators_used),
                )
                return template_run.answer
            return self._fallback_to_advanced_reasoner(
                query=query,
                full_graph=full_graph,
                reason=f"Template verification failed: {verification.summary}",
                rule_traces=rule_traces,
            )

        subject = self._resolve_subject(query, full_graph)
        target_relation = self._resolve_target_relation(query)

        if subject is None:
            self._last_trace = ["Unable to identify a subject entity from query."]
            self._emit_trace(query, [], full_graph, [], 0.0, "", [])
            return "I don't have enough graph context to answer that yet."

        best_path, traversal_trace = self._best_path(subject, target_relation, query, full_graph)
        if not best_path:
            self._last_trace = [
                f"No reasoning path found for subject '{subject}'",
                *rule_traces,
                *traversal_trace,
            ]
            self._emit_trace(query, [subject], full_graph, [], 0.0, "", [])
            return "Insufficient graph evidence."

        confidence = 1.0
        trace = [*traversal_trace]
        for edge in best_path:
            confidence *= edge.confidence
            trace.append(
                f"{edge.subject} -> {edge.relation} -> {edge.object} "
                f"(confidence={edge.confidence:.2f})"
            )
        trace.append(f"Path confidence = {confidence:.4f}")
        reasoning_confidence = self._reasoning_confidence(best_path, confidence)
        trace.append(f"Reasoning confidence = {reasoning_confidence:.4f}")
        trace.extend(rule_traces)

        operators_used: list[str] = []
        final_answer = best_path[-1].object
        operator_checks: list[dict[str, object]] = []
        maybe_comparison = self._comparison_answer(query, full_graph)
        if maybe_comparison is not None:
            final_answer = maybe_comparison.answer
            operators_used.append("equality")
            operator_checks.append(
                {
                    "operator": "equality",
                    "inputs": maybe_comparison.inputs,
                    "result": maybe_comparison.result,
                }
            )
            trace.append(
                "Deterministic operator equality applied for comparison query."
            )

        verification = self._verifier.verify(
            query=query,
            context=full_graph,
            reasoning_path=best_path,
            answer=final_answer,
            confidence_score=reasoning_confidence,
            required_facts=self._required_facts_for_query(query, full_graph),
            operator_checks=operator_checks,
        )
        if not verification.passed:
            return self._fallback_to_advanced_reasoner(
                query=query,
                full_graph=full_graph,
                reason=f"Reasoning verification failed: {verification.summary}",
                rule_traces=trace,
            )

        trace.append("Reasoning verification passed.")
        self._last_trace = trace
        self._emit_trace(
            query=query,
            entities=[subject],
            retrieved_facts=full_graph,
            graph_paths=[best_path],
            confidence_score=reasoning_confidence,
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
                    "trust_score": self._trust_score(st),
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
                        "trust_score": self._trust_score(edge),
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

    def _comparison_answer(self, query: str, graph: list[Statement]) -> _ComparisonResult | None:
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
        result = equality(left_nat, right_nat)
        return _ComparisonResult(
            answer="yes" if result else "no",
            inputs=(left_nat, right_nat),
            result=result,
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

    def _edge_rank(self, edge: Statement, query: str, query_embedding: np.ndarray) -> float:
        semantic_similarity = self._semantic_similarity(edge, query, query_embedding)
        return semantic_similarity * self._trust_score(edge)

    @staticmethod
    def _trust_score(edge: Statement) -> float:
        for key in ("trust_score", "resolved_trust", "confidence_score"):
            value = edge.context.get(key)
            if value is None:
                continue
            try:
                return max(0.0, min(1.0, float(value)))
            except (TypeError, ValueError):
                continue
        return max(0.0, min(1.0, float(edge.confidence)))

    def _semantic_similarity(
        self,
        edge: Statement,
        query: str,
        query_embedding: np.ndarray,
    ) -> float:
        edge_text = " ".join(
            [
                edge.subject,
                edge.relation,
                edge.object,
                str(edge.context.get("text", "")),
            ]
        ).strip()
        edge_embedding = self._embedding_model.embed_text(edge_text)
        query_norm = float(np.linalg.norm(query_embedding))
        edge_norm = float(np.linalg.norm(edge_embedding))
        if query_norm == 0.0 or edge_norm == 0.0:
            return 0.0
        similarity = float(np.dot(query_embedding, edge_embedding) / (query_norm * edge_norm))
        return max(0.0, similarity)

    def _best_path(
        self,
        subject: str,
        target_relation: str | None,
        query: str,
        graph: List[Statement],
        min_confidence: float = 0.85,
        info_gain_epsilon: float = 0.05,
        max_neighbors: int = 3,
    ) -> tuple[List[Statement], list[str]]:
        frontier: list[tuple[str, List[Statement], float]] = [(subject, [], 1.0)]
        best: tuple[List[Statement], int, float] | None = None
        traversal_trace: list[str] = []
        query_embedding = self._embedding_model.embed_text(query)

        while frontier:
            node, path, score = frontier.pop(0)
            if len(path) > 8:
                continue
            outgoing = [edge for edge in graph if edge.subject == node]
            ranked_outgoing = sorted(
                outgoing,
                key=lambda edge: self._edge_rank(edge, query, query_embedding),
                reverse=True,
            )[:max_neighbors]
            for rank, edge in enumerate(ranked_outgoing, start=1):
                rank_score = self._edge_rank(edge, query, query_embedding)
                traversal_trace.append(
                    "Ranked expansion "
                    f"{node} -> {edge.object} via {edge.relation} "
                    f"(rank={rank}, score={rank_score:.4f})"
                )
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
            return best[0], traversal_trace
        return [], traversal_trace


    @staticmethod
    def _reasoning_confidence(path: list[Statement], path_confidence: float) -> float:
        if not path:
            return 0.0
        depth = max(1, len(path))
        return max(0.0, min(1.0, path_confidence ** (1.0 / depth)))

    def _required_facts_for_query(
        self,
        query: str,
        graph: list[Statement],
    ) -> list[tuple[str, str]]:
        lowered = query.lower()
        if "same nationality" not in lowered:
            return []
        entities = sorted({st.subject for st in graph if st.relation == "nationality"})[:2]
        return [(entity, "nationality") for entity in entities]

    def _fallback_to_advanced_reasoner(
        self,
        query: str,
        full_graph: list[Statement],
        reason: str,
        rule_traces: list[str],
    ) -> str:
        answer = self._advanced_reasoner.answer(query, full_graph)
        self._last_trace = [*rule_traces, reason, "Fallback route -> advanced_reasoner"]
        self._emit_trace(
            query=query,
            entities=self._resolve_entities(full_graph),
            retrieved_facts=full_graph,
            graph_paths=[],
            confidence_score=0.0,
            final_answer=answer,
            operators_used=["advanced_reasoner_fallback"],
        )
        return answer

    @staticmethod
    def _resolve_entities(graph: list[Statement]) -> list[str]:
        entities: set[str] = set()
        for st in graph:
            entities.add(st.subject)
            entities.add(st.object)
        return sorted(entities)


@dataclass(slots=True)
class _ComparisonResult:
    answer: str
    inputs: tuple[str, str]
    result: bool
