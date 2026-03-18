"""Central query orchestration skeleton for CKE pipeline."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING
from uuid import uuid4

from cke.entity_resolution.alias_registry import AliasRegistry
from cke.entity_resolution.entity_resolver import EntityResolver
from cke.pipeline.types import (
    QueryResult,
    ReasonerOutcome,
    ReasoningContext,
)
from cke.reasoning.reasoner_adapter import ReasonerAdapter
from cke.reasoning.operator_executor import OperatorExecutor
from cke.reasoning.operator_selector import OperatorSelector
from cke.reasoning.verifier import ReasoningVerifier
from cke.trust.confidence_calibrator import ConfidenceCalibrator

if TYPE_CHECKING:
    from cke.router.query_router import QueryRouter


logger = logging.getLogger(__name__)


class QueryOrchestrator:
    """Coordinates query routing and grounded reasoning steps."""

    def __init__(
        self,
        graph_engine,
        router: "QueryRouter",
        retriever=None,
        assembler=None,
        reasoner=None,
        verifier=None,
        operator_selector: OperatorSelector | None = None,
        operator_executor: OperatorExecutor | None = None,
        entity_resolver: EntityResolver | None = None,
    ):
        self.graph_engine = graph_engine
        self.router = router
        self.retriever = retriever
        self.assembler = assembler
        if reasoner is None:
            from cke.reasoning.path_reasoner import PathReasoner

            self.reasoner = PathReasoner()
        else:
            self.reasoner = reasoner
        self.verifier = verifier or ReasoningVerifier()
        self.operator_selector = operator_selector or OperatorSelector()
        self.operator_executor = operator_executor or OperatorExecutor()
        self.reasoner_adapter = ReasonerAdapter(self.reasoner)
        self.last_context: ReasoningContext | None = None
        self.entity_resolver = entity_resolver or EntityResolver()
        self.confidence_calibrator = ConfidenceCalibrator()

    def answer(self, query: str) -> QueryResult:
        query_plan = self.router.route(query)
        trace_id = str(uuid4())

        candidate_entities = self.router.detect_entities(query)
        for entity in candidate_entities:
            self.entity_resolver.register_alias(entity, entity)
        resolved_entities = self.entity_resolver.resolve_mentions(
            query,
            candidate_entities=candidate_entities,
        )
        target_relations = list(getattr(query_plan, "target_relations", []))

        if self.retriever is not None and self.assembler is not None:
            retrieved_chunks, facts = self.retriever.retrieve(
                query,
                resolved_entities=resolved_entities,
                target_relations=target_relations,
            )
            context = self.assembler.assemble(
                query=query,
                query_plan=query_plan,
                resolved_entities=resolved_entities,
                chunks=retrieved_chunks,
                facts=facts,
            )
        else:
            context = ReasoningContext(
                query=query,
                query_plan=query_plan,
                resolved_entities=resolved_entities,
                retrieved_chunks=[],
                evidence_facts=[],
                candidate_paths=[],
                subgraph=None,
                decomposition=list(getattr(query_plan, "decomposition", [])),
                trace_metadata={},
            )

        context.trace_metadata.setdefault("target_relations", target_relations)
        context.trace_metadata.setdefault(
            "resolved_entities",
            [
                {
                    "surface_form": e.surface_form,
                    "canonical_name": e.canonical_name,
                    "entity_id": e.entity_id,
                    "link_confidence": e.link_confidence,
                }
                for e in resolved_entities
            ],
        )

        logger.info(
            "Mentions detected: %s", [e.surface_form for e in resolved_entities]
        )
        logger.info(
            "Canonical resolution results: %s",
            [
                (e.surface_form, e.canonical_name, e.link_confidence)
                for e in resolved_entities
            ],
        )
        logger.info(
            "Alias matches used: %s", [e.aliases_matched for e in resolved_entities]
        )
        logger.info("Target relations inferred: %s", target_relations)
        logger.info("Resolved entities: %s", len(resolved_entities))
        logger.info("Retrieved chunks: %s", len(context.retrieved_chunks))
        logger.info(
            "Evidence facts before filtering: %s",
            context.trace_metadata.get(
                "evidence_facts_before_filtering", len(context.evidence_facts)
            ),
        )
        logger.info("Evidence facts after filtering: %s", len(context.evidence_facts))
        logger.info("Candidate paths: %s", len(context.candidate_paths))
        logger.info(
            "Path-aware reasoning attempt possible: %s", bool(context.candidate_paths)
        )

        self.last_context = context
        debug_info = self._build_debug_info(
            context=context,
            query_plan=query_plan,
            trace_id=trace_id,
        )
        confidence_signals = self._initial_confidence_signals(
            context=context,
            query_plan=query_plan,
            resolved_entities=resolved_entities,
        )
        debug_info["confidence_signals"] = confidence_signals

        if not context.evidence_facts:
            return self._abstain(
                query_plan.reasoning_route,
                context,
                trace_id,
                debug_info=debug_info,
                confidence_signals=confidence_signals,
                answer="INSUFFICIENT_EVIDENCE",
                summary="reasoning_not_executed",
                failure_mode="no_evidence",
            )

        statements = [fact.statement for fact in context.evidence_facts]
        if not statements:
            return self._abstain(
                query_plan.reasoning_route,
                context,
                trace_id,
                debug_info=debug_info,
                confidence_signals=confidence_signals,
                answer="INSUFFICIENT_EVIDENCE",
                summary="reasoning_not_executed",
                failure_mode="no_evidence",
            )

        entity_terms = {
            AliasRegistry.normalize(entity.canonical_name)
            for entity in resolved_entities
            if entity.canonical_name
        }
        entity_terms.update(
            AliasRegistry.normalize(entity.entity_id)
            for entity in resolved_entities
            if entity.entity_id
        )
        has_entity_grounding = (
            any(
                any(
                    term
                    and (
                        term == AliasRegistry.normalize(statement.subject)
                        or term == AliasRegistry.normalize(statement.object)
                        or term
                        == AliasRegistry.normalize(statement.canonical_subject_id or "")
                        or term
                        == AliasRegistry.normalize(statement.canonical_object_id or "")
                    )
                    for term in entity_terms
                )
                for statement in statements
            )
            if entity_terms
            else True
        )
        if not has_entity_grounding:
            return self._abstain(
                query_plan.reasoning_route,
                context,
                trace_id,
                debug_info=debug_info,
                confidence_signals=confidence_signals,
                answer="INSUFFICIENT_EVIDENCE",
                summary="reasoning_not_executed",
                failure_mode="not_grounded",
            )

        selected_operator = self.operator_selector.select(
            query=query,
            query_plan=query_plan,
            resolved_entities=resolved_entities,
            statements=statements,
        )
        debug_info["selected_operator"] = selected_operator
        reasoner_outcome = None
        if selected_operator:
            operator_outcome = self.operator_executor.execute(
                operator_hint=selected_operator,
                query=query,
                evidence_facts=statements,
                resolved_entities=resolved_entities,
            )
            confidence_signals["operator_confidence"] = (
                float(operator_outcome.confidence)
                if operator_outcome is not None
                else 0.0
            )
            confidence_signals["operator_failed"] = operator_outcome is None and (
                self.operator_executor.required_input_satisfied(
                    selected_operator,
                    query,
                    statements,
                    resolved_entities,
                )
            )
            debug_info["operator_summary"] = (
                operator_outcome.summary if operator_outcome is not None else ""
            )
            reasoner_outcome = self._operator_to_reasoner_outcome(operator_outcome)

        if reasoner_outcome is None:
            reasoner_outcome = self._run_reasoner(
                query,
                statements,
                candidate_paths=context.candidate_paths,
            )
        logger.info("Reasoner executed: %s", reasoner_outcome is not None)
        debug_info["reasoner_summary"] = (
            reasoner_outcome.summary if reasoner_outcome is not None else ""
        )
        debug_info["reasoning_path_length"] = (
            len(reasoner_outcome.reasoning_path) if reasoner_outcome else 0
        )
        if reasoner_outcome is None:
            return self._abstain(
                query_plan.reasoning_route,
                context,
                trace_id,
                debug_info=debug_info,
                confidence_signals=confidence_signals,
                answer="REASONING_FAILED",
                summary="reasoning_failed",
                failure_mode="reasoning_failed",
            )

        confidence_signals["path_score"] = self._path_score_from_outcome(
            reasoner_outcome,
            context,
        )

        verification = self.verifier.verify(
            query=query,
            context=statements,
            reasoning_path=reasoner_outcome.reasoning_path,
            answer=reasoner_outcome.answer,
            confidence_score=reasoner_outcome.confidence,
            required_facts=reasoner_outcome.required_facts,
            operator_checks=reasoner_outcome.operator_checks,
        )
        logger.info("Verifier executed: True")
        logger.info("Verification passed: %s", verification.passed)
        logger.info("Verification issues: %s", verification.issues)
        logger.info("Contradiction detected: %s", verification.contradictory)
        debug_info["verification_passed"] = verification.passed
        debug_info["verification_issues"] = list(verification.issues)
        debug_info["contradiction_detected"] = verification.contradictory
        confidence_signals["verification_pass"] = verification.passed
        confidence_signals["verification_issues"] = list(verification.issues)
        confidence_signals["contradiction_flag"] = verification.contradictory
        calibrated_confidence = self.confidence_calibrator.calibrate(confidence_signals)
        debug_info["calibrated_confidence"] = calibrated_confidence

        if not verification.passed:
            abstain_answer, failure_mode = self._verification_failure_policy(
                verification.issues
            )
            logger.info("Abstention reason: %s", failure_mode)
            return self._abstain(
                query_plan.reasoning_route,
                context,
                trace_id,
                debug_info=debug_info,
                confidence_signals=confidence_signals,
                answer=abstain_answer,
                summary=verification.summary,
                failure_mode=failure_mode,
            )

        if not reasoner_outcome.reasoning_path and reasoner_outcome.answer not in {
            "yes",
            "no",
        }:
            logger.info("Abstention reason: verification_failed")
            return self._abstain(
                query_plan.reasoning_route,
                context,
                trace_id,
                debug_info=debug_info,
                confidence_signals=confidence_signals,
                answer="INSUFFICIENT_EVIDENCE",
                summary="verification_failed:empty_reasoning_path",
                failure_mode="verification_failed",
            )

        if self.confidence_calibrator.should_abstain(
            calibrated_confidence, confidence_signals
        ):
            return self._abstain(
                query_plan.reasoning_route,
                context,
                trace_id,
                debug_info=debug_info,
                confidence_signals=confidence_signals,
                answer="INSUFFICIENT_EVIDENCE",
                summary="verification_failed:calibrated_confidence_below_threshold",
                failure_mode="low_confidence",
            )

        debug_info["abstained"] = False
        debug_info["final_answer"] = reasoner_outcome.answer
        debug_info["final_confidence"] = calibrated_confidence
        debug_info["failure_mode"] = None
        return QueryResult(
            answer=reasoner_outcome.answer,
            confidence=calibrated_confidence,
            reasoning_route=query_plan.reasoning_route,
            evidence_facts=context.evidence_facts,
            candidate_paths=context.candidate_paths,
            confidence_signals=confidence_signals,
            verification_summary=verification.summary,
            trace_id=trace_id,
            failure_mode=None,
            debug_info=debug_info,
        )

    def _build_debug_info(
        self,
        context: ReasoningContext,
        query_plan,
        trace_id: str,
    ) -> dict[str, object]:
        trace_metadata = dict(context.trace_metadata)
        return {
            "trace_id": trace_id,
            "reasoning_route": getattr(query_plan, "reasoning_route", ""),
            "route_confidence": getattr(query_plan, "route_confidence", 0.0),
            "target_relations": list(trace_metadata.get("target_relations", [])),
            "resolved_entities": list(trace_metadata.get("resolved_entities", [])),
            "retrieved_chunk_count": len(context.retrieved_chunks),
            "evidence_fact_count": len(context.evidence_facts),
            "evidence_facts_before_filtering": trace_metadata.get(
                "evidence_facts_before_filtering", len(context.evidence_facts)
            ),
            "evidence_facts_after_filtering": trace_metadata.get(
                "evidence_facts_after_filtering", len(context.evidence_facts)
            ),
            "candidate_path_count": len(context.candidate_paths),
            "candidate_paths_before_scoring": trace_metadata.get(
                "candidate_paths_before_scoring", len(context.candidate_paths)
            ),
            "subgraph_entity_count": trace_metadata.get("subgraph_entity_count", 0),
            "subgraph_edge_count": trace_metadata.get("subgraph_edge_count", 0),
            "selected_operator": None,
            "operator_summary": "",
            "reasoner_summary": "",
            "reasoning_path_length": 0,
            "verification_passed": None,
            "verification_issues": [],
            "contradiction_detected": False,
            "abstained": None,
            "final_answer": "",
            "final_confidence": 0.0,
            "failure_mode": None,
            "confidence_signals": {},
        }

    def _initial_confidence_signals(
        self,
        context: ReasoningContext,
        query_plan,
        resolved_entities,
    ) -> dict[str, object]:
        evidence_scores = [
            max(
                float(getattr(fact, "trust_score", 0.0) or 0.0),
                float(getattr(fact, "retrieval_score", 0.0) or 0.0),
                float(
                    getattr(getattr(fact, "statement", None), "confidence", 0.0) or 0.0
                ),
            )
            for fact in context.evidence_facts
        ]
        entity_confidences = [entity.link_confidence for entity in resolved_entities]
        verification_issues: list[str] = []
        signals = {
            "evidence_count": len(context.evidence_facts),
            "top_evidence_score": max(evidence_scores) if evidence_scores else 0.0,
            "path_score": (
                max((path.path_score for path in context.candidate_paths), default=0.0)
            ),
            "operator_confidence": 0.0,
            "entity_resolution_confidence": (
                sum(entity_confidences) / len(entity_confidences)
                if entity_confidences
                else 0.0
            ),
            "verification_issues": verification_issues,
            "route_confidence": float(
                getattr(
                    query_plan,
                    "route_confidence",
                    getattr(query_plan, "confidence_score", 0.65),
                )
            ),
        }
        signals["verification_pass"] = not verification_issues
        signals["contradiction_flag"] = any(())
        signals["operator_failed"] = any(())
        return signals

    @staticmethod
    def _path_score_from_outcome(
        outcome: ReasonerOutcome,
        context: ReasoningContext,
    ) -> float:
        if outcome.reasoning_path:
            path_keys = {statement.key() for statement in outcome.reasoning_path}
            for candidate_path in context.candidate_paths:
                candidate_keys = {
                    statement.key() for statement in candidate_path.statements
                }
                if candidate_keys and candidate_keys.issuperset(path_keys):
                    return float(candidate_path.path_score)
        return max((path.path_score for path in context.candidate_paths), default=0.0)

    def _operator_to_reasoner_outcome(self, outcome):
        if outcome is None:
            return None
        value = outcome.result_value
        if isinstance(value, bool):
            answer = "yes" if value else "no"
        elif value is None:
            return None
        else:
            answer = str(value)

        required_facts = []
        if outcome.operator_name not in {"count"}:
            required_facts = [
                (st.subject, st.relation) for st in outcome.supporting_facts
            ]

        return ReasonerOutcome(
            answer=answer,
            confidence=max(0.0, min(1.0, float(outcome.confidence))),
            reasoning_path=list(outcome.supporting_facts),
            required_facts=required_facts,
            operator_checks=[
                {
                    "operator": outcome.operator_name,
                    "inputs": outcome.normalized_inputs,
                    "result": outcome.result_value,
                    "summary": outcome.summary,
                }
            ],
            summary=outcome.summary or "operator_completed",
        )

    def _run_reasoner(self, query: str, statements, candidate_paths=None):
        try:
            outcome = self.reasoner_adapter.reason(
                query,
                statements,
                candidate_paths=candidate_paths or [],
            )
            if isinstance(outcome, ReasonerOutcome):
                return outcome
        except Exception:
            logger.exception("Reasoner execution failed")
            return None

        return None

    def _verification_failure_policy(self, issues: list[str]) -> tuple[str, str]:
        if "contradictory_evidence" in issues:
            return "CONFLICTING_EVIDENCE", "contradictory_evidence"
        if "confidence_below_threshold" in issues:
            return "INSUFFICIENT_EVIDENCE", "low_confidence"
        if "not_grounded" in issues:
            return "INSUFFICIENT_EVIDENCE", "not_grounded"
        if "operator_result_invalid" in issues:
            return "INSUFFICIENT_EVIDENCE", "verification_failed"
        if "evidence_incomplete" in issues or "operator_inputs_missing" in issues:
            return "INSUFFICIENT_EVIDENCE", "no_evidence"
        return "INSUFFICIENT_EVIDENCE", "verification_failed"

    def _abstain(
        self,
        reasoning_route: str,
        context: ReasoningContext,
        trace_id: str,
        debug_info: dict[str, object],
        confidence_signals: dict[str, object],
        answer: str,
        summary: str,
        failure_mode: str,
    ) -> QueryResult:
        debug_info["abstained"] = True
        debug_info["final_answer"] = answer
        debug_info["final_confidence"] = 0.0
        debug_info["failure_mode"] = failure_mode
        debug_info["confidence_signals"] = confidence_signals
        return QueryResult(
            answer=answer,
            confidence=0.0,
            reasoning_route=reasoning_route,
            evidence_facts=context.evidence_facts,
            candidate_paths=context.candidate_paths,
            confidence_signals=confidence_signals,
            verification_summary=summary,
            trace_id=trace_id,
            failure_mode=failure_mode,
            debug_info=debug_info,
        )
