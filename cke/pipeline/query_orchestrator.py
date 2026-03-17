"""Central query orchestration skeleton for CKE pipeline."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING
from uuid import uuid4

from cke.pipeline.types import QueryResult, ReasonerOutcome, ReasoningContext, ResolvedEntity
from cke.reasoning.verifier import ReasoningVerifier

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
        self.last_context: ReasoningContext | None = None

    def answer(self, query: str) -> QueryResult:
        query_plan = self.router.route(query)

        detected_entities = self.router.detect_entities(query)
        resolved_entities = [
            ResolvedEntity(
                surface_form=entity,
                canonical_name=entity,
                entity_id=entity,
                link_confidence=0.7,
                aliases_matched=[entity],
            )
            for entity in detected_entities
        ]

        if self.retriever is not None and self.assembler is not None:
            retrieved_chunks, facts = self.retriever.retrieve(query)
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
                subgraph={
                    "entities": [],
                    "facts_by_entity": {},
                    "facts_by_relation": {},
                },
                decomposition=list(getattr(query_plan, "decomposition", [])),
                trace_metadata={},
            )

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

        self.last_context = context
        trace_id = str(uuid4())

        if not context.evidence_facts:
            return self._abstain(
                query_plan.reasoning_route,
                context,
                trace_id,
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
                answer="INSUFFICIENT_EVIDENCE",
                summary="reasoning_not_executed",
                failure_mode="no_evidence",
            )

        entity_terms = [
            entity.canonical_name.lower()
            for entity in resolved_entities
            if entity.canonical_name
        ]
        has_entity_grounding = (
            any(
                any(
                    term in statement.subject.lower() or term in statement.object.lower()
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
                answer="INSUFFICIENT_EVIDENCE",
                summary="reasoning_not_executed",
                failure_mode="not_grounded",
            )

        reasoner_outcome = self._run_reasoner(query, statements)
        logger.info("Reasoner executed: %s", reasoner_outcome is not None)
        if reasoner_outcome is None:
            return self._abstain(
                query_plan.reasoning_route,
                context,
                trace_id,
                answer="REASONING_FAILED",
                summary="reasoning_failed",
                failure_mode="reasoning_failed",
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

        if not verification.passed:
            abstain_answer, failure_mode = self._verification_failure_policy(verification.issues)
            logger.info("Abstention reason: %s", failure_mode)
            return self._abstain(
                query_plan.reasoning_route,
                context,
                trace_id,
                answer=abstain_answer,
                summary=verification.summary,
                failure_mode=failure_mode,
            )

        if not reasoner_outcome.reasoning_path and reasoner_outcome.answer not in {"yes", "no"}:
            logger.info("Abstention reason: verification_failed")
            return self._abstain(
                query_plan.reasoning_route,
                context,
                trace_id,
                answer="INSUFFICIENT_EVIDENCE",
                summary="verification_failed:empty_reasoning_path",
                failure_mode="verification_failed",
            )

        return QueryResult(
            answer=reasoner_outcome.answer,
            confidence=reasoner_outcome.confidence,
            reasoning_route=query_plan.reasoning_route,
            evidence_facts=context.evidence_facts,
            candidate_paths=context.candidate_paths,
            verification_summary=verification.summary,
            trace_id=trace_id,
            failure_mode=None,
        )

    def _run_reasoner(self, query: str, statements):
        try:
            if hasattr(self.reasoner, "reason"):
                outcome = self.reasoner.reason(query, statements)
                if isinstance(outcome, ReasonerOutcome):
                    return outcome
            answer = self.reasoner.answer(query, statements)
        except Exception:
            logger.exception("Reasoner execution failed")
            return None

        if (
            not answer
            or "don't have enough" in answer.lower()
            or "insufficient" in answer.lower()
        ):
            return ReasonerOutcome(
                answer="INSUFFICIENT_EVIDENCE",
                confidence=0.0,
                reasoning_path=[],
                required_facts=[],
                operator_checks=[],
                summary="reasoner_abstained",
            )

        confidence = 0.8 if statements else 0.0
        return ReasonerOutcome(
            answer=answer,
            confidence=confidence,
            reasoning_path=[st for st in statements if st.object.lower() == answer.lower()],
            required_facts=[],
            operator_checks=[],
            summary="reasoner_completed",
        )

    def _verification_failure_policy(self, issues: list[str]) -> tuple[str, str]:
        if "contradictory_evidence" in issues:
            return "CONFLICTING_EVIDENCE", "contradictory_evidence"
        if "confidence_below_threshold" in issues:
            return "INSUFFICIENT_EVIDENCE", "low_confidence"
        if "not_grounded" in issues:
            return "INSUFFICIENT_EVIDENCE", "not_grounded"
        if "evidence_incomplete" in issues:
            return "INSUFFICIENT_EVIDENCE", "no_evidence"
        return "INSUFFICIENT_EVIDENCE", "verification_failed"

    def _abstain(
        self,
        reasoning_route: str,
        context: ReasoningContext,
        trace_id: str,
        answer: str,
        summary: str,
        failure_mode: str,
    ) -> QueryResult:
        return QueryResult(
            answer=answer,
            confidence=0.0,
            reasoning_route=reasoning_route,
            evidence_facts=context.evidence_facts,
            candidate_paths=context.candidate_paths,
            verification_summary=summary,
            trace_id=trace_id,
            failure_mode=failure_mode,
        )
