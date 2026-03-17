"""Central query orchestration skeleton for CKE pipeline."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING
from uuid import uuid4

from cke.pipeline.types import QueryResult, ReasoningContext, ResolvedEntity

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
        self.verifier = verifier
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
            return QueryResult(
                answer="INSUFFICIENT_EVIDENCE",
                confidence=0.0,
                reasoning_route=query_plan.reasoning_route,
                evidence_facts=context.evidence_facts,
                candidate_paths=context.candidate_paths,
                verification_summary="reasoning_not_executed",
                trace_id=trace_id,
                failure_mode="no_evidence_facts",
            )

        statements = [fact.statement for fact in context.evidence_facts]
        if not statements:
            return QueryResult(
                answer="INSUFFICIENT_EVIDENCE",
                confidence=0.0,
                reasoning_route=query_plan.reasoning_route,
                evidence_facts=context.evidence_facts,
                candidate_paths=context.candidate_paths,
                verification_summary="reasoning_not_executed",
                trace_id=trace_id,
                failure_mode="no_statements",
            )

        entity_terms = [
            entity.canonical_name.lower()
            for entity in resolved_entities
            if entity.canonical_name
        ]
        has_entity_grounding = (
            any(
                any(
                    term in statement.subject.lower()
                    or term in statement.object.lower()
                    for term in entity_terms
                )
                for statement in statements
            )
            if entity_terms
            else True
        )
        if not has_entity_grounding:
            return QueryResult(
                answer="INSUFFICIENT_EVIDENCE",
                confidence=0.0,
                reasoning_route=query_plan.reasoning_route,
                evidence_facts=context.evidence_facts,
                candidate_paths=context.candidate_paths,
                verification_summary="reasoning_not_executed",
                trace_id=trace_id,
                failure_mode="entity_not_grounded",
            )

        try:
            answer = self.reasoner.answer(query, statements)
            logger.info("Reasoner executed: True")
        except Exception:
            logger.exception("Reasoner execution failed")
            logger.info("Reasoner executed: False")
            return QueryResult(
                answer="INSUFFICIENT_EVIDENCE",
                confidence=0.0,
                reasoning_route=query_plan.reasoning_route,
                evidence_facts=context.evidence_facts,
                candidate_paths=context.candidate_paths,
                verification_summary="reasoning_failed",
                trace_id=trace_id,
                failure_mode="reasoner_error",
            )

        if (
            not answer
            or "don't have enough" in answer.lower()
            or "insufficient" in answer.lower()
        ):
            return QueryResult(
                answer="INSUFFICIENT_EVIDENCE",
                confidence=0.0,
                reasoning_route=query_plan.reasoning_route,
                evidence_facts=context.evidence_facts,
                candidate_paths=context.candidate_paths,
                verification_summary="not_verified_yet",
                trace_id=trace_id,
                failure_mode="insufficient_evidence",
            )

        confidence = min(1.0, 0.2 + 0.1 * len(context.evidence_facts))
        return QueryResult(
            answer=answer,
            confidence=confidence,
            reasoning_route=query_plan.reasoning_route,
            evidence_facts=context.evidence_facts,
            candidate_paths=context.candidate_paths,
            verification_summary="not_verified_yet",
            trace_id=trace_id,
            failure_mode=None,
        )
