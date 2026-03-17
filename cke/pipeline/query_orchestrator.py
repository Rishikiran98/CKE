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
from cke.reasoning.operator_executor import OperatorExecutor
from cke.reasoning.operator_selector import OperatorSelector
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
        self.last_context: ReasoningContext | None = None
        self.entity_resolver = entity_resolver or EntityResolver()

    def answer(self, query: str) -> QueryResult:
        query_plan = self.router.route(query)

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
                subgraph={
                    "entities": [],
                    "facts_by_entity": {},
                    "facts_by_relation": {},
                },
                decomposition=list(getattr(query_plan, "decomposition", [])),
                trace_metadata={},
            )

        context.trace_metadata.setdefault("target_relations", target_relations)
        context.trace_metadata.setdefault("resolved_entities", [
            {"surface_form": e.surface_form, "canonical_name": e.canonical_name, "entity_id": e.entity_id, "link_confidence": e.link_confidence}
            for e in resolved_entities
        ])

        logger.info("Mentions detected: %s", [e.surface_form for e in resolved_entities])
        logger.info("Canonical resolution results: %s", [(e.surface_form, e.canonical_name, e.link_confidence) for e in resolved_entities])
        logger.info("Alias matches used: %s", [e.aliases_matched for e in resolved_entities])
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
                        or term == AliasRegistry.normalize(statement.canonical_subject_id or "")
                        or term == AliasRegistry.normalize(statement.canonical_object_id or "")
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
        reasoner_outcome = None
        if selected_operator:
            operator_outcome = self.operator_executor.execute(
                operator_hint=selected_operator,
                query=query,
                evidence_facts=statements,
                resolved_entities=resolved_entities,
            )
            reasoner_outcome = self._operator_to_reasoner_outcome(operator_outcome)

        if reasoner_outcome is None:
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
            abstain_answer, failure_mode = self._verification_failure_policy(
                verification.issues
            )
            logger.info("Abstention reason: %s", failure_mode)
            return self._abstain(
                query_plan.reasoning_route,
                context,
                trace_id,
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
            reasoning_path=[
                st for st in statements if st.object.lower() == answer.lower()
            ],
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
