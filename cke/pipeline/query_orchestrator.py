"""Central query orchestration skeleton for CKE pipeline."""

from __future__ import annotations

from uuid import uuid4

from typing import TYPE_CHECKING

from cke.pipeline.types import QueryResult, ReasoningContext, ResolvedEntity

if TYPE_CHECKING:
    from cke.router.query_router import QueryRouter


class QueryOrchestrator:
    """Coordinates query routing and placeholder reasoning steps."""

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

        context = ReasoningContext(
            query=query,
            query_plan=query_plan,
            resolved_entities=resolved_entities,
            retrieved_chunks=[],
            evidence_facts=[],
            candidate_paths=[],
            subgraph=None,
            decomposition=[],
            trace_metadata={},
        )

        self.last_context = context

        trace_id = str(uuid4())

        return QueryResult(
            answer="NOT_IMPLEMENTED",
            confidence=0.0,
            reasoning_route=query_plan.reasoning_route,
            evidence_facts=[],
            candidate_paths=[],
            verification_summary="not_executed",
            trace_id=trace_id,
            failure_mode="not_implemented",
        )
