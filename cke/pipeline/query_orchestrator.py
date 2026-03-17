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

        retrieved_chunks = []
        evidence_facts = []
        if self.retriever is not None and self.assembler is not None:
            retrieved_chunks, facts = self.retriever.retrieve(query)
            evidence_facts = self.assembler.assemble(query, retrieved_chunks, facts)

        logger.info(
            "Passing %s evidence facts to downstream reasoning.", len(evidence_facts)
        )

        context = ReasoningContext(
            query=query,
            query_plan=query_plan,
            resolved_entities=resolved_entities,
            retrieved_chunks=retrieved_chunks,
            evidence_facts=evidence_facts,
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
            evidence_facts=evidence_facts,
            candidate_paths=[],
            verification_summary="not_executed",
            trace_id=trace_id,
            failure_mode="not_implemented",
        )
