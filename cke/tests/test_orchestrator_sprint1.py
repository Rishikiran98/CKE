"""Sprint 1 integration test for query orchestrator skeleton."""

from dataclasses import dataclass

from cke.pipeline.query_orchestrator import QueryOrchestrator
from cke.pipeline.types import QueryResult


@dataclass
class StubQueryPlan:
    reasoning_route: str = "advanced_reasoner"


class StubRouter:
    def route(self, query: str) -> StubQueryPlan:
        return StubQueryPlan(reasoning_route="advanced_reasoner")

    def detect_entities(self, query: str):
        return ["Albert Einstein"]


def test_orchestrator_sprint1_pipeline_skeleton_runs():
    orchestrator = QueryOrchestrator(graph_engine=None, router=StubRouter())

    result = orchestrator.answer("What is the nationality of Albert Einstein?")

    assert isinstance(result, QueryResult)
    assert result.reasoning_route is not None
    assert result.answer == "INSUFFICIENT_EVIDENCE"
    assert result.failure_mode == "no_evidence_facts"

    assert orchestrator.last_context is not None
    assert orchestrator.last_context.resolved_entities
