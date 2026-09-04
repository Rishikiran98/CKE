"""The query orchestrator runs end to end on a minimal graph."""

from cke.pipeline.query_orchestrator import QueryOrchestrator
from cke.pipeline.types import QueryResult
from cke.router.query_router import QueryRouter


def test_pipeline_skeleton_runs():
    orchestrator = QueryOrchestrator(graph_engine=None, router=QueryRouter())

    result = orchestrator.answer("What is the nationality of Albert Einstein?")

    assert isinstance(result, QueryResult)
    assert result.reasoning_route is not None
    assert result.answer == "INSUFFICIENT_EVIDENCE"
    assert result.failure_mode == "no_evidence"

    assert orchestrator.last_context is not None
    assert orchestrator.last_context.resolved_entities
