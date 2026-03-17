"""Sprint 2 integration test for retrieval -> evidence flow."""

from dataclasses import dataclass

from cke.models import Statement
from cke.pipeline.evidence_assembler import EvidenceAssembler
from cke.pipeline.query_orchestrator import QueryOrchestrator
from cke.pipeline.types import QueryResult
from cke.retrieval.chunk_fact_store import ChunkFactStore
from cke.retrieval.evidence_retriever import EvidenceRetriever


@dataclass
class StubQueryPlan:
    reasoning_route: str = "advanced_reasoner"


class StubRouter:
    def route(self, query: str) -> StubQueryPlan:
        return StubQueryPlan(reasoning_route="advanced_reasoner")

    def detect_entities(self, query: str) -> list[str]:
        return ["United States"]


class StubRAGRetriever:
    def __init__(self, docs: list[dict[str, str | float]]) -> None:
        self.docs = docs

    def retrieve(self, query: str, k: int = 5) -> list[dict[str, str | float]]:
        del query
        return self.docs[:k]


def test_orchestrator_sprint2_retrieval_evidence_flow():
    docs = [
        {
            "doc_id": "doc-us-pres::chunk-0",
            "text": "Joe Biden is the president of the United States.",
            "score": 0.01,
            "source": "doc-us-pres",
        },
        {
            "doc_id": "doc-us-capital::chunk-0",
            "text": "Washington, D.C. is the capital of the United States.",
            "score": 0.02,
            "source": "doc-us-capital",
        },
    ]

    store = ChunkFactStore()
    store.add_facts(
        "doc-us-pres::chunk-0",
        [
            Statement(
                subject="Joe Biden",
                relation="is",
                object="President of the United States",
                chunk_id="doc-us-pres::chunk-0",
                source_doc_id="doc-us-pres",
                source="doc-us-pres",
                trust_score=0.9,
                supporting_span=(0, 43),
            )
        ],
    )
    store.add_facts(
        "doc-us-capital::chunk-0",
        [
            Statement(
                subject="Washington, D.C.",
                relation="is",
                object="Capital of the United States",
                chunk_id="doc-us-capital::chunk-0",
                source_doc_id="doc-us-capital",
                source="doc-us-capital",
                trust_score=0.85,
            )
        ],
    )

    orchestrator = QueryOrchestrator(
        graph_engine=None,
        router=StubRouter(),
        retriever=EvidenceRetriever(StubRAGRetriever(docs), store),
        assembler=EvidenceAssembler(),
    )

    result = orchestrator.answer("Who is the president of the United States?")

    assert isinstance(result, QueryResult)
    assert result.answer != "NOT_IMPLEMENTED"
    assert orchestrator.last_context is not None
    assert orchestrator.last_context.retrieved_chunks
    assert result.evidence_facts
    assert orchestrator.last_context.evidence_facts
