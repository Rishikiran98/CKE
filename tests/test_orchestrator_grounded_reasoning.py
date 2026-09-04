"""Grounded reasoning: direct lookup, shallow comparison, and the structured
fallback when evidence is insufficient.
"""

from cke.models import Statement
from cke.pipeline.evidence_assembler import EvidenceAssembler
from cke.pipeline.query_orchestrator import QueryOrchestrator
from cke.retrieval.chunk_fact_store import ChunkFactStore
from cke.retrieval.evidence_retriever import EvidenceRetriever
from cke.router.query_router import QueryRouter


class StubRAGRetriever:
    def __init__(self, docs: list[dict[str, str | float]]) -> None:
        self.docs = docs

    def retrieve(self, query: str, k: int = 5) -> list[dict[str, str | float]]:
        del query
        return self.docs[:k]


def _build_orchestrator() -> QueryOrchestrator:
    docs = [
        {
            "doc_id": "doc-einstein::chunk-0",
            "text": "Albert Einstein nationality German",
            "score": 0.9,
            "source": "doc-einstein",
        },
        {
            "doc_id": "doc-curie::chunk-0",
            "text": "Marie Curie nationality Polish",
            "score": 0.8,
            "source": "doc-curie",
        },
        {
            "doc_id": "doc-derrickson::chunk-0",
            "text": "Scott Derrickson nationality American",
            "score": 0.7,
            "source": "doc-derrickson",
        },
        {
            "doc_id": "doc-edwood::chunk-0",
            "text": "Ed Wood nationality American",
            "score": 0.6,
            "source": "doc-edwood",
        },
    ]

    store = ChunkFactStore()
    store.add_facts(
        "doc-einstein::chunk-0",
        [
            Statement(
                subject="Albert Einstein",
                relation="nationality",
                object="German",
                chunk_id="doc-einstein::chunk-0",
                source="doc-einstein",
                trust_score=0.95,
            )
        ],
    )
    store.add_facts(
        "doc-curie::chunk-0",
        [
            Statement(
                subject="Marie Curie",
                relation="nationality",
                object="Polish",
                chunk_id="doc-curie::chunk-0",
                source="doc-curie",
                trust_score=0.9,
            )
        ],
    )
    store.add_facts(
        "doc-derrickson::chunk-0",
        [
            Statement(
                subject="Scott Derrickson",
                relation="nationality",
                object="American",
                chunk_id="doc-derrickson::chunk-0",
                source="doc-derrickson",
                trust_score=0.85,
            )
        ],
    )
    store.add_facts(
        "doc-edwood::chunk-0",
        [
            Statement(
                subject="Ed Wood",
                relation="nationality",
                object="American",
                chunk_id="doc-edwood::chunk-0",
                source="doc-edwood",
                trust_score=0.84,
            )
        ],
    )

    return QueryOrchestrator(
        graph_engine=None,
        router=QueryRouter(),
        retriever=EvidenceRetriever(StubRAGRetriever(docs), store),
        assembler=EvidenceAssembler(),
    )


def test_direct_fact_lookup():
    orchestrator = _build_orchestrator()

    result = orchestrator.answer("What is the nationality of Albert Einstein?")

    assert result.answer != "NOT_IMPLEMENTED"
    assert result.answer != "INSUFFICIENT_EVIDENCE"
    assert result.evidence_facts
    assert result.failure_mode is None


def test_shallow_comparison():
    orchestrator = _build_orchestrator()

    result = orchestrator.answer(
        "Were Scott Derrickson and Ed Wood of the same nationality?"
    )

    assert result.answer == "yes"


def test_insufficient_evidence_structured_fallback():
    orchestrator = _build_orchestrator()

    result = orchestrator.answer("What is the nationality of Nikola Tesla?")

    assert result.answer == "INSUFFICIENT_EVIDENCE"
    assert result.failure_mode is not None
    assert result.reasoning_route is not None
