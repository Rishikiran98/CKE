"""Sprint 4 integration tests for verification and abstention discipline."""

from dataclasses import dataclass, field

from cke.models import Statement
from cke.pipeline.evidence_assembler import EvidenceAssembler
from cke.pipeline.query_orchestrator import QueryOrchestrator
from cke.pipeline.types import ReasonerOutcome
from cke.retrieval.chunk_fact_store import ChunkFactStore
from cke.retrieval.evidence_retriever import EvidenceRetriever


@dataclass
class StubQueryPlan:
    reasoning_route: str = "advanced_reasoner"
    decomposition: list[dict[str, str | float]] = field(default_factory=list)


class StubRouter:
    def route(self, query: str) -> StubQueryPlan:
        lowered = query.lower()
        decomposition = []
        if "nationality" in lowered:
            decomposition.append(
                {"type": "relation", "value": "nationality", "confidence": 1.0}
            )
        return StubQueryPlan(
            reasoning_route="advanced_reasoner", decomposition=decomposition
        )

    def detect_entities(self, query: str) -> list[str]:
        entities = []
        for candidate in [
            "Albert Einstein",
            "Scott Derrickson",
            "Ed Wood",
            "Unknown Person",
        ]:
            if candidate.lower() in query.lower():
                entities.append(candidate)
        return entities


class StubRAGRetriever:
    def __init__(self, docs: list[dict[str, str | float]]) -> None:
        self.docs = docs

    def retrieve(self, query: str, k: int = 5) -> list[dict[str, str | float]]:
        del query
        return self.docs[:k]


class StubReasoner:
    def reason(self, query: str, statements: list[Statement]) -> ReasonerOutcome:
        lowered = query.lower()
        if "same nationality" in lowered:
            left = next(st for st in statements if st.subject == "Scott Derrickson")
            right = next(st for st in statements if st.subject == "Ed Wood")
            return ReasonerOutcome(
                answer="yes",
                confidence=0.95,
                reasoning_path=[left, right],
                required_facts=[
                    ("Scott Derrickson", "nationality"),
                    ("Ed Wood", "nationality"),
                ],
                operator_checks=[
                    {
                        "operator": "equality",
                        "inputs": (left.object, right.object),
                        "result": True,
                    }
                ],
                summary="comparison",
            )
        if "albert einstein" in lowered:
            einstein_facts = [
                st
                for st in statements
                if st.subject == "Albert Einstein" and st.relation == "nationality"
            ]
            return ReasonerOutcome(
                answer=einstein_facts[0].object,
                confidence=0.9,
                reasoning_path=einstein_facts,
                required_facts=[("Albert Einstein", "nationality")],
                operator_checks=[],
                summary="direct_lookup",
            )
        return ReasonerOutcome(
            answer="Martian",
            confidence=0.9,
            reasoning_path=[],
            required_facts=[],
            operator_checks=[],
            summary="unsupported",
        )


def _build_orchestrator(docs, fact_map) -> QueryOrchestrator:
    store = ChunkFactStore()
    for chunk_id, statements in fact_map.items():
        store.add_facts(chunk_id, statements)
    return QueryOrchestrator(
        graph_engine=None,
        router=StubRouter(),
        retriever=EvidenceRetriever(StubRAGRetriever(docs), store),
        assembler=EvidenceAssembler(),
        reasoner=StubReasoner(),
    )


def test_sprint4_verified_success():
    docs = [
        {
            "doc_id": "d1::c0",
            "text": "Scott Derrickson nationality American",
            "score": 0.9,
            "source": "d1",
        },
        {
            "doc_id": "d2::c0",
            "text": "Ed Wood nationality American",
            "score": 0.8,
            "source": "d2",
        },
    ]
    facts = {
        "d1::c0": [
            Statement("Scott Derrickson", "nationality", "American", trust_score=0.9)
        ],
        "d2::c0": [Statement("Ed Wood", "nationality", "American", trust_score=0.9)],
    }
    orchestrator = _build_orchestrator(docs, facts)

    result = orchestrator.answer(
        "Were Scott Derrickson and Ed Wood of the same nationality?"
    )

    assert result.answer == "yes"
    assert result.failure_mode is None
    assert result.verification_summary == "verification_passed"


def test_sprint4_not_grounded_failure_returns_abstention():
    docs = [
        {
            "doc_id": "d3::c0",
            "text": "Scott Derrickson nationality American",
            "score": 0.9,
            "source": "d3",
        },
    ]
    facts = {
        "d3::c0": [
            Statement("Scott Derrickson", "nationality", "American", trust_score=0.9)
        ],
    }
    orchestrator = _build_orchestrator(docs, facts)

    result = orchestrator.answer("What is the favorite planet of Scott Derrickson?")

    assert result.answer == "INSUFFICIENT_EVIDENCE"
    assert result.failure_mode in {"not_grounded", "verification_failed"}
    assert result.verification_summary.startswith("verification_failed")


def test_sprint4_contradiction_returns_conflicting_evidence():
    docs = [
        {
            "doc_id": "d4::c0",
            "text": "Albert Einstein nationality German",
            "score": 0.9,
            "source": "d4",
        },
        {
            "doc_id": "d4::c1",
            "text": "Albert Einstein nationality Swiss",
            "score": 0.85,
            "source": "d4",
        },
    ]
    facts = {
        "d4::c0": [
            Statement("Albert Einstein", "nationality", "German", trust_score=0.9)
        ],
        "d4::c1": [
            Statement("Albert Einstein", "nationality", "Swiss", trust_score=0.88)
        ],
    }
    orchestrator = _build_orchestrator(docs, facts)

    result = orchestrator.answer("What is the nationality of Albert Einstein?")

    assert result.answer == "CONFLICTING_EVIDENCE"
    assert result.failure_mode in {"contradictory_evidence", "verification_failed"}


def test_sprint4_missing_evidence_returns_insufficient_evidence():
    docs = []
    facts = {}
    orchestrator = _build_orchestrator(docs, facts)

    result = orchestrator.answer("What is the nationality of Unknown Person?")

    assert result.answer == "INSUFFICIENT_EVIDENCE"
    assert result.failure_mode == "no_evidence"
