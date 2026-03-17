"""Sprint 5 integration tests for deterministic operator coverage."""

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
    operator_hint: str | None = None


class StubRouter:
    def route(self, query: str) -> StubQueryPlan:
        lowered = query.lower()
        hint = None
        decomposition = []
        if "same nationality" in lowered:
            hint = "equality"
            decomposition.append(
                {"type": "relation", "value": "nationality", "confidence": 1.0}
            )
        elif "how many" in lowered:
            hint = "count"
            decomposition.append(
                {"type": "relation", "value": "child", "confidence": 1.0}
            )
        elif lowered.startswith("did ") or lowered.startswith("is "):
            hint = "existence"
            decomposition.append(
                {"type": "relation", "value": "directed", "confidence": 1.0}
            )
        elif "released later" in lowered or "later" in lowered:
            hint = "temporal_compare"
            decomposition.append(
                {"type": "relation", "value": "release_year", "confidence": 1.0}
            )
        return StubQueryPlan(decomposition=decomposition, operator_hint=hint)

    def detect_entities(self, query: str) -> list[str]:
        entities = []
        for candidate in [
            "Scott Derrickson",
            "Ed Wood",
            "Person X",
            "Christopher Nolan",
            "Inception",
            "Film A",
            "Film B",
            "Unknown Film",
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


class GuardReasoner:
    """Reasoner that should only execute for explicit fallback test."""

    def reason(self, query: str, statements: list[Statement]) -> ReasonerOutcome:
        if "unknown film" in query.lower():
            return ReasonerOutcome(
                answer="INSUFFICIENT_EVIDENCE",
                confidence=0.0,
                reasoning_path=[],
                required_facts=[],
                operator_checks=[],
                summary="fallback_abstain",
            )
        raise AssertionError(
            "Deterministic operator path should have handled this query"
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
        reasoner=GuardReasoner(),
    )


def test_sprint5_equality_operator_yes():
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
        "d1::c0": [Statement("Scott Derrickson", "nationality", "American")],
        "d2::c0": [Statement("Ed Wood", "nationality", "American")],
    }
    orchestrator = _build_orchestrator(docs, facts)

    result = orchestrator.answer(
        "Were Scott Derrickson and Ed Wood of the same nationality?"
    )

    assert result.answer == "yes"
    assert result.failure_mode is None
    assert result.verification_summary == "verification_passed"


def test_sprint5_count_operator():
    docs = [
        {"doc_id": "d3::c0", "text": "Person X child A", "score": 0.9, "source": "d3"},
        {"doc_id": "d3::c1", "text": "Person X child B", "score": 0.85, "source": "d3"},
        {"doc_id": "d3::c2", "text": "Person X child C", "score": 0.83, "source": "d3"},
    ]
    facts = {
        "d3::c0": [Statement("Person X", "child", "A")],
        "d3::c1": [Statement("Person X", "child", "B")],
        "d3::c2": [Statement("Person X", "child", "C")],
    }
    orchestrator = _build_orchestrator(docs, facts)

    result = orchestrator.answer("How many children did Person X have?")

    assert result.answer == "3"
    assert result.failure_mode is None
    assert result.verification_summary == "verification_passed"


def test_sprint5_existence_operator():
    docs = [
        {
            "doc_id": "d4::c0",
            "text": "Christopher Nolan directed Inception",
            "score": 0.92,
            "source": "d4",
        },
    ]
    facts = {
        "d4::c0": [Statement("Christopher Nolan", "directed", "Inception")],
    }
    orchestrator = _build_orchestrator(docs, facts)

    result = orchestrator.answer("Did Christopher Nolan direct Inception?")

    assert result.answer == "yes"
    assert result.failure_mode is None
    assert result.verification_summary == "verification_passed"


def test_sprint5_temporal_compare_operator():
    docs = [
        {
            "doc_id": "d5::c0",
            "text": "Film A release_year 1990",
            "score": 0.9,
            "source": "d5",
        },
        {
            "doc_id": "d5::c1",
            "text": "Film B release_year 2001",
            "score": 0.88,
            "source": "d5",
        },
    ]
    facts = {
        "d5::c0": [Statement("Film A", "release_year", "1990")],
        "d5::c1": [Statement("Film B", "release_year", "2001")],
    }
    orchestrator = _build_orchestrator(docs, facts)

    result = orchestrator.answer("Which film was released later, Film A or Film B?")

    assert result.answer == "Film B"
    assert result.failure_mode is None
    assert result.verification_summary == "verification_passed"


def test_sprint5_missing_input_abstention():
    docs = [
        {
            "doc_id": "d6::c0",
            "text": "Film A release_year 1990",
            "score": 0.9,
            "source": "d6",
        },
    ]
    facts = {
        "d6::c0": [Statement("Film A", "release_year", "1990")],
    }
    orchestrator = _build_orchestrator(docs, facts)

    result = orchestrator.answer(
        "Which film was released later, Film A or Unknown Film?"
    )

    assert result.answer == "INSUFFICIENT_EVIDENCE"
    assert result.failure_mode in {
        "no_evidence",
        "verification_failed",
        "low_confidence",
    }
