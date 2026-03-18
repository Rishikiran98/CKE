"""Sprint 9 confidence calibration, routing, and abstention tests."""

from dataclasses import dataclass, field

from cke.models import Statement
from cke.pipeline.evidence_assembler import EvidenceAssembler
from cke.pipeline.query_orchestrator import QueryOrchestrator
from cke.pipeline.types import ReasonerOutcome
from cke.reasoning.verification_types import VerificationOutcome
from cke.retrieval.chunk_fact_store import ChunkFactStore
from cke.retrieval.evidence_retriever import EvidenceRetriever
from cke.router.query_plan import QueryPlan
from cke.router.query_router import QueryRouter
from cke.trust.confidence_calibrator import ConfidenceCalibrator


class StubRAGRetriever:
    def __init__(self, docs: list[dict[str, str | float]]) -> None:
        self.docs = docs

    def retrieve(self, query: str, k: int = 5) -> list[dict[str, str | float]]:
        del query
        return self.docs[:k]


class StubReasoner:
    def __init__(self, answer: str, confidence: float) -> None:
        self.answer = answer
        self.confidence = confidence

    def reason(self, query: str, statements: list[Statement]) -> ReasonerOutcome:
        del query
        match = statements[0]
        return ReasonerOutcome(
            answer=self.answer,
            confidence=self.confidence,
            reasoning_path=[match],
            required_facts=[(match.subject, match.relation)],
            operator_checks=[],
            summary="stub_reasoner",
        )


class PassingVerifier:
    def verify(self, **kwargs) -> VerificationOutcome:
        del kwargs
        return VerificationOutcome(
            passed=True,
            evidence_complete=True,
            logical_valid=True,
            confidence_valid=True,
            grounded=True,
            contradictory=False,
            summary="verification_passed",
            issues=[],
        )


class FailingVerifier:
    def verify(self, **kwargs) -> VerificationOutcome:
        del kwargs
        return VerificationOutcome(
            passed=False,
            evidence_complete=True,
            logical_valid=True,
            confidence_valid=False,
            grounded=True,
            contradictory=False,
            summary="verification_failed:confidence_below_threshold",
            issues=["confidence_below_threshold"],
        )


@dataclass
class StubRouter:
    query_plan: QueryPlan = field(
        default_factory=lambda: QueryPlan(
            reasoning_route="graph_traversal",
            route_confidence=0.8,
            target_relations=["nationality"],
        )
    )

    def route(self, query: str) -> QueryPlan:
        del query
        return self.query_plan

    def detect_entities(self, query: str) -> list[str]:
        del query
        return ["Albert Einstein"]


def _build_orchestrator(
    *,
    docs: list[dict[str, str | float]],
    facts: dict[str, list[Statement]],
    reasoner_confidence: float,
    verifier,
    route_confidence: float = 0.8,
) -> QueryOrchestrator:
    store = ChunkFactStore()
    for chunk_id, chunk_facts in facts.items():
        store.add_facts(chunk_id, chunk_facts)

    return QueryOrchestrator(
        graph_engine=None,
        router=StubRouter(
            QueryPlan(
                reasoning_route="graph_traversal",
                route_confidence=route_confidence,
                target_relations=["nationality"],
            )
        ),
        retriever=EvidenceRetriever(StubRAGRetriever(docs), store),
        assembler=EvidenceAssembler(),
        reasoner=StubReasoner(answer="German", confidence=reasoner_confidence),
        verifier=verifier,
    )


def test_sprint9_confidence_sanity_ordering():
    calibrator = ConfidenceCalibrator()

    strong = calibrator.calibrate(
        {
            "evidence_count": 3,
            "top_evidence_score": 0.95,
            "path_score": 0.9,
            "operator_confidence": 0.85,
            "entity_resolution_confidence": 0.95,
            "verification_pass": True,
            "verification_issues": [],
            "contradiction_flag": False,
            "route_confidence": 0.85,
        }
    )
    weak = calibrator.calibrate(
        {
            "evidence_count": 1,
            "top_evidence_score": 0.25,
            "path_score": 0.0,
            "operator_confidence": 0.0,
            "entity_resolution_confidence": 0.35,
            "verification_pass": True,
            "verification_issues": ["confidence_below_threshold"],
            "contradiction_flag": False,
            "route_confidence": 0.3,
        }
    )

    assert strong > weak
    assert strong > 0.7
    assert weak < 0.4


def test_sprint9_abstention_threshold_respects_calibrated_confidence():
    docs = [
        {
            "doc_id": "d1::c0",
            "text": "Albert Einstein nationality German",
            "score": 0.35,
            "source": "d1",
        }
    ]
    facts = {
        "d1::c0": [
            Statement("Albert Einstein", "nationality", "German", confidence=0.3)
        ]
    }
    orchestrator = _build_orchestrator(
        docs=docs,
        facts=facts,
        reasoner_confidence=0.32,
        verifier=PassingVerifier(),
        route_confidence=0.2,
    )
    orchestrator.confidence_calibrator.config.abstain_threshold = 0.9

    result = orchestrator.answer("What is the nationality of Albert Einstein?")

    assert result.answer == "INSUFFICIENT_EVIDENCE"
    assert result.failure_mode == "low_confidence"
    assert result.confidence == 0.0
    assert result.confidence_signals["evidence_count"] == 1


def test_sprint9_verification_override_forces_zero_confidence_and_abstention():
    docs = [
        {
            "doc_id": "d2::c0",
            "text": "Albert Einstein nationality German",
            "score": 0.95,
            "source": "d2",
        }
    ]
    facts = {
        "d2::c0": [
            Statement("Albert Einstein", "nationality", "German", confidence=0.95)
        ]
    }
    orchestrator = _build_orchestrator(
        docs=docs,
        facts=facts,
        reasoner_confidence=0.92,
        verifier=FailingVerifier(),
    )

    result = orchestrator.answer("What is the nationality of Albert Einstein?")

    assert result.answer == "INSUFFICIENT_EVIDENCE"
    assert result.confidence == 0.0
    assert result.confidence_signals["verification_pass"] is False
    assert result.confidence_signals["verification_issues"] == [
        "confidence_below_threshold"
    ]


def test_sprint9_router_prefers_operator_and_multihop_routes_with_confidence():
    router = QueryRouter(graph_engine=None)

    operator_plan = router.route("How many children does Albert Einstein have?")
    path_plan = router.route(
        "Which film is associated with the character portrayed by Person X?"
    )
    direct_plan = router.route("What is the nationality of Albert Einstein?")

    assert operator_plan.operator_hint == "count"
    assert operator_plan.reasoning_route == "answer_immediately"
    assert operator_plan.route_confidence >= 0.85

    assert path_plan.multi_hop_hint is True
    assert path_plan.reasoning_route == "advanced_reasoner"
    assert path_plan.route_confidence >= 0.8

    assert direct_plan.reasoning_route == "answer_immediately"
    assert direct_plan.route_confidence >= 0.8
