"""Sprint 7 integration tests for local subgraph and candidate path assembly."""

from dataclasses import dataclass, field
import re

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
    target_relations: list[str] = field(default_factory=list)
    multi_hop_hint: bool = False
    bridge_entities_expected: bool = False


class StubRouter:
    def route(self, query: str) -> StubQueryPlan:
        lowered = query.lower()
        decomposition: list[dict[str, str | float]] = []
        target_relations: list[str] = []
        multi_hop_hint = any(
            marker in lowered
            for marker in ["character portrayed by", "connected", "between", "via"]
        )
        bridge_entities_expected = multi_hop_hint or "same nationality" in lowered

        if "located" in lowered:
            decomposition.append(
                {"type": "relation", "value": "located_in", "confidence": 1.0}
            )
            target_relations.append("located_in")
        if "portrayed" in lowered:
            decomposition.append(
                {"type": "relation", "value": "portrayed", "confidence": 1.0}
            )
            target_relations.append("portrayed")
        if "film" in lowered or "appears" in lowered:
            decomposition.append(
                {"type": "relation", "value": "appears_in", "confidence": 0.9}
            )
            target_relations.append("appears_in")
        if "nationality" in lowered:
            decomposition.append(
                {"type": "relation", "value": "nationality", "confidence": 1.0}
            )
            target_relations.append("nationality")

        return StubQueryPlan(
            decomposition=decomposition,
            operator_hint=None,
            target_relations=target_relations,
            multi_hop_hint=multi_hop_hint,
            bridge_entities_expected=bridge_entities_expected,
        )

    def detect_entities(self, query: str) -> list[str]:
        entities = []
        lowered = query.lower()
        for candidate in [
            "A",
            "B",
            "C",
            "Person X",
            "Character Y",
            "Film Z",
            "Scott Derrickson",
            "Ed Wood",
            "Bridge Missing",
        ]:
            candidate_lower = candidate.lower()
            if len(candidate) == 1:
                pattern = rf"\b{re.escape(candidate_lower)}\b"
                matched = re.search(pattern, lowered) is not None
            else:
                matched = candidate_lower in lowered
            if matched:
                entities.append(candidate)
        return entities


class StubRAGRetriever:
    def __init__(self, docs: list[dict[str, str | float]]) -> None:
        self.docs = docs

    def retrieve(self, query: str, k: int = 5) -> list[dict[str, str | float]]:
        del query
        return self.docs[:k]


class PathAwareStubReasoner:
    def __init__(self) -> None:
        self.last_candidate_paths = None
        self.last_statements = None

    def reason(
        self,
        query: str,
        statements: list[Statement],
        candidate_paths=None,
    ) -> ReasonerOutcome:
        self.last_candidate_paths = list(candidate_paths or [])
        self.last_statements = list(statements)
        lowered = query.lower()

        if "where is a located in" in lowered:
            best_path = next(
                (
                    path
                    for path in self.last_candidate_paths
                    if len(path.statements) == 2
                ),
                None,
            )
            if best_path:
                return ReasonerOutcome(
                    answer=best_path.statements[-1].object,
                    confidence=0.88,
                    reasoning_path=list(best_path.statements),
                    summary="path_answer",
                )

        if (
            "which film is associated with the character portrayed by person x"
            in lowered
        ):
            best_path = next(
                (
                    path
                    for path in self.last_candidate_paths
                    if len(path.statements) == 2
                    and path.statements[-1].relation == "appears_in"
                ),
                None,
            )
            if best_path:
                return ReasonerOutcome(
                    answer=best_path.statements[-1].object,
                    confidence=0.91,
                    reasoning_path=list(best_path.statements),
                    summary="path_answer",
                )

        if "same nationality" in lowered:
            nationality_facts = [s for s in statements if s.relation == "nationality"]
            if len(nationality_facts) >= 2:
                return ReasonerOutcome(
                    answer="yes",
                    confidence=0.9,
                    reasoning_path=nationality_facts[:2],
                    summary="comparison_context_preserved",
                )

        return ReasonerOutcome(
            answer="INSUFFICIENT_EVIDENCE",
            confidence=0.0,
            reasoning_path=[],
            summary="fallback_abstain",
        )


def _build_orchestrator(
    docs, fact_map
) -> tuple[QueryOrchestrator, PathAwareStubReasoner]:
    store = ChunkFactStore()
    for chunk_id, statements in fact_map.items():
        store.add_facts(chunk_id, statements)

    reasoner = PathAwareStubReasoner()
    orchestrator = QueryOrchestrator(
        graph_engine=None,
        router=StubRouter(),
        retriever=EvidenceRetriever(StubRAGRetriever(docs), store),
        assembler=EvidenceAssembler(max_candidate_paths=10),
        reasoner=reasoner,
    )
    return orchestrator, reasoner


def test_sprint7_direct_path_generation():
    docs = [
        {"doc_id": "d1::c0", "text": "A located_in B", "score": 0.95, "source": "d1"},
        {"doc_id": "d1::c1", "text": "B located_in C", "score": 0.9, "source": "d1"},
    ]
    facts = {
        "d1::c0": [Statement("A", "located_in", "B", trust_score=0.93)],
        "d1::c1": [Statement("B", "located_in", "C", trust_score=0.91)],
    }
    orchestrator, reasoner = _build_orchestrator(docs, facts)

    result = orchestrator.answer("Where is A located in?")

    assert orchestrator.last_context is not None
    assert orchestrator.last_context.subgraph is not None
    assert orchestrator.last_context.subgraph.entities
    assert any(
        len(path.statements) == 2 for path in orchestrator.last_context.candidate_paths
    )
    assert orchestrator.last_context.candidate_paths
    assert reasoner.last_candidate_paths
    assert result.answer == "C"


def test_sprint7_simple_bridge_query_builds_connected_path():
    docs = [
        {
            "doc_id": "d2::c0",
            "text": "Person X portrayed Character Y",
            "score": 0.95,
            "source": "d2",
        },
        {
            "doc_id": "d2::c1",
            "text": "Character Y appears_in Film Z",
            "score": 0.91,
            "source": "d2",
        },
    ]
    facts = {
        "d2::c0": [Statement("Person X", "portrayed", "Character Y", trust_score=0.95)],
        "d2::c1": [Statement("Character Y", "appears_in", "Film Z", trust_score=0.92)],
    }
    orchestrator, reasoner = _build_orchestrator(docs, facts)

    result = orchestrator.answer(
        "Which film is associated with the character portrayed by Person X?"
    )

    assert any(
        [statement.relation for statement in path.statements]
        == ["portrayed", "appears_in"]
        for path in orchestrator.last_context.candidate_paths
    )
    assert reasoner.last_candidate_paths
    assert reasoner.last_statements[:2][0].relation == "portrayed"
    assert reasoner.last_statements[:2][1].relation == "appears_in"
    assert result.answer == "Film Z"


def test_sprint7_dual_entity_connection_preserves_both_sides():
    docs = [
        {
            "doc_id": "d3::c0",
            "text": "Scott Derrickson nationality American",
            "score": 0.93,
            "source": "d3",
        },
        {
            "doc_id": "d3::c1",
            "text": "Ed Wood nationality American",
            "score": 0.92,
            "source": "d3",
        },
    ]
    facts = {
        "d3::c0": [Statement("Scott Derrickson", "nationality", "American")],
        "d3::c1": [Statement("Ed Wood", "nationality", "American")],
    }
    orchestrator, _ = _build_orchestrator(docs, facts)

    result = orchestrator.answer(
        "Were Scott Derrickson and Ed Wood of the same nationality?"
    )

    subjects = {fact.statement.subject for fact in result.evidence_facts}
    assert {"Scott Derrickson", "Ed Wood"}.issubset(subjects)
    assert result.answer in {"yes", "INSUFFICIENT_EVIDENCE"}


def test_sprint7_missing_bridge_abstains_without_hallucinating():
    docs = [
        {
            "doc_id": "d4::c0",
            "text": "Person X portrayed Character Y",
            "score": 0.95,
            "source": "d4",
        },
    ]
    facts = {
        "d4::c0": [Statement("Person X", "portrayed", "Character Y", trust_score=0.95)],
    }
    orchestrator, _ = _build_orchestrator(docs, facts)

    result = orchestrator.answer(
        "Which film is associated with the character portrayed by Person X?"
    )

    two_hop_paths = [
        path
        for path in orchestrator.last_context.candidate_paths
        if len(path.statements) == 2
    ]
    assert two_hop_paths == []
    assert result.answer == "INSUFFICIENT_EVIDENCE"
    assert result.failure_mode in {
        "low_confidence",
        "verification_failed",
        "no_evidence",
    }


def test_sprint7_path_deduplication_removes_duplicate_two_hop_paths():
    docs = [
        {
            "doc_id": "d5::c0",
            "text": "Person X portrayed Character Y",
            "score": 0.95,
            "source": "d5",
        },
        {
            "doc_id": "d5::c1",
            "text": "Character Y appears_in Film Z",
            "score": 0.94,
            "source": "d5",
        },
        {
            "doc_id": "d5::c2",
            "text": "Duplicate Character Y appears_in Film Z",
            "score": 0.93,
            "source": "d5",
        },
    ]
    facts = {
        "d5::c0": [Statement("Person X", "portrayed", "Character Y", trust_score=0.95)],
        "d5::c1": [Statement("Character Y", "appears_in", "Film Z", trust_score=0.92)],
        "d5::c2": [Statement("Character Y", "appears_in", "Film Z", trust_score=0.90)],
    }
    orchestrator, _ = _build_orchestrator(docs, facts)

    orchestrator.answer(
        "Which film is associated with the character portrayed by Person X?"
    )

    two_hop_keys = [
        tuple(statement.key() for statement in path.statements)
        for path in orchestrator.last_context.candidate_paths
        if len(path.statements) == 2
    ]
    assert len(two_hop_keys) == len(set(two_hop_keys))
