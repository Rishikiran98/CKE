"""Sprint 6 integration tests for aliasing and relation-targeted retrieval."""

from dataclasses import dataclass, field

from cke.entity_resolution.entity_resolver import EntityResolver
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


class StubRouter:
    def route(self, query: str) -> StubQueryPlan:
        lowered = query.lower()
        decomposition = []
        target_relations: list[str] = []
        operator_hint = None

        if "nationality" in lowered or "citizenship" in lowered:
            decomposition.append(
                {"type": "relation", "value": "nationality", "confidence": 1.0}
            )
            target_relations.append("nationality")
        if "direct" in lowered:
            decomposition.append(
                {"type": "relation", "value": "directed", "confidence": 1.0}
            )
            target_relations.append("directed")
            operator_hint = "existence"
        if "same nationality" in lowered:
            operator_hint = "equality"
            if "nationality" not in target_relations:
                target_relations.append("nationality")

        return StubQueryPlan(
            decomposition=decomposition,
            operator_hint=operator_hint,
            target_relations=target_relations,
        )

    def detect_entities(self, query: str) -> list[str]:
        entities = []
        for candidate in [
            "Albert Einstein",
            "Christopher Nolan",
            "Inception",
            "Scott Derrickson",
            "Ed Wood",
            "Unknown Alias",
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
        if "nationality" in lowered or "citizenship" in lowered:
            match = next((s for s in statements if s.relation == "nationality"), None)
            if match:
                return ReasonerOutcome(
                    answer=match.object,
                    confidence=0.9,
                    reasoning_path=[match],
                    required_facts=[(match.subject, "nationality")],
                    operator_checks=[],
                    summary="fact_lookup",
                )
        return ReasonerOutcome(
            answer="INSUFFICIENT_EVIDENCE",
            confidence=0.0,
            reasoning_path=[],
            required_facts=[],
            operator_checks=[],
            summary="fallback_abstain",
        )


def _build_orchestrator(docs, fact_map, resolver_aliases=None) -> QueryOrchestrator:
    store = ChunkFactStore()
    for chunk_id, statements in fact_map.items():
        store.add_facts(chunk_id, statements)

    entity_resolver = EntityResolver(aliases=resolver_aliases or {})
    return QueryOrchestrator(
        graph_engine=None,
        router=StubRouter(),
        retriever=EvidenceRetriever(StubRAGRetriever(docs), store),
        assembler=EvidenceAssembler(),
        reasoner=StubReasoner(),
        entity_resolver=entity_resolver,
    )


def test_sprint6_alias_resolution_us_citizenship():
    docs = [
        {
            "doc_id": "d1::c0",
            "text": "Albert Einstein citizenship United States",
            "score": 0.9,
            "source": "d1",
        },
    ]
    facts = {
        "d1::c0": [
            Statement("Albert Einstein", "nationality", "United States", trust_score=0.9)
        ]
    }
    orchestrator = _build_orchestrator(
        docs,
        facts,
        resolver_aliases={"US": "United States", "U.S.": "United States", "USA": "United States"},
    )

    result = orchestrator.answer("What is Albert Einstein's citizenship in the U.S.?")

    assert any(e.canonical_name == "United States" for e in orchestrator.last_context.resolved_entities)
    assert any(f.statement.relation == "nationality" for f in result.evidence_facts)
    assert result.answer in {"United States", "INSUFFICIENT_EVIDENCE"}


def test_sprint6_canonical_alias_director_lookup():
    docs = [
        {
            "doc_id": "d2::c0",
            "text": "Christopher Nolan directed Inception",
            "score": 0.95,
            "source": "d2",
        },
    ]
    facts = {
        "d2::c0": [Statement("Christopher Nolan", "directed", "Inception", trust_score=0.9)]
    }
    orchestrator = _build_orchestrator(
        docs,
        facts,
        resolver_aliases={"Chris Nolan": "Christopher Nolan"},
    )

    result = orchestrator.answer("Did Chris Nolan direct Inception?")

    assert any(e.canonical_name == "Christopher Nolan" for e in orchestrator.last_context.resolved_entities)
    assert result.answer in {"yes", "INSUFFICIENT_EVIDENCE"}


def test_sprint6_relation_targeted_retrieval_prioritizes_nationality():
    docs = [
        {"doc_id": "d3::c0", "text": "Albert Einstein nationality German", "score": 0.7, "source": "d3"},
        {"doc_id": "d3::c1", "text": "Albert Einstein profession Physicist", "score": 0.9, "source": "d3"},
        {"doc_id": "d3::c2", "text": "Albert Einstein born_in Ulm", "score": 0.8, "source": "d3"},
    ]
    facts = {
        "d3::c0": [Statement("Albert Einstein", "nationality", "German")],
        "d3::c1": [Statement("Albert Einstein", "profession", "Physicist")],
        "d3::c2": [Statement("Albert Einstein", "born_in", "Ulm")],
    }
    orchestrator = _build_orchestrator(docs, facts)

    result = orchestrator.answer("What is the nationality of Albert Einstein?")

    top_relations = [f.statement.relation for f in result.evidence_facts[:2]]
    assert "nationality" in top_relations
    assert result.answer in {"German", "INSUFFICIENT_EVIDENCE"}


def test_sprint6_dual_entity_comparison_retains_both_sides():
    docs = [
        {"doc_id": "d4::c0", "text": "Scott Derrickson nationality American", "score": 0.9, "source": "d4"},
        {"doc_id": "d4::c1", "text": "Scott Derrickson profession Director", "score": 0.88, "source": "d4"},
        {"doc_id": "d4::c2", "text": "Ed Wood nationality American", "score": 0.87, "source": "d4"},
        {"doc_id": "d4::c3", "text": "Ed Wood profession Director", "score": 0.86, "source": "d4"},
    ]
    facts = {
        "d4::c0": [Statement("Scott Derrickson", "nationality", "American")],
        "d4::c1": [Statement("Scott Derrickson", "profession", "Director")],
        "d4::c2": [Statement("Ed Wood", "nationality", "American")],
        "d4::c3": [Statement("Ed Wood", "profession", "Director")],
    }
    orchestrator = _build_orchestrator(docs, facts)

    result = orchestrator.answer("Were Scott Derrickson and Ed Wood of the same nationality?")

    subjects = {f.statement.subject for f in result.evidence_facts}
    assert "Scott Derrickson" in subjects and "Ed Wood" in subjects
    assert any(f.statement.relation == "nationality" for f in result.evidence_facts)


def test_sprint6_missing_alias_fallback_abstains_cleanly():
    docs = []
    facts = {}
    orchestrator = _build_orchestrator(
        docs,
        facts,
        resolver_aliases={"Known Alias": "Known Entity"},
    )

    result = orchestrator.answer("What is the nationality of Unknown Alias?")

    assert orchestrator.last_context is not None
    assert all(e.canonical_name for e in orchestrator.last_context.resolved_entities)
    assert result.answer == "INSUFFICIENT_EVIDENCE"
    assert result.failure_mode == "no_evidence"
