"""Alias resolution and relation-targeted retrieval through the orchestrator."""

from cke.entity_resolution.entity_resolver import EntityResolver
from cke.models import Statement
from cke.pipeline.evidence_assembler import EvidenceAssembler
from cke.pipeline.query_orchestrator import QueryOrchestrator
from cke.pipeline.types import ReasonerOutcome
from cke.retrieval.chunk_fact_store import ChunkFactStore
from cke.retrieval.evidence_retriever import EvidenceRetriever
from cke.router.query_router import QueryRouter


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
        router=QueryRouter(),
        retriever=EvidenceRetriever(StubRAGRetriever(docs), store),
        assembler=EvidenceAssembler(),
        reasoner=StubReasoner(),
        entity_resolver=entity_resolver,
    )


def test_alias_resolution_us_citizenship():
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
            Statement(
                "Albert Einstein", "nationality", "United States", trust_score=0.9
            )
        ]
    }
    orchestrator = _build_orchestrator(
        docs,
        facts,
        resolver_aliases={
            "US": "United States",
            "U.S.": "United States",
            "USA": "United States",
        },
    )

    result = orchestrator.answer("What is Albert Einstein's citizenship in the U.S.?")

    assert any(
        e.canonical_name == "United States"
        for e in orchestrator.last_context.resolved_entities
    )
    assert any(f.statement.relation == "nationality" for f in result.evidence_facts)
    assert result.answer == "United States"


def test_canonical_alias_director_lookup():
    docs = [
        {
            "doc_id": "d2::c0",
            "text": "Christopher Nolan directed Inception",
            "score": 0.95,
            "source": "d2",
        },
    ]
    facts = {
        "d2::c0": [
            Statement("Christopher Nolan", "directed", "Inception", trust_score=0.9)
        ]
    }
    orchestrator = _build_orchestrator(
        docs,
        facts,
        resolver_aliases={"Chris Nolan": "Christopher Nolan"},
    )

    result = orchestrator.answer("Did Chris Nolan direct Inception?")

    assert any(
        e.canonical_name == "Christopher Nolan"
        for e in orchestrator.last_context.resolved_entities
    )
    # Was `in {"yes", "INSUFFICIENT_EVIDENCE"}`, which accepted the right
    # answer and a refusal as equally correct, so a regression from answering
    # to abstaining was invisible.
    assert result.answer == "yes"
    assert [e.canonical_name for e in orchestrator.last_context.resolved_entities] == [
        "Christopher Nolan",
        "Inception",
    ]


def test_relation_targeted_retrieval_prioritizes_nationality():
    docs = [
        {
            "doc_id": "d3::c0",
            "text": "Albert Einstein nationality German",
            "score": 0.7,
            "source": "d3",
        },
        {
            "doc_id": "d3::c1",
            "text": "Albert Einstein profession Physicist",
            "score": 0.9,
            "source": "d3",
        },
        {
            "doc_id": "d3::c2",
            "text": "Albert Einstein born_in Ulm",
            "score": 0.8,
            "source": "d3",
        },
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
    assert result.answer == "German"


def test_dual_entity_comparison_retains_both_sides():
    docs = [
        {
            "doc_id": "d4::c0",
            "text": "Scott Derrickson nationality American",
            "score": 0.9,
            "source": "d4",
        },
        {
            "doc_id": "d4::c1",
            "text": "Scott Derrickson profession Director",
            "score": 0.88,
            "source": "d4",
        },
        {
            "doc_id": "d4::c2",
            "text": "Ed Wood nationality American",
            "score": 0.87,
            "source": "d4",
        },
        {
            "doc_id": "d4::c3",
            "text": "Ed Wood profession Director",
            "score": 0.86,
            "source": "d4",
        },
    ]
    facts = {
        "d4::c0": [Statement("Scott Derrickson", "nationality", "American")],
        "d4::c1": [Statement("Scott Derrickson", "profession", "Director")],
        "d4::c2": [Statement("Ed Wood", "nationality", "American")],
        "d4::c3": [Statement("Ed Wood", "profession", "Director")],
    }
    orchestrator = _build_orchestrator(docs, facts)

    result = orchestrator.answer(
        "Were Scott Derrickson and Ed Wood of the same nationality?"
    )

    subjects = {f.statement.subject for f in result.evidence_facts}
    assert "Scott Derrickson" in subjects and "Ed Wood" in subjects
    assert any(f.statement.relation == "nationality" for f in result.evidence_facts)


def test_missing_alias_fallback_abstains_cleanly():
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


def test_the_entity_confidence_is_not_manufactured_by_the_orchestrator():
    """0.95 for every detected entity was a round trip, not a measurement.

    The orchestrator registered each surface form the router detected as its
    own canonical name, then asked the resolver whether it was a canonical
    name. resolve_with_score scores an exact canonical match 0.95, so every
    detected entity scored 0.95 whatever the resolver actually knew about it,
    and that figure is averaged into entity_resolution_confidence and weighted
    into the confidence the pipeline reports.

    Now an entity the resolver can place scores higher than one it cannot.
    """
    from cke.entity_resolution.entity_resolver import EntityResolver

    resolver = EntityResolver(aliases={"Chris Nolan": "Christopher Nolan"})

    resolved = {
        entity.surface_form: entity
        for entity in resolver.resolve_mentions(
            "Did Chris Nolan direct Inception?",
            candidate_entities=["Chris Nolan", "Inception"],
        )
    }

    known = resolved["Chris Nolan"]
    unknown = resolved["Inception"]

    assert known.canonical_name == "Christopher Nolan"
    assert unknown.canonical_name == "Inception"
    assert known.link_confidence > unknown.link_confidence, (
        "an entity resolved through the alias registry must not score the "
        "same as one the resolver knows nothing about"
    )
