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


def test_a_mention_resolves_the_same_way_every_time_it_is_asked(offline_embedder):
    """Codex raised this on #103, and it was right and understated.

    Every rung of resolve_with_score except containment caches its answer by
    registering the mention as an alias, which makes it a canonical name —
    and the canonical rung reports 0.95, not the confidence of the rung that
    actually did the work. So the same mention scored one number the first
    time a run met it and 0.95 on every repeat. Entity confidence is averaged
    into entity_resolution_confidence and weighted into the confidence the
    pipeline reports, so a benchmark scored the same entity differently
    depending on which item it happened to appear in.

    Codex proposed passing the graph engine into the orchestrator's default
    resolver. That does not fix it: it changes which two values alternate.
    Measured before this change, on "What is the citizenship of Albert
    Einstein?" asked three times:

        without a graph engine   0.50, 0.95, 0.95
        with a graph engine      1.00, 0.95, 0.95
    """
    from cke.entity_resolution.entity_resolver import EntityResolver
    from cke.graph_engine.graph_engine import KnowledgeGraphEngine

    engine = KnowledgeGraphEngine()
    engine.add_statement("Albert Einstein", "citizenship", "United States")
    question = "What is the citizenship of Albert Einstein?"

    for label, resolver in (
        ("no graph engine", EntityResolver()),
        ("graph engine", EntityResolver(graph_engine=engine)),
    ):
        scores = []
        for _ in range(3):
            resolved = resolver.resolve_mentions(
                question, candidate_entities=["Albert Einstein"]
            )[0]
            scores.append(resolved.link_confidence)
        assert len(set(scores)) == 1, (
            f"with {label}, the same mention scored {scores} across three "
            f"identical requests, so a reported confidence depends on "
            f"whether the entity was seen earlier in the run"
        )


def test_a_caller_registering_an_alias_invalidates_what_was_cached(
    offline_embedder,
):
    """The cache must not outlive the knowledge it was built without.

    A resolver that has already answered for a mention, and is then told by a
    caller what that mention canonically is, must use what it was told rather
    than serve the answer it worked out earlier.
    """
    from cke.entity_resolution.entity_resolver import EntityResolver
    from cke.graph_engine.graph_engine import KnowledgeGraphEngine

    engine = KnowledgeGraphEngine()
    engine.add_statement("Christopher Nolan", "directed", "Inception")
    resolver = EntityResolver(graph_engine=engine)

    # A misspelling, so the fuzzy rung resolves it without an embedding
    # model: "Chris Nolan" reaches "Christopher Nolan" only through
    # embeddings, which would make this a test about the encoder.
    mention = "Christopher Nolen"
    assert resolver.resolve_with_score(mention).canonical == "Christopher Nolan"

    resolver.register_alias(mention, "Someone Else Entirely")

    assert resolver.resolve_with_score(mention).canonical == "Someone Else Entirely"


def test_a_caller_can_correct_a_name_the_resolver_guessed_at(offline_embedder):
    """The case #106's invalidation test had to route around.

    The unresolvable rungs answer with the mention's own title case and
    register that, which made every mention nobody could place a canonical
    entity in its own right. The exact-canonical rung is checked before the
    alias registry, so once a mention had been guessed at, a caller saying
    what it actually meant could not change the answer — #106 had to test
    invalidation with a misspelling the fuzzy rung resolved elsewhere,
    because this obvious case did not work.
    """
    from cke.entity_resolution.entity_resolver import EntityResolver

    resolver = EntityResolver()

    guessed = resolver.resolve_with_score("Chris Nolan")
    assert guessed.canonical == "Chris Nolan"

    resolver.register_alias("Chris Nolan", "Christopher Nolan")

    corrected = resolver.resolve_with_score("Chris Nolan")
    assert corrected.canonical == "Christopher Nolan"
    assert corrected.confidence == 0.90, (
        "resolving through the alias registry must be scored as such, not as "
        "the exact-canonical match the guess had turned it into"
    )


def test_a_guessed_name_is_not_reported_as_a_known_entity(offline_embedder):
    """known_entities() is what a caller reads to see what the graph holds."""
    from cke.entity_resolution.entity_resolver import EntityResolver

    resolver = EntityResolver()
    resolver.resolve_with_score("Some Mention Nobody Can Place")
    resolver.register_alias("Some Mention Nobody Can Place", "A Real Entity")

    known = list(resolver.known_entities())

    assert "A Real Entity" in known
    assert "Some Mention Nobody Can Place" not in known


def test_clustering_still_folds_a_variant_onto_the_name_it_matched(offline_embedder):
    """Registering a guess is what lets later mentions cluster onto it.

    Not removing that: an unplaceable mention still becomes something later
    variants can match against. What changed is only that a name registered
    as an alias for something else stops being canonical in its own right.
    """
    from cke.entity_resolution.entity_resolver import EntityResolver

    resolver = EntityResolver()
    canonical = resolver.resolve_entity("Redis")

    assert resolver.resolve_entity("Redis DB") == canonical
    assert resolver.resolve_entity("Redis database") == canonical
