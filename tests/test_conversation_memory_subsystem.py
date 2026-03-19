from __future__ import annotations

from cke.conversation.answering.abstention import AbstentionDecider
from cke.conversation.answering.evidence_selection import EvidenceSelector
from cke.conversation.answering.grounded_generation import GroundedAnswerComposer
from cke.conversation.config import ConversationConfig
from cke.conversation.consolidation import MemoryConsolidator
from cke.conversation.ingestion import ConversationIngestionPipeline
from cke.conversation.memory_store import ConversationMemoryStore
from cke.conversation.resolution.reference_resolution import (
    ConversationalReferenceResolver,
)
from cke.conversation.retriever import ConversationalRetriever
from cke.conversation.types import (
    CandidateMemory,
    ConfidenceBand,
    EvidenceSet,
    MemoryKind,
    MemorySourceSpan,
    RetrievalBundle,
)
from cke.conversation.validation import CandidateMemoryValidator


def _candidate(
    *,
    candidate_id: str,
    subject: str,
    relation: str,
    object_: str,
    confidence: float = 0.8,
    kind: MemoryKind = MemoryKind.FACT,
) -> CandidateMemory:
    return CandidateMemory(
        candidate_id=candidate_id,
        conversation_id="conv",
        event_id="event-1",
        turn_id="turn-1",
        kind=kind,
        subject=subject,
        relation=relation,
        object=object_,
        confidence=confidence,
        confidence_band=ConfidenceBand.HIGH if confidence >= 0.7 else ConfidenceBand.MEDIUM,
        provenance=[
            MemorySourceSpan(
                turn_id="turn-1",
                start=0,
                end=max(4, len(object_)),
                text=object_,
                extractor="test",
            )
        ],
    )


def test_ingestion_stores_raw_event_and_promotes_canonical_memory() -> None:
    store = ConversationMemoryStore()
    pipeline = ConversationIngestionPipeline(store)

    result = pipeline.ingest_turn(
        "conv-1",
        "user",
        "I prefer quiet coffee shops. The meetup is on March 14.",
        timestamp="2026-03-19T10:00:00+00:00",
    )

    assert result.event.text.startswith("I prefer quiet")
    assert store.get_events("conv-1")[0].event_id == result.event.event_id
    assert store.get_turns("conv-1")[0].turn_id == result.turn.turn_id
    assert any(memory.relation == "prefers" for memory in result.accepted_memories)
    assert any(memory.relation == "occurs_on" for memory in result.accepted_memories)
    assert store.get_canonical_memories("conv-1")


def test_candidate_validation_rejects_low_signal_placeholder_memory() -> None:
    validator = CandidateMemoryValidator()
    accepted, rejected = validator.validate(
        [
            _candidate(
                candidate_id="bad-1",
                subject="user",
                relation="prefers",
                object_="something",
                confidence=0.2,
                kind=MemoryKind.PREFERENCE,
            )
        ]
    )

    assert not accepted
    assert rejected
    assert {"placeholder_object", "low_signal_confidence"} <= set(
        rejected[0].rejection_reasons
    )


def test_consolidation_dedupes_and_supersedes_conflicting_memories() -> None:
    consolidator = MemoryConsolidator()
    existing = []
    first = _candidate(
        candidate_id="cand-1",
        subject="project alpha",
        relation="status",
        object_="planning",
        kind=MemoryKind.STATUS,
    )
    second = _candidate(
        candidate_id="cand-2",
        subject="project alpha",
        relation="status",
        object_="active",
        kind=MemoryKind.STATUS,
    )

    first_decision = consolidator.consolidate([first], existing, timestamp="2026-03-19T10:00:00+00:00")
    existing = [first_decision[0].canonical_memory]
    second_decision = consolidator.consolidate([second], existing, timestamp="2026-03-19T11:00:00+00:00")

    assert first_decision[0].decision.value == "accept"
    assert second_decision[0].decision.value == "update"
    assert second_decision[0].conflict is not None
    assert existing[0].status.value == "superseded"


def test_retrieval_combines_raw_turns_and_canonical_memories() -> None:
    store = ConversationMemoryStore()
    pipeline = ConversationIngestionPipeline(store)
    pipeline.ingest_turn(
        "conv-2",
        "user",
        "I prefer remote work and async communication.",
        timestamp="2026-03-19T10:00:00+00:00",
    )
    pipeline.ingest_turn(
        "conv-2",
        "user",
        "The architecture review is on March 14.",
        timestamp="2026-03-19T10:05:00+00:00",
    )

    retriever = ConversationalRetriever(store)
    bundle = retriever.retrieve("When is the architecture review?", conversation_id="conv-2")

    assert bundle.retrieved_turns
    assert bundle.retrieved_memories
    assert any(fact.relation == "occurs_on" for fact in bundle.retrieved_facts)


def test_reference_resolution_uses_recent_entities_and_roles() -> None:
    store = ConversationMemoryStore()
    pipeline = ConversationIngestionPipeline(store)
    pipeline.ingest_turn(
        "conv-3",
        "user",
        "The recruiter from Stripe still hasn't replied about the backend engineer role.",
    )

    resolver = ConversationalReferenceResolver(store)
    rewritten, bindings = resolver.rewrite(
        "What did that company say about that role?",
        conversation_id="conv-3",
        retrieved_turns=[],
    )

    assert "Stripe" in rewritten
    assert "backend engineer role" in rewritten.lower()
    assert bindings


def test_abstention_on_weak_or_conflicting_evidence() -> None:
    decider = AbstentionDecider()
    weak = EvidenceSet(query="q", rewritten_query="q")
    conflict = EvidenceSet(
        query="q",
        rewritten_query="q",
        supporting_turns=[],
        supporting_memories=[],
        conflicts=[],
    )
    conflict.conflicts.append(
        type("Conflict", (), {"reason": "mismatch"})()  # lightweight stand-in
    )

    assert decider.should_abstain(weak)[0] is True
    assert decider.should_abstain(conflict)[1] == "conflicting_evidence"


def test_grounded_generation_abstains_when_conflicts_are_present() -> None:
    composer = GroundedAnswerComposer(evidence_selector=EvidenceSelector())
    bundle = RetrievalBundle(rewritten_query="What is the status?")
    evidence = EvidenceSet(query="What is the status?", rewritten_query="What is the status?")
    evidence.conflicts.append(type("Conflict", (), {"reason": "two statuses"})())
    bundle.evidence = evidence

    answer = composer.compose("What is the status?", bundle)

    assert answer.grounded is False
    assert "conflicting evidence" in answer.answer.lower()
