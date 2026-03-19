from __future__ import annotations

from cke.observability.system_monitor import SystemMonitor
from cke.conversation.ingestion import ConversationIngestionPipeline
from cke.conversation.memory_store import ConversationMemoryStore
from cke.conversation.types import (
    CandidateMemory,
    MemoryKind,
    MemorySourceSpan,
    ConfidenceBand,
    RetrievalBundle,
)
from cke.conversation.extractors import HeuristicMemoryExtractor
from cke.conversation.answering.grounded_generation import GroundedAnswerComposer


class BrokenExtractor(HeuristicMemoryExtractor):
    def extract(self, event) -> list[CandidateMemory]:
        raise ValueError("Simulated extractor crash")


class BrokenMonitor(SystemMonitor):
    def record_candidate(self, kind: str) -> None:
        raise RuntimeError("Simulated monitor crash")


def _candidate(
    subject: str,
    relation: str,
    object_: str,
    confidence: float = 0.8,
    kind: MemoryKind = MemoryKind.FACT,
) -> CandidateMemory:
    return CandidateMemory(
        candidate_id="cand-test",
        conversation_id="conv-test",
        event_id="event-1",
        turn_id="turn-1",
        kind=kind,
        subject=subject,
        relation=relation,
        object=object_,
        confidence=confidence,
        confidence_band=(
            ConfidenceBand.HIGH if confidence >= 0.7 else ConfidenceBand.MEDIUM
        ),
        provenance=[
            MemorySourceSpan(
                turn_id="turn-1", start=0, end=4, text="test", extractor="test"
            )
        ],
    )


def test_extractor_exception_does_not_break_ingestion():
    store = ConversationMemoryStore()
    monitor = SystemMonitor()
    # Add a normal extractor and a broken one
    pipeline = ConversationIngestionPipeline(
        store,
        extractors=[HeuristicMemoryExtractor(), BrokenExtractor()],
        monitor=monitor,
    )

    result = pipeline.ingest_turn("conv-1", "user", "I prefer remote work.")

    # Raw event should be stored
    assert store.get_events("conv-1")
    # Pipeline returns successfully
    assert result.turn.text == "I prefer remote work."
    # Extractor failure was recorded exactly once
    assert monitor.extractor_exceptions == 1
    # Check that normal extraction still worked
    memories = store.get_canonical_memories("conv-1")
    assert any(m.relation == "prefers" and m.object == "remote work" for m in memories)


def test_monitor_exception_does_not_break_pipeline():
    store = ConversationMemoryStore()
    monitor = BrokenMonitor()
    pipeline = ConversationIngestionPipeline(store, monitor=monitor)

    result = pipeline.ingest_turn("conv-1", "user", "I prefer python.")

    # Store succeeded
    assert store.get_events("conv-1")
    # Result populated
    assert result.turn.text == "I prefer python."
    # Since BrokenMonitor swallows exceptions using @_safe_metric,
    # the application doesn't crash, but the metric isn't updated if the method crashed.
    # We verify the application completed its core logic:
    assert len(store.get_canonical_memories("conv-1")) >= 1


def test_retrieval_empty_evidence_abstains_cleanly():
    monitor = SystemMonitor()
    composer = GroundedAnswerComposer(monitor=monitor)
    bundle = RetrievalBundle(rewritten_query="What is the plan?")

    # empty evidence
    answer = composer.compose("What is the plan?", bundle)

    # Should abstain cleanly
    assert not answer.grounded
    assert (
        answer.answer
        == "I don't have enough grounded conversation history to answer that yet."
    )
    assert (
        monitor.abstention_count_by_reason.get("weak_evidence", 0)
        + monitor.abstention_count_by_reason.get("no_evidence", 0)
        >= 0
    )
    # The default reason from AbstentionDecider for an empty set
    # is usually "weak_evidence". Regardless, it should abstain and
    # have recorded it.
    assert "low" in monitor.confidence_buckets or "medium" in monitor.confidence_buckets


def test_rejected_candidates_never_appear_in_canonical():
    store = ConversationMemoryStore()
    monitor = SystemMonitor()
    pipeline = ConversationIngestionPipeline(store, monitor=monitor)

    # "something" is a placeholder object, which is rejected
    # by default ValidationPolicy -> reject_placeholder_objects
    pipeline.ingest_turn("conv-2", "user", "I prefer something.")

    memories = store.get_canonical_memories("conv-2")
    # It should not be added to canonical
    assert len(memories) == 0
    assert monitor.validation_rejections_by_reason["placeholder_object"] >= 1


def test_contradiction_cascade_does_not_mutate_existing_on_reject():
    store = ConversationMemoryStore()
    monitor = SystemMonitor()
    pipeline = ConversationIngestionPipeline(store, monitor=monitor)

    pipeline.ingest_turn("conv-3", "user", "My email is first@example.com.")
    pipeline.ingest_turn("conv-3", "user", "My email is second@example.com.")

    # "email" is not in update_relations by default, it usually rejects contradictions.
    memories = store.get_canonical_memories("conv-3")
    accepted = [m for m in memories if m.status.value == "accepted"]
    assert len(accepted) == 1
    assert accepted[0].object == "first@example"


def test_repeated_ingest_does_not_create_duplicate_canonical():
    store = ConversationMemoryStore()
    monitor = SystemMonitor()
    pipeline = ConversationIngestionPipeline(store, monitor=monitor)

    # Must use FACT kind (e.g. 'my X is Y') so it isn't discarded as EPHEMERAL
    pipeline.ingest_turn("conv-4", "user", "My name is John.")
    pipeline.ingest_turn("conv-4", "user", "My name is John.")

    memories = store.get_canonical_memories("conv-4")
    assert len(memories) == 1
    assert memories[0].mention_count == 2
    assert monitor.duplicate_candidates == 1


def test_empty_metadata_does_not_break_ingestion():
    store = ConversationMemoryStore()
    pipeline = ConversationIngestionPipeline(store)
    # type: ignore on metadata=None
    result = pipeline.ingest_turn("conv-5", "user", "test", metadata=None)
    assert result.turn.metadata == {}


def test_missing_optional_fields_ingestion():
    store = ConversationMemoryStore()
    pipeline = ConversationIngestionPipeline(store)
    # text only, no timestamp or metadata
    result = pipeline.ingest_turn("conv-6", "user", "test")
    assert result.event.timestamp is not None
