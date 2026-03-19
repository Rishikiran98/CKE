from __future__ import annotations

import numpy as np

from cke.conversation.memory import ConversationalMemoryStore
from cke.conversation.retriever import ConversationalRetriever
from cke.evaluation.conversation_cases import get_conversation_scenarios
from cke.pipeline.conversational_orchestrator import ConversationalOrchestrator


def _run_scenario(scenario):
    orchestrator = ConversationalOrchestrator()
    conversation_id = scenario.scenario_id
    for role, text in scenario.turns:
        orchestrator.ingest_turn(conversation_id, role, text)
    answer = orchestrator.answer(conversation_id, scenario.query)
    return orchestrator, answer


def test_conversation_scenarios_produce_expected_fragments() -> None:
    for scenario in get_conversation_scenarios():
        _, answer = _run_scenario(scenario)
        rendered = answer.answer.lower()
        for fragment in scenario.expected_fragments:
            assert fragment.lower() in rendered, scenario.scenario_id


def test_ingestion_preserves_raw_turns_entities_and_facts() -> None:
    orchestrator = ConversationalOrchestrator()
    orchestrator.ingest_turn(
        "memory-shape",
        "user",
        "I have an Apple interview next Tuesday for a backend engineer role.",
        timestamp="2026-03-18T10:00:00+00:00",
    )
    turns = orchestrator.memory_store.get_turns("memory-shape")
    assert len(turns) == 1
    turn = turns[0]
    assert (
        turn.text
        == "I have an Apple interview next Tuesday for a backend engineer role."
    )
    assert turn.role == "user"
    assert turn.turn_order == 1
    assert "Apple" in turn.entities
    relations = {fact.relation for fact in turn.facts}
    assert {"mentioned_interview", "scheduled_for", "has_role"} <= relations
    assert all(fact.context["turn_id"] == turn.turn_id for fact in turn.facts)


def test_reference_resolution_handles_follow_up_that_company() -> None:
    orchestrator = ConversationalOrchestrator()
    conversation_id = "reference-company"
    orchestrator.ingest_turn(
        conversation_id,
        "user",
        (
            "The recruiter from Stripe still hasn't replied about the "
            "backend engineer role."
        ),
    )
    bundle = orchestrator.retriever.retrieve(
        "What did that company say about that role?", conversation_id=conversation_id
    )
    assert "Stripe" in bundle.rewritten_query
    assert "backend engineer role" in bundle.rewritten_query.lower()


def test_semantic_retrieval_uses_raw_turn_text_not_only_facts() -> None:
    orchestrator = ConversationalOrchestrator()
    conversation_id = "semantic-raw-text"
    orchestrator.ingest_turn(
        conversation_id,
        "user",
        "I bombed the Apple interview because the systems design round was rough.",
    )
    answer = orchestrator.answer(
        conversation_id, "What did I tell you about my Apple interview?"
    )
    assert "systems design" in answer.answer.lower()
    assert answer.grounded is True


def test_abstains_only_when_retrieval_is_weak() -> None:
    orchestrator = ConversationalOrchestrator()
    conversation_id = "abstention"
    orchestrator.ingest_turn(
        conversation_id,
        "user",
        "I talked to one recruiter last week.",
    )
    answer = orchestrator.answer(conversation_id, "What salary did they offer?")
    assert answer.grounded is False
    assert (
        "can't answer that confidently" in answer.answer.lower()
        or "don't have enough" in answer.answer.lower()
    )


def test_ingestion_normalizes_facts_before_graph_insert() -> None:
    orchestrator = ConversationalOrchestrator()
    conversation_id = "normalized-facts"
    orchestrator.ingest_turn(
        conversation_id,
        "user",
        "I prefer   remote work.",
    )

    facts = orchestrator.memory_store.facts_for_conversation(conversation_id)
    preference = next(fact for fact in facts if fact.relation == "prefers")
    assert preference.subject == "user"
    assert preference.object == "remote work"


class CountingEmbeddingModel:
    def __init__(self) -> None:
        self.embed_text_calls = 0
        self.embed_texts_calls = 0

    def embed_text(self, text: str) -> np.ndarray:
        self.embed_text_calls += 1
        return np.asarray([float(len(text)), 1.0], dtype=np.float32)

    def embed_texts(self, texts, batch_size: int = 32) -> np.ndarray:  # noqa: ARG002
        self.embed_texts_calls += 1
        return np.asarray(
            [[float(len(text)), 1.0] for text in texts],
            dtype=np.float32,
        )


def test_retriever_caches_turn_embeddings_between_queries() -> None:
    memory_store = ConversationalMemoryStore()
    memory_store.ingest_turn(
        "cached-retrieval",
        "user",
        "I have an Apple interview next Tuesday.",
    )
    embedding_model = CountingEmbeddingModel()
    retriever = ConversationalRetriever(
        memory_store=memory_store,
        embedding_model=embedding_model,
    )

    retriever.retrieve(
        "What did I say about Apple?", conversation_id="cached-retrieval"
    )
    retriever.retrieve("When is it?", conversation_id="cached-retrieval")

    assert embedding_model.embed_texts_calls == 1
    assert embedding_model.embed_text_calls == 4
