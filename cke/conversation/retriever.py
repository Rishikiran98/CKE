"""Compatibility wrapper for multi-stage conversational retrieval."""

from __future__ import annotations

from cke.conversation.memory import ConversationalMemoryStore
from cke.conversation.retrieval import (
    CandidateGenerator,
    FactRetriever,
    GraphExpander,
    RetrievalReranker,
    SummaryRetriever,
)
from cke.conversation.resolution import ConversationalReferenceResolver
from cke.conversation.types import RetrievalBundle
from cke.retrieval.embedding_model import EmbeddingModel


class ConversationalRetriever:
    """Retrieve evidence across raw events, canonical memories, summaries, and graph facts."""

    def __init__(
        self,
        memory_store: ConversationalMemoryStore,
        embedding_model: EmbeddingModel | None = None,
        reference_resolver: ConversationalReferenceResolver | None = None,
        config=None,
    ) -> None:
        self.memory_store = memory_store
        self.candidate_generator = CandidateGenerator(
            memory_store,
            embedding_model=embedding_model,
            config=getattr(config, "retrieval", config),
        )
        self.reference_resolver = reference_resolver or ConversationalReferenceResolver(
            memory_store
        )
        self.fact_retriever = FactRetriever(memory_store)
        self.summary_retriever = SummaryRetriever(memory_store)
        self.graph_expander = GraphExpander(memory_store.graph_engine)
        self.reranker = RetrievalReranker()
        self.config = (
            getattr(memory_store, "config", None).retrieval
            if getattr(memory_store, "config", None)
            else None
        )

    def retrieve(
        self, query: str, *, conversation_id: str, top_k: int = 5
    ) -> RetrievalBundle:
        seed = self.candidate_generator.generate(query, conversation_id=conversation_id)
        seed_turns = [item for item in seed if item.memory_type == "event"][:top_k]
        rewritten_query, bindings = self.reference_resolver.rewrite(
            query, conversation_id=conversation_id, retrieved_turns=seed_turns
        )
        reranked = self.reranker.rerank(
            self.candidate_generator.generate(
                rewritten_query, conversation_id=conversation_id
            ),
            top_k=max(top_k * 2, top_k),
        )
        retrieved_turns = [item for item in reranked if item.memory_type == "event"][
            :top_k
        ]
        memory_ids = [
            item.memory_id
            for item in reranked
            if item.memory_type == "canonical_memory"
        ]
        retrieved_memories = [
            memory
            for memory in self.memory_store.get_canonical_memories(conversation_id)
            if memory.memory_id in set(memory_ids)
        ][:top_k]
        all_memories, facts = self.fact_retriever.retrieve(
            conversation_id, limit=max(top_k * 2, top_k)
        )
        subjects = [memory.subject for memory in retrieved_memories] + [
            memory.object
            for memory in retrieved_memories
            if memory.object[:1].isupper()
        ]
        graph_neighbors = self.graph_expander.expand(subjects, limit=top_k)
        summaries = self.summary_retriever.retrieve(conversation_id)
        return RetrievalBundle(
            rewritten_query=rewritten_query,
            raw_candidates=reranked,
            retrieved_turns=retrieved_turns,
            retrieved_memories=retrieved_memories or all_memories[:top_k],
            retrieved_facts=facts[:top_k],
            graph_neighbors=graph_neighbors,
            summaries=summaries,
            metadata={"reference_bindings": bindings},
        )
