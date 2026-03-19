"""Evidence selection over retrieval bundles."""

from __future__ import annotations

from cke.conversation.types import EvidenceSet, RetrievalBundle


class EvidenceSelector:
    """Select compact supporting evidence for downstream answer generation."""

    def select(self, query: str, bundle: RetrievalBundle) -> EvidenceSet:
        turns = bundle.retrieved_turns[:3]
        memories = bundle.retrieved_memories[:5]
        facts = bundle.retrieved_facts[:5]
        evidence = EvidenceSet(
            query=query,
            rewritten_query=bundle.rewritten_query,
            supporting_turns=turns,
            supporting_memories=memories,
            supporting_facts=facts,
            graph_facts=bundle.graph_neighbors[:5],
            summaries=bundle.summaries[:3],
            metadata=dict(bundle.metadata),
        )
        bundle.evidence = evidence
        return evidence
