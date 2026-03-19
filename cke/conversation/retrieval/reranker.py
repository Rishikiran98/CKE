"""Simple deterministic reranking for conversational evidence."""

from __future__ import annotations

from cke.conversation.types import RetrievedMemory


class RetrievalReranker:
    """Rerank mixed candidates with lightweight boosts for canonical evidence."""

    def rerank(
        self, candidates: list[RetrievedMemory], *, top_k: int = 5
    ) -> list[RetrievedMemory]:
        for candidate in candidates:
            if candidate.memory_type == "canonical_memory":
                candidate.score += 0.03
        return sorted(candidates, key=lambda item: item.score, reverse=True)[:top_k]
