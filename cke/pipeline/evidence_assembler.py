"""Lightweight assembly of retrieved evidence facts."""

from __future__ import annotations

from cke.pipeline.types import EvidenceFact, RetrievedChunk


class EvidenceAssembler:
    """Deduplicate and bound retrieved evidence facts."""

    def __init__(self, max_facts: int = 20) -> None:
        self.max_facts = max_facts

    def assemble(
        self,
        query: str,
        chunks: list[RetrievedChunk],
        facts: list[EvidenceFact],
    ) -> list[EvidenceFact]:
        del query, chunks

        seen: set[tuple[str, str, str, str]] = set()
        deduped: list[EvidenceFact] = []
        for fact in facts:
            key = (
                fact.chunk_id,
                fact.statement.subject,
                fact.statement.relation,
                fact.statement.object,
            )
            if key in seen:
                continue
            seen.add(key)
            deduped.append(fact)
            if len(deduped) >= self.max_facts:
                break
        return deduped
