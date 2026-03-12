"""Hybrid evidence composition for graph + dense retrieval."""

from __future__ import annotations

from dataclasses import dataclass, field

from cke.models import Statement

GRAPH_WEIGHT = 0.8
DENSE_WEIGHT = 0.2


@dataclass(slots=True)
class EvidencePack:
    """Combined evidence payload returned by hybrid retrieval."""

    graph_statements: list[Statement] = field(default_factory=list)
    fallback_chunks: list[str] = field(default_factory=list)


@dataclass(slots=True)
class WeightedEvidence:
    """Weighted evidence used for downstream ranking/inspection."""

    content: str
    source: str
    weight: float


class HybridRetrievalMerger:
    """Merge graph statements and dense chunks with duplicate removal."""

    def merge(
        self,
        graph_statements: list[Statement],
        fallback_chunks: list[str],
    ) -> tuple[EvidencePack, list[WeightedEvidence]]:
        deduped_graph = self._dedupe_graph(graph_statements)
        deduped_chunks = self._dedupe_chunks(fallback_chunks, deduped_graph)

        pack = EvidencePack(
            graph_statements=deduped_graph,
            fallback_chunks=deduped_chunks,
        )
        weighted = [
            WeightedEvidence(
                content=statement.as_text(),
                source="graph",
                weight=GRAPH_WEIGHT,
            )
            for statement in deduped_graph
        ]
        weighted.extend(
            WeightedEvidence(content=chunk, source="dense", weight=DENSE_WEIGHT)
            for chunk in deduped_chunks
        )
        return pack, weighted

    def _dedupe_graph(self, statements: list[Statement]) -> list[Statement]:
        seen: set[tuple[str, str, str]] = set()
        deduped: list[Statement] = []
        for statement in statements:
            key = statement.key()
            if key in seen:
                continue
            seen.add(key)
            deduped.append(statement)
        return deduped

    def _dedupe_chunks(
        self,
        chunks: list[str],
        graph_statements: list[Statement],
    ) -> list[str]:
        seen = {self._normalize(statement.as_text()) for statement in graph_statements}
        deduped: list[str] = []
        for chunk in chunks:
            token = self._normalize(chunk)
            if token in seen:
                continue
            seen.add(token)
            deduped.append(chunk)
        return deduped

    def _normalize(self, value: str) -> str:
        return " ".join(value.lower().split())
