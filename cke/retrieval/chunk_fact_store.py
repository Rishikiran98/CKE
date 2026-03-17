"""In-memory mapping from chunk IDs to extracted statements."""

from __future__ import annotations

from cke.models import Statement


class ChunkFactStore:
    """Store and retrieve statement lists keyed by chunk IDs."""

    def __init__(self) -> None:
        self.chunk_to_facts: dict[str, list[Statement]] = {}

    def add_facts(self, chunk_id: str, statements: list[Statement]) -> None:
        self.chunk_to_facts[chunk_id] = statements

    def get_facts(self, chunk_id: str) -> list[Statement]:
        return self.chunk_to_facts.get(chunk_id, [])
