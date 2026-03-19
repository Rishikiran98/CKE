"""Graph-backed evidence expansion from retrieved memories."""

from __future__ import annotations

from cke.models import Statement


class GraphExpander:
    """Expand supporting graph facts from entity anchors."""

    def __init__(self, graph_engine) -> None:
        self.graph_engine = graph_engine

    def expand(self, anchors: list[str], *, limit: int = 5) -> list[Statement]:
        seen: dict[tuple[str, str, str], Statement] = {}
        for anchor in anchors:
            for statement in self.graph_engine.get_neighbors(anchor):
                seen.setdefault(statement.key(), statement)
                if len(seen) >= limit:
                    return list(seen.values())
        return list(seen.values())
