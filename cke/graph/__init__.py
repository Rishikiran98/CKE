"""Graph helpers and memory interfaces for CKE."""

from cke.graph.entity_resolver import EntityResolver

__all__ = [
    "EntityResolver",
    "AssertionValidator",
    "GraphStore",
    "GraphAssertion",
    "GraphEntity",
]


def __getattr__(name: str):
    if name == "AssertionValidator":
        from cke.graph.assertion_validator import AssertionValidator

        return AssertionValidator
    if name in {"GraphStore", "GraphAssertion", "GraphEntity"}:
        from cke.graph.graph_store import GraphAssertion, GraphEntity, GraphStore
        return {"GraphStore": GraphStore, "GraphAssertion": GraphAssertion, "GraphEntity": GraphEntity}[name]

    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
