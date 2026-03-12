"""Graph helpers and memory interfaces for CKE."""

from cke.graph.entity_resolver import EntityResolver

__all__ = [
    "EntityResolver",
    "AssertionValidator",
]


def __getattr__(name: str):
    """Lazy-load the AssertionValidator to avoid circular imports or overhead."""
    if name == "AssertionValidator":
        from cke.graph.assertion_validator import AssertionValidator
        return AssertionValidator

    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")