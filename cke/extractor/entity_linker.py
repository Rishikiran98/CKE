"""Deprecated – use :mod:`cke.entity_resolution.entity_resolver`.

This shim preserves backward compatibility for existing imports.
"""

from cke.entity_resolution.entity_resolver import (  # noqa: F401
    EntityResolver,
    ResolutionResult,
)

__all__ = ["EntityResolver", "ResolutionResult"]
