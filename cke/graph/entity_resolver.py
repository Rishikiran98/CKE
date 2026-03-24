"""Deprecated – use :mod:`cke.entity_resolution.entity_resolver`.

This shim preserves backward compatibility for existing imports.
"""

from cke.entity_resolution.entity_resolver import EntityResolver  # noqa: F401

__all__ = ["EntityResolver"]
