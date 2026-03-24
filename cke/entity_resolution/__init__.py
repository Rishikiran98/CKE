"""Entity resolution package exports."""

from cke.entity_resolution.alias_registry import AliasRegistry
from cke.entity_resolution.entity_resolver import EntityResolver, ResolutionResult

__all__ = ["AliasRegistry", "EntityResolver", "ResolutionResult"]
