"""API service package.

``create_app`` is imported eagerly; the ASGI ``app`` is built lazily on first
attribute access so that importing this package does not require fastapi.
"""

from cke.api.server import MissingDependencyError, create_app

__all__ = ["MissingDependencyError", "app", "create_app"]


def __getattr__(name: str):
    if name == "app":
        return create_app()
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
