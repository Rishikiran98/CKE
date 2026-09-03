"""API service package.

``create_app`` is imported eagerly; the ASGI ``app`` is built lazily on first
attribute access so that importing this package does not require fastapi.
Accessing ``app`` here returns the same object as ``cke.api.server.app``.
"""

from cke.api.server import MissingDependencyError, create_app, get_app, reset_app

# "app" is deliberately absent: a star-import resolves every name in __all__,
# which would build the application and fail without fastapi. Reach it as
# cke.api.app or via get_app().
__all__ = ["MissingDependencyError", "create_app", "get_app", "reset_app"]


def __getattr__(name: str):
    if name == "app":
        return get_app()
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
