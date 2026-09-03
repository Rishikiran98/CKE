"""FastAPI service for querying and ingesting graph assertions."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field

from cke.graph_engine.graph_engine import KnowledgeGraphEngine

try:
    from fastapi import FastAPI
except ImportError:  # pragma: no cover - optional runtime dependency
    FastAPI = None


class MissingDependencyError(RuntimeError):
    """Raised when an optional dependency needed for this feature is absent."""


class QueryRequest(BaseModel):
    question: str
    max_depth: int = 3


class IngestRequest(BaseModel):
    assertions: list[dict[str, Any]] = Field(default_factory=list)


def create_app(graph_engine: KnowledgeGraphEngine | None = None):
    """Build the FastAPI application.

    Raises if fastapi is not installed. Importing this module does not build
    the app, so ``cke.api.server`` can be imported and inspected in an
    environment without fastapi.
    """
    if FastAPI is None:
        raise MissingDependencyError(
            "fastapi is required to build the CKE API application. "
            "Install it with `pip install fastapi` (and `uvicorn` to serve)."
        )

    app = FastAPI(title="CKE API", version="0.1.0")
    engine = graph_engine or KnowledgeGraphEngine()

    @app.get("/health")
    def health() -> dict[str, str]:
        return {"status": "ok"}

    @app.post("/ingest")
    def ingest(payload: IngestRequest) -> dict[str, int]:
        for assertion in payload.assertions:
            engine.add_statement(
                assertion["subject"],
                assertion["relation"],
                assertion["object"],
                context=assertion.get("context"),
                confidence=float(assertion.get("confidence", 1.0)),
                source=assertion.get("source"),
                timestamp=assertion.get("timestamp"),
            )
        return {"assertions_added": len(payload.assertions)}

    @app.post("/query")
    def query(payload: QueryRequest) -> dict[str, Any]:
        tokens = payload.question.split()
        entities = [tok.strip("?.!,") for tok in tokens if tok[:1].isupper()]
        evidence = []
        for entity in entities:
            for edge in engine.get_neighbors(entity):
                evidence.append(
                    {
                        "subject": edge.subject,
                        "relation": edge.relation,
                        "object": edge.object,
                        "trust_score": edge.confidence,
                    }
                )

        answer = evidence[0]["object"] if evidence else "No answer found"
        confidence = evidence[0].get("trust_score", 0.0) if evidence else 0.0
        return {
            "answer": answer,
            "evidence": evidence[: max(1, payload.max_depth * 2)],
            "confidence": confidence,
        }

    return app


#: The module-level ASGI application, built on first access to ``app``.
#: Cached so that repeated access returns the same object, as it did when the
#: module built it eagerly: routes, middleware and state registered on one
#: access must still be there on the next.
_APP = None


def get_app():
    """Return the module-level ASGI application, building it once."""
    global _APP
    if _APP is None:
        _APP = create_app()
    return _APP


def reset_app() -> None:
    """Discard the cached application. For tests."""
    global _APP
    _APP = None


def __getattr__(name: str):
    """Build the ASGI app on first attribute access, not at import time.

    Keeps ``uvicorn cke.api.server:app`` working while leaving the module
    importable without fastapi installed (PEP 562).
    """
    if name == "app":
        return get_app()
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "IngestRequest",
    "MissingDependencyError",
    "QueryRequest",
    "create_app",
    "get_app",
    "reset_app",
]
