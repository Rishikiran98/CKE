"""FastAPI service for querying and ingesting graph assertions."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field

from cke.graph_engine.graph_engine import KnowledgeGraphEngine

try:
    from fastapi import FastAPI
except Exception:  # pragma: no cover
    FastAPI = None


class QueryRequest(BaseModel):
    question: str
    max_depth: int = 3


class IngestRequest(BaseModel):
    assertions: list[dict[str, Any]] = Field(default_factory=list)


def create_app(graph_engine: KnowledgeGraphEngine | None = None):
    if FastAPI is None:
        raise RuntimeError("fastapi is not installed")

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


app = create_app()
