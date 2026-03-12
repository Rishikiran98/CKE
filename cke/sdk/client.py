"""Simple Python SDK for interacting with CKE API service."""

from __future__ import annotations

import json
from dataclasses import dataclass
from urllib import request


@dataclass
class QueryResponse:
    answer: str
    evidence: list[dict]
    confidence: float


class CKEClient:
    def __init__(self, base_url: str) -> None:
        self.base_url = base_url.rstrip("/")

    def _post(self, path: str, payload: dict) -> dict:
        req = request.Request(
            url=f"{self.base_url}{path}",
            method="POST",
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
        )
        with request.urlopen(req) as resp:  # noqa: S310 - user-provided URL by design
            return json.loads(resp.read().decode("utf-8"))

    def _get(self, path: str) -> dict:
        with request.urlopen(f"{self.base_url}{path}") as resp:  # noqa: S310
            return json.loads(resp.read().decode("utf-8"))

    def query(self, question: str, max_depth: int = 3) -> QueryResponse:
        raw = self._post("/query", {"question": question, "max_depth": max_depth})
        return QueryResponse(
            answer=str(raw.get("answer", "")),
            evidence=list(raw.get("evidence", [])),
            confidence=float(raw.get("confidence", 0.0)),
        )

    def ingest(self, assertions: list[dict]) -> dict:
        return self._post("/ingest", {"assertions": assertions})

    def health(self) -> dict:
        return self._get("/health")
