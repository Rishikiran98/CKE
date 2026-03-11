"""Assertion and evidence models used by graph update workflows."""

from __future__ import annotations

import hashlib
import json
import time
from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class Entity(BaseModel):
    """Canonical entity metadata used in contextual assertions."""

    model_config = ConfigDict(validate_assignment=True)

    id: str
    name: str
    aliases: list[str] = Field(default_factory=list)


class Evidence(BaseModel):
    """Evidence span metadata that supports an assertion."""

    model_config = ConfigDict(validate_assignment=True)

    doc_id: str | None = None
    chunk_id: str | None = None
    span: tuple[int, int] | None = None
    confidence: float = 1.0

    # Backward-compatible optional metadata used by trust/update flows.
    text: str = ""
    source: str | None = None
    extractor_confidence: float = 1.0

    @property
    def span_start(self) -> int | None:
        return self.span[0] if self.span else None

    @property
    def span_end(self) -> int | None:
        return self.span[1] if self.span else None


class Assertion(BaseModel):
    """Contextual assertion with quality and provenance metadata."""

    model_config = ConfigDict(validate_assignment=True)

    subject: str
    relation: str
    object: str
    qualifiers: dict[str, Any] = Field(default_factory=dict)
    evidence: list[Evidence] = Field(default_factory=list)
    confidence: float = 0.5
    timestamp: float = Field(default_factory=lambda: float(time.time()))

    # Backward-compatible fields used by current CKE components/tests.
    trust_score: float = 0.5
    source: str = "unknown"
    evidence_count: int = 1
    extractor_confidence: float = 1.0
    secondary: bool = False

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        # Keep positional initialization support:
        # Assertion(subject, relation, object, ...)
        if args:
            positional_keys = ("subject", "relation", "object")
            for index, value in enumerate(args[: len(positional_keys)]):
                kwargs.setdefault(positional_keys[index], value)
        super().__init__(**kwargs)
        if "confidence" not in kwargs:
            self.confidence = self.trust_score

    def key(self) -> tuple[str, str, str]:
        """Identity key used for grouping candidate conflicts."""
        return (self.subject, self.relation, self.object)

    def hash(self) -> str:
        """Stable hash for assertion identity + qualifiers."""
        raw = "||".join(
            [
                self.subject,
                self.relation,
                self.object,
                json.dumps(self.qualifiers, sort_keys=True, separators=(",", ":")),
            ]
        )
        return hashlib.sha256(raw.encode("utf-8")).hexdigest()

    def to_dict(self) -> dict[str, Any]:
        """Serialize assertion into a plain dictionary."""
        return self.model_dump(mode="python")
