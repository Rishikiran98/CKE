"""Stable contracts for reasoning verification results."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(slots=True)
class VerificationOutcome:
    """Structured verifier result consumed by orchestrators and adapters."""

    passed: bool
    evidence_complete: bool
    logical_valid: bool
    confidence_valid: bool
    grounded: bool
    contradictory: bool
    summary: str
    issues: list[str] = field(default_factory=list)
