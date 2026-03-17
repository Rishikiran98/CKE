"""Shared deterministic operator result contracts."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from cke.models import Statement


@dataclass(slots=True)
class OperatorOutcome:
    """Structured outcome emitted by deterministic operators."""

    operator_name: str
    passed: bool
    result_value: str | bool | int | float | None
    normalized_inputs: tuple[Any, ...] = field(default_factory=tuple)
    supporting_facts: list[Statement] = field(default_factory=list)
    confidence: float = 0.0
    summary: str = ""
