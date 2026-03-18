"""Golden evaluation case definitions for Sprint 8 diagnostics."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(slots=True)
class GoldenCase:
    case_id: str
    query: str
    expected_answer: str | int | float | None
    acceptable_answers: list[str] = field(default_factory=list)
    expected_entities: list[str] = field(default_factory=list)
    expected_relations: list[str] = field(default_factory=list)
    expected_operator: str | None = None
    expected_failure_mode: str | None = None
    notes: str = ""
