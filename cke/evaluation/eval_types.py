"""Structured result contracts for end-to-end evaluation."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(slots=True)
class CaseEvaluationResult:
    case_id: str
    query: str
    predicted_answer: str
    expected_answer: str | int | float | None
    correct: bool
    acceptable_match: bool
    abstained: bool
    failure_mode: str | None
    verification_summary: str
    reasoning_route: str
    trace_id: str
    stage_diagnostics: dict = field(default_factory=dict)


@dataclass(slots=True)
class EvaluationSummary:
    total_cases: int
    exact_matches: int
    acceptable_matches: int
    abstentions: int
    failed_cases: int
    accuracy: float
    acceptable_accuracy: float
    failure_breakdown: dict[str, int] = field(default_factory=dict)
    stage_failure_breakdown: dict[str, int] = field(default_factory=dict)
    retrieval_metrics: dict[str, float] = field(default_factory=dict)
    retrieval_miss_breakdown: dict[str, int] = field(default_factory=dict)
