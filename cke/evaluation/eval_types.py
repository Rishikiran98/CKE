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
    confidence: float
    confidence_bucket: str
    trace_id: str
    confidence_signals: dict = field(default_factory=dict)
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
    average_confidence_correct: float = 0.0
    average_confidence_incorrect: float = 0.0
    high_confidence_error_rate: float = 0.0
    calibration_by_bucket: dict[str, dict[str, float | int]] = field(
        default_factory=dict
    )
    failure_breakdown: dict[str, int] = field(default_factory=dict)
    stage_failure_breakdown: dict[str, int] = field(default_factory=dict)
    retrieval_metrics: dict[str, float] = field(default_factory=dict)
    retrieval_miss_breakdown: dict[str, int] = field(default_factory=dict)
