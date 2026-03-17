"""Deterministic equality operator."""

from __future__ import annotations

from cke.reasoning.operator_types import OperatorOutcome


def _normalize(value: object) -> str:
    return str(value).strip().lower()


def equality(a, b) -> bool:
    """Legacy bool-only equality check."""
    return _normalize(a) == _normalize(b)


def evaluate_equality(a, b) -> OperatorOutcome:
    left = _normalize(a)
    right = _normalize(b)
    result = left == right
    return OperatorOutcome(
        operator_name="equality",
        passed=result,
        result_value=result,
        normalized_inputs=(left, right),
        supporting_facts=[],
        confidence=1.0,
        summary=f"Compared normalized values '{left}' and '{right}'.",
    )
