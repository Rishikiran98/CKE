"""Deterministic containment operator."""

from __future__ import annotations

from cke.reasoning.operator_types import OperatorOutcome


def _normalize(value: object) -> str:
    return str(value).strip().lower()


def contains(container, item) -> bool:
    if isinstance(container, (list, tuple, set)):
        normalized_container = [_normalize(entry) for entry in container]
        return _normalize(item) in normalized_container
    return _normalize(item) in _normalize(container)


def evaluate_contains(container, item) -> OperatorOutcome:
    result = contains(container, item)
    return OperatorOutcome(
        operator_name="containment",
        passed=result,
        result_value=result,
        normalized_inputs=(container, item),
        supporting_facts=[],
        confidence=1.0,
        summary="Containment membership check completed.",
    )
