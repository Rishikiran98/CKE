"""Deterministic numeric comparison operators."""

from __future__ import annotations

from cke.reasoning.operator_types import OperatorOutcome


def _to_float(value: object) -> float | None:
    try:
        return float(str(value).strip())
    except (TypeError, ValueError):
        return None


def numeric_compare(left: float, right: float, operator: str = "==") -> bool:
    if operator == "==":
        return left == right
    if operator == "!=":
        return left != right
    if operator == ">":
        return left > right
    if operator == "<":
        return left < right
    if operator == ">=":
        return left >= right
    if operator == "<=":
        return left <= right
    raise ValueError(f"Unsupported numeric operator: {operator}")


def evaluate_numeric_compare(
    left: object, right: object, operator: str = "=="
) -> OperatorOutcome:
    left_num = _to_float(left)
    right_num = _to_float(right)
    if left_num is None or right_num is None:
        return OperatorOutcome(
            operator_name="numeric_compare",
            passed=False,
            result_value=None,
            normalized_inputs=(left, right, operator),
            confidence=0.0,
            summary="Unable to parse one or both numeric inputs.",
        )

    result = numeric_compare(left_num, right_num, operator)
    return OperatorOutcome(
        operator_name="numeric_compare",
        passed=result,
        result_value=result,
        normalized_inputs=(left_num, right_num, operator),
        confidence=1.0,
        summary=f"Numeric compare computed {left_num} {operator} {right_num}.",
    )
