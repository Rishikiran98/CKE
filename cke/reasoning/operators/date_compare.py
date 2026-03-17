"""Deterministic date comparison operator."""

from __future__ import annotations

from datetime import datetime

from cke.reasoning.operator_types import OperatorOutcome


def _to_datetime(value: str) -> datetime:
    for fmt in ("%Y-%m-%d", "%Y/%m/%d", "%Y"):
        try:
            return datetime.strptime(value, fmt)
        except ValueError:
            continue
    raise ValueError(f"Unsupported date format: {value}")


def date_compare(left: str, right: str, operator: str = "==") -> bool:
    left_dt = _to_datetime(left)
    right_dt = _to_datetime(right)
    if operator == "==":
        return left_dt == right_dt
    if operator == "!=":
        return left_dt != right_dt
    if operator == ">":
        return left_dt > right_dt
    if operator == "<":
        return left_dt < right_dt
    if operator == ">=":
        return left_dt >= right_dt
    if operator == "<=":
        return left_dt <= right_dt
    raise ValueError(f"Unsupported date operator: {operator}")


def evaluate_date_compare(
    left: object, right: object, operator: str = "=="
) -> OperatorOutcome:
    left_text = str(left).strip()
    right_text = str(right).strip()
    try:
        result = date_compare(left_text, right_text, operator)
    except ValueError:
        return OperatorOutcome(
            operator_name="date_compare",
            passed=False,
            result_value=None,
            normalized_inputs=(left_text, right_text, operator),
            confidence=0.0,
            summary="Unable to parse one or both date inputs.",
        )

    return OperatorOutcome(
        operator_name="date_compare",
        passed=result,
        result_value=result,
        normalized_inputs=(left_text, right_text, operator),
        confidence=1.0,
        summary=f"Date compare computed {left_text} {operator} {right_text}.",
    )
