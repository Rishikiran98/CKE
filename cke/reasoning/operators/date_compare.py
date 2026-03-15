"""Deterministic date comparison operator."""

from __future__ import annotations

from datetime import datetime


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
