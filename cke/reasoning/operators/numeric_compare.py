"""Deterministic numeric comparison operators."""


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
