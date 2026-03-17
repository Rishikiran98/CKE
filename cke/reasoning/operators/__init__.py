"""Deterministic reasoning operators."""

from cke.reasoning.operators.containment_operator import contains, evaluate_contains
from cke.reasoning.operators.count_operator import count_facts
from cke.reasoning.operators.date_compare import date_compare, evaluate_date_compare
from cke.reasoning.operators.equality_operator import equality, evaluate_equality
from cke.reasoning.operators.exists_operator import exists
from cke.reasoning.operators.numeric_compare import (
    evaluate_numeric_compare,
    numeric_compare,
)
from cke.reasoning.operators.set_intersection import set_intersection

__all__ = [
    "equality",
    "evaluate_equality",
    "numeric_compare",
    "evaluate_numeric_compare",
    "date_compare",
    "evaluate_date_compare",
    "set_intersection",
    "contains",
    "evaluate_contains",
    "exists",
    "count_facts",
]
