"""Deterministic count operator."""

from __future__ import annotations

from cke.models import Statement
from cke.reasoning.operator_types import OperatorOutcome


def count_facts(facts: list[Statement]) -> OperatorOutcome:
    count = len(facts)
    return OperatorOutcome(
        operator_name="count",
        passed=True,
        result_value=count,
        normalized_inputs=(count,),
        supporting_facts=list(facts),
        confidence=1.0 if facts else 0.0,
        summary=f"Counted {count} matching fact(s).",
    )
