"""Deterministic existence operator."""

from __future__ import annotations

from cke.models import Statement
from cke.reasoning.operator_types import OperatorOutcome


def exists(facts: list[Statement]) -> OperatorOutcome:
    has_facts = bool(facts)
    return OperatorOutcome(
        operator_name="exists",
        passed=has_facts,
        result_value=has_facts,
        normalized_inputs=(len(facts),),
        supporting_facts=list(facts),
        confidence=1.0 if has_facts else 0.0,
        summary=(
            f"Found {len(facts)} supporting fact(s)."
            if has_facts
            else "No supporting facts found."
        ),
    )
