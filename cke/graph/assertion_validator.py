"""Validation helpers for extracted assertions/statements."""

from __future__ import annotations

from cke.extractor.llm_extractor import validate_qualifiers
from cke.models import Statement


class AssertionValidator:
    """Validate assertion/statement fields for graph ingestion."""

    def validate(self, assertion: Statement) -> bool:
        if not assertion.subject or not assertion.subject.strip():
            return False
        if not assertion.object or not assertion.object.strip():
            return False
        if not assertion.relation or not assertion.relation.strip():
            return False
        if not (0.0 <= float(assertion.confidence) <= 1.0):
            return False
        return validate_qualifiers(assertion.qualifiers)
