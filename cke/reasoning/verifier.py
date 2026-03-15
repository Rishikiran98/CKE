"""Reasoning verification for deterministic graph reasoning."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable

from cke.models import Statement
from cke.reasoning.operators import equality


@dataclass(slots=True)
class VerificationResult:
    """Structured verification output for a reasoning attempt."""

    passed: bool
    evidence_complete: bool
    logical_valid: bool
    confidence_valid: bool
    summary: str
    issues: list[str] = field(default_factory=list)


class ReasoningVerifier:
    """Verifies evidence completeness, operator correctness, and confidence."""

    def __init__(self, confidence_threshold: float = 0.7) -> None:
        self.confidence_threshold = confidence_threshold

    def verify(
        self,
        query: str,
        context: Iterable[Statement],
        reasoning_path: Iterable[Statement],
        answer: str,
        confidence_score: float,
        required_facts: Iterable[tuple[str, str]] | None = None,
        operator_checks: Iterable[dict[str, object]] | None = None,
    ) -> VerificationResult:
        context_list = list(context)
        path_list = list(reasoning_path)
        issues: list[str] = []

        evidence_complete = self._check_evidence_completeness(
            context=context_list,
            required_facts=required_facts,
            path=path_list,
        )
        if not evidence_complete:
            issues.append("evidence_incomplete")

        logical_valid = self._check_logical_validity(operator_checks)
        if not logical_valid:
            issues.append("logical_invalid")

        confidence_valid = confidence_score >= self.confidence_threshold
        if not confidence_valid:
            issues.append(
                f"confidence_below_threshold:{confidence_score:.3f}<{self.confidence_threshold:.3f}"
            )

        passed = evidence_complete and logical_valid and confidence_valid
        summary = (
            "verification_passed"
            if passed
            else "verification_failed:" + ",".join(issues)
        )
        return VerificationResult(
            passed=passed,
            evidence_complete=evidence_complete,
            logical_valid=logical_valid,
            confidence_valid=confidence_valid,
            summary=summary,
            issues=issues,
        )

    def _check_evidence_completeness(
        self,
        context: list[Statement],
        required_facts: Iterable[tuple[str, str]] | None,
        path: list[Statement],
    ) -> bool:
        required = list(required_facts or [])
        if not required:
            return bool(path)

        available = {(st.subject, st.relation) for st in context}
        return all(req in available for req in required)

    def _check_logical_validity(
        self, operator_checks: Iterable[dict[str, object]] | None
    ) -> bool:
        for check in operator_checks or []:
            operator_name = str(check.get("operator", ""))
            if operator_name != "equality":
                continue
            inputs = check.get("inputs")
            result = bool(check.get("result"))
            if not isinstance(inputs, tuple) or len(inputs) != 2:
                return False
            if equality(str(inputs[0]), str(inputs[1])) != result:
                return False
        return True
