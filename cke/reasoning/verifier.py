"""Reasoning verification for deterministic graph reasoning."""

from __future__ import annotations

from typing import Iterable

from cke.models import Statement
from cke.reasoning.operators import equality
from cke.reasoning.verification_types import VerificationOutcome


class ReasoningVerifier:
    """Verifies evidence completeness, logical validity, grounding, and confidence."""

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
    ) -> VerificationOutcome:
        del query
        context_list = list(context)
        path_list = list(reasoning_path)
        checks = list(operator_checks or [])
        issues: list[str] = []

        evidence_complete = self._check_evidence_completeness(
            context=context_list,
            required_facts=required_facts,
            path=path_list,
        )
        if not evidence_complete:
            issues.append("evidence_incomplete")

        logical_valid = self._check_logical_validity(checks)
        if not logical_valid:
            issues.append("logical_invalid")

        confidence_valid = confidence_score >= self.confidence_threshold
        if not confidence_valid:
            issues.append("confidence_below_threshold")

        grounded = self._check_groundedness(
            answer=answer,
            context=context_list,
            path=path_list,
            operator_checks=checks,
        )
        if not grounded:
            issues.append("not_grounded")

        contradiction_relevant = self._has_relevant_contradiction(
            context=context_list,
            answer=answer,
            required_facts=required_facts,
        )
        if contradiction_relevant:
            issues.append("contradictory_evidence")

        passed = (
            evidence_complete
            and logical_valid
            and confidence_valid
            and grounded
            and not contradiction_relevant
        )
        summary = (
            "verification_passed"
            if passed
            else "verification_failed:" + ",".join(issues)
        )
        return VerificationOutcome(
            passed=passed,
            evidence_complete=evidence_complete,
            logical_valid=logical_valid,
            confidence_valid=confidence_valid,
            grounded=grounded,
            contradictory=contradiction_relevant,
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

    def _check_groundedness(
        self,
        answer: str,
        context: list[Statement],
        path: list[Statement],
        operator_checks: list[dict[str, object]],
    ) -> bool:
        if not answer:
            return False
        normalized_answer = answer.strip().lower()

        if any(normalized_answer == st.object.strip().lower() for st in path):
            return True
        if any(normalized_answer == st.object.strip().lower() for st in context):
            return True

        for check in operator_checks:
            if check.get("operator") != "equality":
                continue
            expected = "yes" if bool(check.get("result")) else "no"
            if normalized_answer == expected:
                return True
        return False

    def _has_relevant_contradiction(
        self,
        context: list[Statement],
        answer: str,
        required_facts: Iterable[tuple[str, str]] | None,
    ) -> bool:
        slot_values: dict[tuple[str, str], set[str]] = {}
        for st in context:
            slot = (st.subject, st.relation)
            slot_values.setdefault(slot, set()).add(st.object)

        contradictory_slots = {
            slot for slot, values in slot_values.items() if len(values) > 1
        }
        if not contradictory_slots:
            return False

        answer_lower = answer.strip().lower()
        answer_slots = {
            (st.subject, st.relation)
            for st in context
            if st.object.strip().lower() == answer_lower
        }
        required_slots = set(required_facts or [])

        relevant_slots = answer_slots | required_slots
        if not relevant_slots:
            return bool(contradictory_slots)

        return any(slot in contradictory_slots for slot in relevant_slots)
