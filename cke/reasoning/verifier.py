"""Reasoning verification for deterministic graph reasoning."""

from __future__ import annotations

from typing import Iterable

from cke.models import Statement
from cke.reasoning.operators import (
    contains,
    date_compare,
    equality,
    numeric_compare,
)
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
            operator_checks=checks,
        )
        if not evidence_complete:
            issues.append("evidence_incomplete")

        logical_valid, logical_issue = self._check_logical_validity(checks)
        if not logical_valid:
            issues.append(logical_issue)

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
        operator_checks: list[dict[str, object]],
    ) -> bool:
        required = list(required_facts or [])
        if operator_checks and not path:
            return False
        if not required:
            return bool(path)

        available = {(st.subject, st.relation) for st in context}
        return all(req in available for req in required)

    def _check_logical_validity(
        self, operator_checks: Iterable[dict[str, object]] | None
    ) -> tuple[bool, str]:
        for check in operator_checks or []:
            operator_name = str(check.get("operator", ""))
            inputs = check.get("inputs")
            result = check.get("result")

            if operator_name == "equality":
                if not isinstance(inputs, tuple) or len(inputs) != 2:
                    return False, "operator_inputs_missing"
                if equality(str(inputs[0]), str(inputs[1])) != bool(result):
                    return False, "operator_result_invalid"
                continue

            if operator_name == "containment":
                if not isinstance(inputs, tuple) or len(inputs) < 2:
                    return False, "operator_inputs_missing"
                if contains(inputs[0], inputs[1]) != bool(result):
                    return False, "operator_result_invalid"
                continue

            if operator_name in {"date_compare", "numeric_compare"}:
                if not isinstance(inputs, tuple) or len(inputs) != 3:
                    return False, "operator_inputs_missing"
                left, right, operator = inputs
                try:
                    recomputed = (
                        date_compare(str(left), str(right), str(operator))
                        if operator_name == "date_compare"
                        else numeric_compare(float(left), float(right), str(operator))
                    )
                except (TypeError, ValueError):
                    return False, "operator_result_invalid"
                if bool(recomputed) != bool(result):
                    return False, "operator_result_invalid"
                continue

            if operator_name == "count":
                if not isinstance(result, int):
                    return False, "operator_result_invalid"
                continue

            if operator_name == "exists":
                if not isinstance(result, bool):
                    return False, "operator_result_invalid"
                continue
        return True, ""

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

        if any(
            normalized_answer in {st.object.strip().lower(), st.subject.strip().lower()}
            for st in path
        ):
            return True
        if any(
            normalized_answer in {st.object.strip().lower(), st.subject.strip().lower()}
            for st in context
        ):
            return True

        for check in operator_checks:
            operator = check.get("operator")
            if operator in {"equality", "containment", "exists"}:
                expected = "yes" if bool(check.get("result")) else "no"
                if normalized_answer == expected:
                    return True
            if (
                operator == "count"
                and str(check.get("result")).strip() == normalized_answer
            ):
                return True
            if (
                operator in {"date_compare", "numeric_compare"}
                and str(check.get("result", "")).strip().lower() == normalized_answer
            ):
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
            if st.context.get("inferred"):
                continue
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
            return False

        return any(slot in contradictory_slots for slot in relevant_slots)
