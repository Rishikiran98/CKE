"""Repeatable end-to-end evaluator for Sprint 8 golden cases."""

from __future__ import annotations

from collections import Counter

from cke.evaluation.diagnostics import extract_stage_diagnostics, is_abstained
from cke.evaluation.eval_types import CaseEvaluationResult, EvaluationSummary
from cke.evaluation.failure_classifier import classify_failure, failure_stage
from cke.evaluation.golden_cases import GoldenCase


class E2EEvaluator:
    def __init__(self, orchestrator):
        self.orchestrator = orchestrator

    def evaluate_case(self, case: GoldenCase) -> CaseEvaluationResult:
        query_result = self.orchestrator.answer(case.query)
        context = getattr(self.orchestrator, "last_context", None)
        stage_diagnostics = extract_stage_diagnostics(query_result, context)
        predicted_answer = str(query_result.answer)
        exact_match = _answer_matches(predicted_answer, case.expected_answer)
        acceptable_match = exact_match or any(
            _answer_matches(predicted_answer, answer)
            for answer in case.acceptable_answers
        )
        abstained = is_abstained(query_result)

        if case.expected_answer is None:
            exact_match = abstained
            acceptable_match = abstained

        failure_mode = None
        if not acceptable_match:
            failure_mode = classify_failure(
                case=case,
                predicted_answer=predicted_answer,
                abstained=abstained,
                stage_diagnostics=stage_diagnostics,
                verification_summary=query_result.verification_summary,
                upstream_failure_mode=query_result.failure_mode,
            )

        return CaseEvaluationResult(
            case_id=case.case_id,
            query=case.query,
            predicted_answer=predicted_answer,
            expected_answer=case.expected_answer,
            correct=exact_match,
            acceptable_match=acceptable_match,
            abstained=abstained,
            failure_mode=failure_mode,
            verification_summary=query_result.verification_summary,
            reasoning_route=query_result.reasoning_route,
            trace_id=query_result.trace_id,
            stage_diagnostics=stage_diagnostics,
        )

    def evaluate_cases(
        self, cases: list[GoldenCase]
    ) -> tuple[list[CaseEvaluationResult], EvaluationSummary]:
        results = [self.evaluate_case(case) for case in cases]
        total_cases = len(results)
        exact_matches = sum(result.correct for result in results)
        acceptable_matches = sum(result.acceptable_match for result in results)
        abstentions = sum(result.abstained for result in results)
        failed_cases = total_cases - acceptable_matches

        failure_counter = Counter(
            result.failure_mode for result in results if result.failure_mode
        )
        stage_counter = Counter(
            stage
            for result in results
            for stage in [failure_stage(result.failure_mode)]
            if stage
        )

        summary = EvaluationSummary(
            total_cases=total_cases,
            exact_matches=exact_matches,
            acceptable_matches=acceptable_matches,
            abstentions=abstentions,
            failed_cases=failed_cases,
            accuracy=(exact_matches / total_cases) if total_cases else 0.0,
            acceptable_accuracy=(
                (acceptable_matches / total_cases) if total_cases else 0.0
            ),
            failure_breakdown=dict(sorted(failure_counter.items())),
            stage_failure_breakdown=dict(sorted(stage_counter.items())),
        )
        return results, summary


def _normalize_answer(value: str | int | float | None) -> str:
    if value is None:
        return ""
    text = str(value).strip().lower()
    return " ".join(text.split())


def _answer_matches(predicted: str, expected: str | int | float | None) -> bool:
    return _normalize_answer(predicted) == _normalize_answer(expected)
