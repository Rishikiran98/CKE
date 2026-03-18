"""Repeatable end-to-end evaluator for Sprint 8 golden cases."""

from __future__ import annotations

from collections import Counter

from cke.evaluation.diagnostics import extract_stage_diagnostics, is_abstained
from cke.evaluation.eval_types import CaseEvaluationResult, EvaluationSummary
from cke.evaluation.failure_classifier import classify_failure, failure_stage
from cke.evaluation.golden_cases import GoldenCase
from cke.evaluation.retrieval_tuning import categorize_retrieval_miss


class E2EEvaluator:
    def __init__(self, orchestrator):
        self.orchestrator = orchestrator

    def evaluate_case(self, case: GoldenCase) -> CaseEvaluationResult:
        query_result = self.orchestrator.answer(case.query)
        context = getattr(self.orchestrator, "last_context", None)
        stage_diagnostics = extract_stage_diagnostics(query_result, context)
        self._attach_retrieval_expectations(stage_diagnostics, case, context)
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
            retrieval_metrics=self._summarize_retrieval_metrics(results, total_cases),
            retrieval_miss_breakdown=self._summarize_retrieval_misses(results),
        )
        return results, summary

    def _attach_retrieval_expectations(self, stage_diagnostics, case, context) -> None:
        retrieval_diag = stage_diagnostics.setdefault("retrieval", {})
        evidence_diag = stage_diagnostics.setdefault("evidence_assembly", {})
        chunks = list(getattr(context, "retrieved_chunks", []) if context else [])
        facts = list(getattr(context, "evidence_facts", []) if context else [])
        paths = list(getattr(context, "candidate_paths", []) if context else [])

        expected_entities = {
            _normalize_answer(value) for value in case.expected_entities
        }
        expected_relations = {
            _normalize_answer(value) for value in case.expected_relations if value
        }
        expected_answer = _normalize_answer(case.expected_answer)

        evidence_text = " ".join(
            _normalize_answer(
                " ".join(
                    [
                        fact.statement.subject,
                        fact.statement.relation,
                        fact.statement.object,
                        fact.statement.canonical_subject_id or "",
                        fact.statement.canonical_object_id or "",
                    ]
                )
            )
            for fact in facts
        )
        top_fact_text = " ".join(
            _normalize_answer(fact.statement.as_text()) for fact in facts[:5]
        )
        path_text = " ".join(
            _normalize_answer(
                " ".join(statement.as_text() for statement in path.statements)
            )
            for path in paths
        )

        entity_retrieved = (
            any(entity in evidence_text for entity in expected_entities)
            if expected_entities
            else False
        )
        relation_retrieved = (
            any(relation in evidence_text for relation in expected_relations)
            if expected_relations
            else False
        )
        answer_retrieved = bool(expected_answer and expected_answer in evidence_text)
        answer_in_top_5 = bool(expected_answer and expected_answer in top_fact_text)
        supporting_path_found = bool(expected_answer and expected_answer in path_text)
        if not supporting_path_found and expected_relations:
            supporting_path_found = any(
                any(
                    _normalize_answer(statement.relation) in expected_relations
                    for statement in path.statements
                )
                for path in paths
            )

        retrieval_diag.update(
            {
                "retrieved_chunk_count": len(chunks),
                "entity_retrieved": entity_retrieved,
                "relation_retrieved": relation_retrieved,
                "answer_retrieved": answer_retrieved,
                "answer_in_top_5_facts": answer_in_top_5,
                "supporting_path_found": supporting_path_found,
                "expected_entity_present": entity_retrieved,
                "expected_relation_present": relation_retrieved,
                "expected_answer_present": answer_retrieved,
            }
        )
        evidence_diag.setdefault("evidence_fact_count_before_filtering", len(facts))
        evidence_diag.setdefault("evidence_fact_count_after_filtering", len(facts))

    def _summarize_retrieval_metrics(
        self,
        results: list[CaseEvaluationResult],
        total_cases: int,
    ) -> dict[str, float]:
        if total_cases == 0:
            return {}
        metrics = {
            "retrieval_success_rate": 0,
            "answer_retrieved_rate": 0,
            "answer_in_top_5_facts_rate": 0,
            "supporting_path_found_rate": 0,
        }
        for result in results:
            retrieval = result.stage_diagnostics.get("retrieval", {})
            metrics["retrieval_success_rate"] += int(
                retrieval.get("entity_retrieved", False)
                or retrieval.get("relation_retrieved", False)
            )
            metrics["answer_retrieved_rate"] += int(
                retrieval.get("answer_retrieved", False)
            )
            metrics["answer_in_top_5_facts_rate"] += int(
                retrieval.get("answer_in_top_5_facts", False)
            )
            metrics["supporting_path_found_rate"] += int(
                retrieval.get("supporting_path_found", False)
            )
        return {key: value / total_cases for key, value in metrics.items()}

    def _summarize_retrieval_misses(
        self, results: list[CaseEvaluationResult]
    ) -> dict[str, int]:
        counter = Counter()
        for result in results:
            category = categorize_retrieval_miss(result)
            if category:
                counter[category] += 1
        return dict(sorted(counter.items()))


def _normalize_answer(value: str | int | float | None) -> str:
    if value is None:
        return ""
    text = str(value).strip().lower()
    return " ".join(text.split())


def _answer_matches(predicted: str, expected: str | int | float | None) -> bool:
    return _normalize_answer(predicted) == _normalize_answer(expected)
