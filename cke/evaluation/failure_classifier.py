"""Transparent rule-based failure classification for evaluation runs."""

from __future__ import annotations

from cke.evaluation.golden_cases import GoldenCase


def classify_failure(
    case: GoldenCase,
    predicted_answer: str,
    abstained: bool,
    stage_diagnostics: dict[str, dict[str, object]],
    verification_summary: str,
    upstream_failure_mode: str | None = None,
) -> str | None:
    if case.expected_answer is None and abstained:
        if (
            _contradiction_detected(stage_diagnostics)
            or case.expected_failure_mode == "contradiction_handled"
        ):
            return "contradiction_handled"
        return "expected_abstention"

    if not abstained and case.expected_answer is None:
        if case.expected_failure_mode == "contradiction_handled":
            return "contradiction_missed"
        return "unexpected_answer"

    entity_diag = stage_diagnostics.get("entity_resolution", {})
    relation_diag = stage_diagnostics.get("relation_targeting", {})
    retrieval_diag = stage_diagnostics.get("retrieval", {})
    evidence_diag = stage_diagnostics.get("evidence_assembly", {})
    path_diag = stage_diagnostics.get("path_generation", {})
    operator_diag = stage_diagnostics.get("operator_execution", {})
    verification_diag = stage_diagnostics.get("verification", {})

    if case.expected_entities and int(entity_diag.get("resolved_entity_count", 0)) == 0:
        return "entity_resolution_failure"

    if (
        case.expected_relations
        and int(relation_diag.get("target_relation_count", 0)) == 0
    ):
        return "relation_targeting_failure"

    if int(retrieval_diag.get("retrieved_chunk_count", 0)) == 0:
        return "retrieval_failure"

    if int(evidence_diag.get("evidence_fact_count_before_filtering", 0)) == 0:
        return "fact_mapping_failure"

    if int(evidence_diag.get("evidence_fact_count_after_filtering", 0)) == 0:
        return "retrieval_failure"

    if _expects_path(case) and int(path_diag.get("candidate_path_count", 0)) == 0:
        return "path_generation_failure"

    if case.expected_operator and not operator_diag.get("operator_selected"):
        return "operator_input_failure"

    if upstream_failure_mode == "contradictory_evidence" or _contradiction_detected(
        stage_diagnostics
    ):
        return (
            "contradiction_missed"
            if predicted_answer not in {"CONFLICTING_EVIDENCE", "INSUFFICIENT_EVIDENCE"}
            else "contradiction_handled"
        )

    verifier_issues = list(verification_diag.get("verifier_issues", []))
    if verification_summary.startswith("verification_failed") or verifier_issues:
        return "verification_failure"

    if abstained:
        return "unexpected_abstention"

    return "reasoning_failure"


def failure_stage(failure_mode: str | None) -> str | None:
    mapping = {
        "entity_resolution_failure": "entity_resolution",
        "relation_targeting_failure": "relation_targeting",
        "retrieval_failure": "retrieval",
        "fact_mapping_failure": "evidence_assembly",
        "path_generation_failure": "path_generation",
        "operator_input_failure": "operator_execution",
        "reasoning_failure": "reasoning",
        "verification_failure": "verification",
        "unexpected_abstention": "final_output",
        "unexpected_answer": "final_output",
        "expected_abstention": "final_output",
        "contradiction_handled": "verification",
        "contradiction_missed": "verification",
    }
    return mapping.get(failure_mode)


def _expects_path(case: GoldenCase) -> bool:
    text = f"{case.notes} {case.query}".lower()
    return case.expected_operator == "path" or "2-hop" in text or "two-hop" in text


def _contradiction_detected(stage_diagnostics: dict[str, dict[str, object]]) -> bool:
    verification_diag = stage_diagnostics.get("verification", {})
    return bool(verification_diag.get("contradiction_detected", False))
