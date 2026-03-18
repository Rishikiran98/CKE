"""Helpers for mining retrieval/ranking misses from golden-set evaluations."""

from __future__ import annotations

from collections import Counter

from cke.evaluation.eval_types import CaseEvaluationResult


def categorize_retrieval_miss(result: CaseEvaluationResult) -> str | None:
    """Bucket retrieval misses into manual tuning categories."""
    if result.acceptable_match:
        return None
    retrieval = result.stage_diagnostics.get("retrieval", {})
    evidence = result.stage_diagnostics.get("evidence_assembly", {})
    path = result.stage_diagnostics.get("path_generation", {})

    if retrieval.get("entity_retrieved") and not retrieval.get("relation_retrieved"):
        return "correct_entity_retrieved_wrong_relation_prioritized"
    if retrieval.get("answer_retrieved") and not retrieval.get("answer_in_top_5_facts"):
        return "right_fact_ranked_too_low"
    if (
        int(evidence.get("evidence_fact_count_before_filtering", 0)) > 0
        and int(evidence.get("evidence_fact_count_after_filtering", 0)) == 0
    ):
        return "right_chunk_retrieved_fact_filtered_out"
    if retrieval.get("entity_retrieved") and not retrieval.get("answer_retrieved"):
        return "alias_mismatch_lowered_evidence_rank"
    if retrieval.get("answer_retrieved") and not retrieval.get("supporting_path_found"):
        return "correct_path_exists_but_not_found"
    if (
        retrieval.get("supporting_path_found")
        and int(path.get("candidate_path_count", 0)) == 0
    ):
        return "correct_path_ranked_too_low"
    if retrieval.get("relation_retrieved") and not retrieval.get("answer_retrieved"):
        return "right_chunk_retrieved_wrong_fact_prioritized"
    return "unclassified_retrieval_miss"


def mine_retrieval_failures(
    results: list[CaseEvaluationResult],
) -> dict[str, object]:
    """Return categorized misses for heuristic weight tuning."""
    categories = Counter()
    affected_case_ids: dict[str, list[str]] = {}
    for result in results:
        category = categorize_retrieval_miss(result)
        if category is None:
            continue
        categories[category] += 1
        affected_case_ids.setdefault(category, []).append(result.case_id)
    return {
        "category_counts": dict(sorted(categories.items())),
        "case_ids_by_category": affected_case_ids,
    }
