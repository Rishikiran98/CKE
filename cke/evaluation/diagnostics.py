"""Helpers for extracting deterministic stage diagnostics from query runs."""

from __future__ import annotations

from typing import Any

from cke.pipeline.types import QueryResult, ReasoningContext

ABSTAIN_ANSWERS = {
    "INSUFFICIENT_EVIDENCE",
    "CONFLICTING_EVIDENCE",
    "REASONING_FAILED",
}

STAGE_LABELS = [
    "entity_resolution",
    "relation_targeting",
    "retrieval",
    "evidence_assembly",
    "path_generation",
    "operator_execution",
    "reasoning",
    "verification",
    "final_output",
]


def is_abstained(result: QueryResult) -> bool:
    return result.answer in ABSTAIN_ANSWERS or result.failure_mode is not None


def extract_stage_diagnostics(
    result: QueryResult,
    context: ReasoningContext | None = None,
) -> dict[str, dict[str, Any]]:
    debug_info = dict(result.debug_info)
    trace_metadata = dict(getattr(context, "trace_metadata", {})) if context else {}

    resolved_entities = trace_metadata.get(
        "resolved_entities", debug_info.get("resolved_entities", [])
    )
    target_relations = trace_metadata.get(
        "target_relations", debug_info.get("target_relations", [])
    )
    evidence_before = debug_info.get(
        "evidence_facts_before_filtering",
        trace_metadata.get(
            "evidence_facts_before_filtering", len(result.evidence_facts)
        ),
    )
    evidence_after = debug_info.get(
        "evidence_facts_after_filtering",
        trace_metadata.get(
            "evidence_facts_after_filtering", len(result.evidence_facts)
        ),
    )
    candidate_paths_before = debug_info.get(
        "candidate_paths_before_scoring",
        trace_metadata.get(
            "candidate_paths_before_scoring", len(result.candidate_paths)
        ),
    )
    candidate_paths_after = debug_info.get(
        "candidate_path_count",
        trace_metadata.get("candidate_paths", len(result.candidate_paths)),
    )

    diagnostics = {
        "entity_resolution": {
            "resolved_entity_count": len(resolved_entities),
            "resolved_entity_names": [
                entity.get("canonical_name", "") for entity in resolved_entities
            ],
            "aliases_matched": [
                entity.get("surface_form", "") for entity in resolved_entities
            ],
            "ok": len(resolved_entities) > 0,
        },
        "relation_targeting": {
            "target_relations": list(target_relations),
            "target_relation_count": len(target_relations),
            "ok": len(target_relations) > 0,
        },
        "retrieval": {
            "retrieved_chunk_count": debug_info.get(
                "retrieved_chunk_count",
                len(getattr(context, "retrieved_chunks", [])) if context else 0,
            ),
            "ok": debug_info.get(
                "retrieved_chunk_count",
                len(getattr(context, "retrieved_chunks", [])) if context else 0,
            )
            > 0,
        },
        "evidence_assembly": {
            "evidence_fact_count_before_filtering": int(evidence_before),
            "evidence_fact_count_after_filtering": int(evidence_after),
            "subgraph_entity_count": int(
                debug_info.get(
                    "subgraph_entity_count",
                    trace_metadata.get("subgraph_entity_count", 0),
                )
            ),
            "subgraph_edge_count": int(
                debug_info.get(
                    "subgraph_edge_count",
                    trace_metadata.get("subgraph_edge_count", 0),
                )
            ),
            "ok": int(evidence_after) > 0,
        },
        "path_generation": {
            "candidate_path_count_before_scoring": int(candidate_paths_before),
            "candidate_path_count": int(candidate_paths_after),
            "ok": int(candidate_paths_after) > 0,
        },
        "operator_execution": {
            "selected_operator": debug_info.get("selected_operator"),
            "operator_selected": bool(debug_info.get("selected_operator")),
            "operator_summary": debug_info.get("operator_summary", ""),
            "ok": True,
        },
        "reasoning": {
            "reasoning_route": result.reasoning_route,
            "route_confidence": debug_info.get("route_confidence", 0.0),
            "reasoner_summary": debug_info.get("reasoner_summary", ""),
            "reasoning_path_length": int(debug_info.get("reasoning_path_length", 0)),
            "ok": bool(debug_info.get("reasoner_summary")) or not is_abstained(result),
        },
        "verification": {
            "verification_summary": result.verification_summary,
            "verifier_passed": debug_info.get("verification_passed"),
            "verifier_issues": list(debug_info.get("verification_issues", [])),
            "contradiction_detected": bool(
                debug_info.get("contradiction_detected", False)
            ),
            "confidence_signals": dict(result.confidence_signals),
            "ok": result.verification_summary == "reasoning_not_executed"
            or not result.verification_summary.startswith("verification_failed"),
        },
        "final_output": {
            "abstained": is_abstained(result),
            "failure_mode": result.failure_mode,
            "trace_id": result.trace_id,
            "answer": result.answer,
            "ok": result.failure_mode is None,
        },
    }

    return diagnostics
