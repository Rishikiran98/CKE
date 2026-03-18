"""Sprint 8 tests for the evaluation harness and failure analytics."""

from cke.evaluation.diagnostics import extract_stage_diagnostics
from cke.evaluation.e2e_evaluator import E2EEvaluator
from cke.evaluation.failure_classifier import classify_failure, failure_stage
from cke.evaluation.golden_cases import GoldenCase
from cke.models import Statement
from cke.pipeline.types import (
    EvidenceFact,
    QueryResult,
    ReasoningContext,
    ResolvedEntity,
)
from cke.router.query_plan import QueryPlan


class StubOrchestrator:
    def __init__(
        self, responses: dict[str, QueryResult], contexts: dict[str, ReasoningContext]
    ):
        self.responses = responses
        self.contexts = contexts
        self.last_context: ReasoningContext | None = None

    def answer(self, query: str) -> QueryResult:
        self.last_context = self.contexts[query]
        return self.responses[query]


def _context(
    *,
    resolved_entities: list[str],
    target_relations: list[str],
    retrieved_chunks: int,
    evidence_before: int,
    evidence_after: int,
    candidate_paths: int,
) -> ReasoningContext:
    return ReasoningContext(
        query="",
        query_plan=QueryPlan(target_relations=target_relations),
        resolved_entities=[
            ResolvedEntity(
                surface_form=name,
                canonical_name=name,
                entity_id=name.lower().replace(" ", "_"),
                link_confidence=0.95,
            )
            for name in resolved_entities
        ],
        retrieved_chunks=[object()] * retrieved_chunks,
        evidence_facts=[
            EvidenceFact(
                statement=Statement("Albert Einstein", "nationality", "German"),
                chunk_id="d1::c0",
                source="d1",
                trust_score=0.9,
                retrieval_score=0.9,
                entity_alignment_score=0.2,
            )
        ]
        * evidence_after,
        candidate_paths=[object()] * candidate_paths,
        trace_metadata={
            "target_relations": target_relations,
            "resolved_entities": [
                {
                    "surface_form": name,
                    "canonical_name": name,
                    "entity_id": name.lower().replace(" ", "_"),
                    "link_confidence": 0.95,
                }
                for name in resolved_entities
            ],
            "evidence_facts_before_filtering": evidence_before,
            "evidence_facts_after_filtering": evidence_after,
            "candidate_paths_before_scoring": candidate_paths,
            "candidate_paths": candidate_paths,
            "subgraph_entity_count": max(len(resolved_entities), evidence_after),
            "subgraph_edge_count": evidence_after,
        },
    )


def test_sprint8_single_case_evaluation():
    query = "What is the nationality of Albert Einstein?"
    result = QueryResult(
        answer="German",
        confidence=0.9,
        reasoning_route="advanced_reasoner",
        verification_summary="verification_passed",
        trace_id="trace-1",
        debug_info={
            "resolved_entities": [{"canonical_name": "Albert Einstein"}],
            "target_relations": ["nationality"],
            "retrieved_chunk_count": 1,
            "evidence_facts_before_filtering": 1,
            "evidence_facts_after_filtering": 1,
            "candidate_path_count": 0,
            "candidate_paths_before_scoring": 0,
            "selected_operator": None,
            "reasoner_summary": "direct_lookup",
            "reasoning_path_length": 1,
            "verification_passed": True,
            "verification_issues": [],
            "contradiction_detected": False,
        },
    )
    context = _context(
        resolved_entities=["Albert Einstein"],
        target_relations=["nationality"],
        retrieved_chunks=1,
        evidence_before=1,
        evidence_after=1,
        candidate_paths=0,
    )
    evaluator = E2EEvaluator(StubOrchestrator({query: result}, {query: context}))
    case = GoldenCase(
        case_id="direct-1",
        query=query,
        expected_answer="German",
        expected_entities=["Albert Einstein"],
        expected_relations=["nationality"],
    )

    rows, summary = evaluator.evaluate_cases([case])

    assert len(rows) == 1
    row = rows[0]
    assert row.case_id == "direct-1"
    assert row.correct is True
    assert row.acceptable_match is True
    assert row.failure_mode is None
    assert row.stage_diagnostics["retrieval"]["retrieved_chunk_count"] == 1
    assert summary.total_cases == 1
    assert summary.exact_matches == 1
    assert summary.acceptable_matches == 1
    assert summary.failed_cases == 0


def test_sprint8_mixed_case_evaluation_summary():
    success_query = "What is the nationality of Albert Einstein?"
    abstain_query = "What is the nationality of Unknown Person?"
    failure_query = "Which film is associated with the character portrayed by Person X?"

    responses = {
        success_query: QueryResult(
            answer="German",
            confidence=0.9,
            reasoning_route="advanced_reasoner",
            verification_summary="verification_passed",
            trace_id="trace-success",
            debug_info={
                "resolved_entities": [{"canonical_name": "Albert Einstein"}],
                "target_relations": ["nationality"],
                "retrieved_chunk_count": 1,
                "evidence_facts_before_filtering": 1,
                "evidence_facts_after_filtering": 1,
                "candidate_path_count": 0,
                "candidate_paths_before_scoring": 0,
                "reasoner_summary": "direct_lookup",
                "reasoning_path_length": 1,
                "verification_passed": True,
                "verification_issues": [],
            },
        ),
        abstain_query: QueryResult(
            answer="INSUFFICIENT_EVIDENCE",
            confidence=0.0,
            reasoning_route="advanced_reasoner",
            verification_summary="reasoning_not_executed",
            trace_id="trace-abstain",
            failure_mode="no_evidence",
            debug_info={
                "resolved_entities": [{"canonical_name": "Unknown Person"}],
                "target_relations": ["nationality"],
                "retrieved_chunk_count": 0,
                "evidence_facts_before_filtering": 0,
                "evidence_facts_after_filtering": 0,
                "candidate_path_count": 0,
                "candidate_paths_before_scoring": 0,
                "verification_passed": None,
                "verification_issues": [],
            },
        ),
        failure_query: QueryResult(
            answer="INSUFFICIENT_EVIDENCE",
            confidence=0.0,
            reasoning_route="advanced_reasoner",
            verification_summary="verification_failed:not_grounded",
            trace_id="trace-failure",
            failure_mode="verification_failed",
            debug_info={
                "resolved_entities": [{"canonical_name": "Person X"}],
                "target_relations": ["portrayed", "appears_in"],
                "retrieved_chunk_count": 2,
                "evidence_facts_before_filtering": 2,
                "evidence_facts_after_filtering": 2,
                "candidate_path_count": 0,
                "candidate_paths_before_scoring": 0,
                "selected_operator": None,
                "reasoner_summary": "fallback_abstain",
                "reasoning_path_length": 0,
                "verification_passed": False,
                "verification_issues": ["not_grounded"],
            },
        ),
    }
    contexts = {
        success_query: _context(
            resolved_entities=["Albert Einstein"],
            target_relations=["nationality"],
            retrieved_chunks=1,
            evidence_before=1,
            evidence_after=1,
            candidate_paths=0,
        ),
        abstain_query: _context(
            resolved_entities=["Unknown Person"],
            target_relations=["nationality"],
            retrieved_chunks=0,
            evidence_before=0,
            evidence_after=0,
            candidate_paths=0,
        ),
        failure_query: _context(
            resolved_entities=["Person X"],
            target_relations=["portrayed", "appears_in"],
            retrieved_chunks=2,
            evidence_before=2,
            evidence_after=2,
            candidate_paths=0,
        ),
    }
    evaluator = E2EEvaluator(StubOrchestrator(responses, contexts))
    cases = [
        GoldenCase(
            case_id="success",
            query=success_query,
            expected_answer="German",
            expected_entities=["Albert Einstein"],
            expected_relations=["nationality"],
        ),
        GoldenCase(
            case_id="abstain",
            query=abstain_query,
            expected_answer=None,
            expected_entities=["Unknown Person"],
            expected_relations=["nationality"],
            expected_failure_mode="expected_abstention",
        ),
        GoldenCase(
            case_id="failure",
            query=failure_query,
            expected_answer="Film Z",
            expected_entities=["Person X", "Character Y", "Film Z"],
            expected_relations=["portrayed", "appears_in"],
            expected_operator="path",
            notes="2-hop bridge retrieval",
        ),
    ]

    rows, summary = evaluator.evaluate_cases(cases)

    assert len(rows) == 3
    assert summary.total_cases == 3
    assert summary.exact_matches == 2
    assert summary.acceptable_matches == 2
    assert summary.abstentions == 2
    assert summary.failed_cases == 1
    assert summary.failure_breakdown["path_generation_failure"] == 1
    assert summary.stage_failure_breakdown["path_generation"] == 1


def test_sprint8_failure_classification_rules():
    case = GoldenCase(
        case_id="path-failure",
        query="Which film is associated with the character portrayed by Person X?",
        expected_answer="Film Z",
        expected_entities=["Person X", "Character Y", "Film Z"],
        expected_relations=["portrayed", "appears_in"],
        expected_operator="path",
        notes="2-hop bridge retrieval",
    )
    result = QueryResult(
        answer="INSUFFICIENT_EVIDENCE",
        confidence=0.0,
        reasoning_route="advanced_reasoner",
        verification_summary="verification_failed:not_grounded",
        trace_id="trace-3",
        failure_mode="verification_failed",
        debug_info={
            "resolved_entities": [{"canonical_name": "Person X"}],
            "target_relations": ["portrayed", "appears_in"],
            "retrieved_chunk_count": 2,
            "evidence_facts_before_filtering": 2,
            "evidence_facts_after_filtering": 2,
            "candidate_path_count": 0,
            "candidate_paths_before_scoring": 0,
            "verification_passed": False,
            "verification_issues": ["not_grounded"],
        },
    )
    context = _context(
        resolved_entities=["Person X"],
        target_relations=["portrayed", "appears_in"],
        retrieved_chunks=2,
        evidence_before=2,
        evidence_after=2,
        candidate_paths=0,
    )

    diagnostics = extract_stage_diagnostics(result, context)
    failure_mode = classify_failure(
        case=case,
        predicted_answer=result.answer,
        abstained=True,
        stage_diagnostics=diagnostics,
        verification_summary=result.verification_summary,
        upstream_failure_mode=result.failure_mode,
    )

    assert failure_mode == "path_generation_failure"
    assert failure_stage(failure_mode) == "path_generation"
