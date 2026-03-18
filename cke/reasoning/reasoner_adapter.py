"""Adapter that prepares path-aware reasoning inputs for underlying reasoners."""

from __future__ import annotations

from dataclasses import dataclass

from cke.models import Statement
from cke.pipeline.types import ReasonerOutcome
from cke.retrieval.path_types import CandidatePath


@dataclass(slots=True)
class AdaptedReasoningInput:
    statements: list[Statement]
    selected_path: CandidatePath | None
    path_aware: bool


class ReasonerAdapter:
    """Thin adapter that prioritizes top candidate paths before fallback facts."""

    def __init__(self, reasoner) -> None:
        self.reasoner = reasoner

    def build_input(
        self,
        evidence_facts: list[Statement],
        candidate_paths: list[CandidatePath] | None = None,
    ) -> AdaptedReasoningInput:
        candidate_paths = candidate_paths or []
        selected_path = candidate_paths[0] if candidate_paths else None
        ordered: list[Statement] = []
        seen: set[tuple[str, str, str]] = set()

        if selected_path is not None:
            for statement in selected_path.statements:
                if statement.key() in seen:
                    continue
                seen.add(statement.key())
                ordered.append(statement)

        for statement in evidence_facts:
            if statement.key() in seen:
                continue
            seen.add(statement.key())
            ordered.append(statement)

        return AdaptedReasoningInput(
            statements=ordered,
            selected_path=selected_path,
            path_aware=selected_path is not None,
        )

    def reason(
        self,
        query: str,
        evidence_facts: list[Statement],
        candidate_paths: list[CandidatePath] | None = None,
    ) -> ReasonerOutcome | None:
        adapted = self.build_input(evidence_facts, candidate_paths)
        reasoner = self.reasoner
        outcome = None

        if hasattr(reasoner, "reason"):
            try:
                outcome = reasoner.reason(
                    query,
                    adapted.statements,
                    candidate_paths=candidate_paths or [],
                )
            except TypeError:
                outcome = reasoner.reason(query, adapted.statements)

        if outcome is None and hasattr(reasoner, "answer"):
            answer = reasoner.answer(query, adapted.statements)
            if (
                not answer
                or "don't have enough" in answer.lower()
                or "insufficient" in answer.lower()
            ):
                return ReasonerOutcome(
                    answer="INSUFFICIENT_EVIDENCE",
                    confidence=0.0,
                    reasoning_path=[],
                    required_facts=[],
                    operator_checks=[],
                    summary="reasoner_abstained",
                )
            if not answer:
                return None
            reasoning_path = (
                list(adapted.selected_path.statements)
                if adapted.selected_path is not None
                else []
            )
            return ReasonerOutcome(
                answer=answer,
                confidence=0.8 if adapted.statements else 0.0,
                reasoning_path=reasoning_path,
                required_facts=[],
                operator_checks=[],
                summary=(
                    "path_aware_reasoner_completed"
                    if adapted.path_aware
                    else "reasoner_completed"
                ),
            )

        if isinstance(outcome, ReasonerOutcome) and adapted.selected_path is not None:
            if not outcome.reasoning_path:
                outcome.reasoning_path = list(adapted.selected_path.statements)
            if not outcome.summary:
                outcome.summary = "path_aware_reasoner_completed"
        return outcome
