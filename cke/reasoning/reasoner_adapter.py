"""Adapter that prepares path-aware reasoning inputs for underlying reasoners."""

from __future__ import annotations

from dataclasses import dataclass

from cke.diagnostics import DegradationMixin
from cke.models import Statement
from cke.pipeline.types import ReasonerOutcome
from cke.retrieval.path_types import CandidatePath


@dataclass(slots=True)
class AdaptedReasoningInput:
    statements: list[Statement]
    selected_path: CandidatePath | None
    path_aware: bool


#: Reported when a wrapped reasoner returns an answer with no confidence
#: of its own. Not a measurement.
_SUBSTITUTED_CONFIDENCE = 0.8


class ReasonerAdapter(DegradationMixin):
    """Thin adapter that prioritizes top candidate paths before fallback facts."""

    def __init__(self, reasoner, strict: bool = False) -> None:
        self._init_degradation(strict)
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
            reasoning_path = (
                list(adapted.selected_path.statements)
                if adapted.selected_path is not None
                else []
            )
            if adapted.statements:
                # The wrapped reasoner returned a bare answer string with no
                # confidence of its own. A constant standing in here is read
                # downstream as the reasoner's own certainty.
                self._degrade(
                    f"{type(self.reasoner).__name__} returned an answer "
                    "carrying no confidence, so a substituted value "
                    f"({_SUBSTITUTED_CONFIDENCE}) is reported as the "
                    "reasoner's confidence",
                )
            return ReasonerOutcome(
                answer=answer,
                confidence=_SUBSTITUTED_CONFIDENCE if adapted.statements else 0.0,
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
