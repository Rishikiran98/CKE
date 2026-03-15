"""Reasoning template memory for recurring deterministic query patterns."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable

from cke.models import Statement
from cke.reasoning.operators import equality


@dataclass(frozen=True, slots=True)
class ReasoningTemplate:
    """A lightweight reusable reasoning template."""

    name: str
    query_type: str
    steps: tuple[str, ...]


@dataclass(slots=True)
class TemplateExecution:
    """Execution result for a template invocation."""

    template_name: str
    answer: str
    path: list[Statement]
    confidence_score: float
    entities: list[str] = field(default_factory=list)
    operators_used: list[str] = field(default_factory=list)
    operator_checks: list[dict[str, object]] = field(default_factory=list)
    required_facts: list[tuple[str, str]] = field(default_factory=list)
    trace: list[str] = field(default_factory=list)


class PatternMemory:
    """Stores and executes reusable reasoning patterns (not graph paths)."""

    def __init__(self) -> None:
        self.templates = (
            ReasoningTemplate(
                name="nationality_comparison",
                query_type="comparison",
                steps=(
                    "retrieve nationality(entity1)",
                    "retrieve nationality(entity2)",
                    "equality_operator",
                ),
            ),
        )

    def execute(
        self,
        query: str,
        context: Iterable[Statement],
    ) -> TemplateExecution | None:
        context_list = list(context)
        if "same nationality" not in query.lower():
            return None

        entities = sorted(
            {st.subject for st in context_list if st.relation == "nationality"}
        )
        if len(entities) < 2:
            return None

        left_entity, right_entity = entities[0], entities[1]
        left_fact = self._find_fact(context_list, left_entity, "nationality")
        right_fact = self._find_fact(context_list, right_entity, "nationality")
        if left_fact is None or right_fact is None:
            return None

        equal_result = equality(left_fact.object, right_fact.object)
        confidence = left_fact.confidence * right_fact.confidence
        answer = "yes" if equal_result else "no"

        return TemplateExecution(
            template_name="nationality_comparison",
            answer=answer,
            path=[left_fact, right_fact],
            confidence_score=confidence,
            entities=[left_entity, right_entity],
            operators_used=["equality"],
            operator_checks=[
                {
                    "operator": "equality",
                    "inputs": (left_fact.object, right_fact.object),
                    "result": equal_result,
                }
            ],
            required_facts=[
                (left_entity, "nationality"),
                (right_entity, "nationality"),
            ],
            trace=[
                "Pattern template matched: nationality_comparison",
                "Template step: retrieve nationality(entity1)",
                "Template step: retrieve nationality(entity2)",
                "Template step: equality_operator",
            ],
        )

    @staticmethod
    def _find_fact(
        context: list[Statement],
        subject: str,
        relation: str,
    ) -> Statement | None:
        for statement in context:
            if statement.subject == subject and statement.relation == relation:
                return statement
        return None
