"""Execute deterministic operators over grounded evidence."""

from __future__ import annotations

import re

from cke.models import Statement
from cke.pipeline.types import ResolvedEntity
from cke.reasoning.operator_types import OperatorOutcome
from cke.reasoning.operators import (
    count_facts,
    evaluate_contains,
    evaluate_date_compare,
    evaluate_equality,
    evaluate_numeric_compare,
    exists,
)


class OperatorExecutor:
    @staticmethod
    def _relation_matches(relation_hint: str | None, relation: str) -> bool:
        if relation_hint is None:
            return True
        hint = relation_hint.replace("_", "").lower()
        rel = relation.replace("_", "").lower()
        return (
            hint in rel
            or rel in hint
            or hint.startswith("release")
            and rel.startswith("release")
        )

    def execute(
        self,
        operator_hint: str,
        query: str,
        evidence_facts: list[Statement],
        resolved_entities: list[ResolvedEntity],
    ) -> OperatorOutcome | None:
        op = operator_hint.strip().lower()
        if op == "equality":
            return self._execute_equality(query, evidence_facts, resolved_entities)
        if op == "count":
            return self._execute_count(query, evidence_facts, resolved_entities)
        if op == "existence":
            return self._execute_exists(query, evidence_facts, resolved_entities)
        if op == "containment":
            return self._execute_containment(query, evidence_facts, resolved_entities)
        if op in {"temporal_compare", "numeric_compare"}:
            return self._execute_compare(op, query, evidence_facts, resolved_entities)
        return None

    def _relation_hint(self, query: str) -> str | None:
        lowered = query.lower()
        candidates = [
            "nationality",
            "release_year",
            "released",
            "child",
            "children",
            "member",
            "part_of",
            "directed",
            "director",
        ]
        for rel in candidates:
            if rel.replace("_", " ") in lowered or rel in lowered:
                return rel
        return None

    def _matching_facts(
        self,
        query: str,
        evidence_facts: list[Statement],
        resolved_entities: list[ResolvedEntity],
    ) -> list[Statement]:
        lowered = query.lower()
        entity_names = [entity.canonical_name.lower() for entity in resolved_entities]
        relation_hint = self._relation_hint(query)
        matched: list[Statement] = []
        for fact in evidence_facts:
            subject = fact.subject.lower()
            obj = fact.object.lower()
            rel = fact.relation.lower()
            entity_ok = not entity_names or any(
                entity in subject or entity in obj for entity in entity_names
            )
            relation_ok = self._relation_matches(relation_hint, rel)
            lexical_ok = any(token in f" {lowered} " for token in [subject, obj, rel])
            if entity_ok and (relation_ok or lexical_ok):
                matched.append(fact)
        return matched

    def _execute_equality(
        self, query: str, facts: list[Statement], entities: list[ResolvedEntity]
    ) -> OperatorOutcome | None:
        relation_hint = self._relation_hint(query)
        values: list[tuple[str, Statement]] = []
        for entity in entities:
            fact = next(
                (
                    st
                    for st in facts
                    if st.subject.lower() == entity.canonical_name.lower()
                    and self._relation_matches(relation_hint, st.relation.lower())
                ),
                None,
            )
            if fact is not None:
                values.append((fact.object, fact))
        if len(values) < 2:
            return None
        outcome = evaluate_equality(values[0][0], values[1][0])
        outcome.supporting_facts = [values[0][1], values[1][1]]
        return outcome

    def _execute_count(
        self, query: str, facts: list[Statement], entities: list[ResolvedEntity]
    ) -> OperatorOutcome | None:
        matched = self._matching_facts(query, facts, entities)
        if not matched:
            return None
        return count_facts(matched)

    def _execute_exists(
        self, query: str, facts: list[Statement], entities: list[ResolvedEntity]
    ) -> OperatorOutcome | None:
        matched = self._matching_facts(query, facts, entities)
        if not matched:
            return None
        return exists(matched)

    def _execute_containment(
        self, query: str, facts: list[Statement], entities: list[ResolvedEntity]
    ) -> OperatorOutcome | None:
        if len(entities) < 2:
            return None
        left = entities[0].canonical_name.lower()
        right = entities[1].canonical_name.lower()
        matched = [
            st
            for st in facts
            if left in st.subject.lower() and right in st.object.lower()
        ]
        if not matched:
            matched = [
                st
                for st in facts
                if right in st.subject.lower() and left in st.object.lower()
            ]
        if not matched:
            return None
        outcome = evaluate_contains(
            [st.object for st in matched], entities[1].canonical_name
        )
        outcome.supporting_facts = matched
        return outcome

    def _execute_compare(
        self,
        operator_hint: str,
        query: str,
        facts: list[Statement],
        entities: list[ResolvedEntity],
    ) -> OperatorOutcome | None:
        if len(entities) < 2:
            return None
        relation_hint = self._relation_hint(query)
        matched_values: list[tuple[str, Statement]] = []
        for entity in entities[:2]:
            fact = next(
                (
                    st
                    for st in facts
                    if st.subject.lower() == entity.canonical_name.lower()
                    and self._relation_matches(relation_hint, st.relation.lower())
                ),
                None,
            )
            if fact:
                matched_values.append((fact.object, fact))
        if len(matched_values) < 2:
            return None

        lowered = query.lower()
        comparator = ">"
        if any(
            token in lowered
            for token in ["older", "earlier", "before", "smaller", "less", "first"]
        ):
            comparator = "<"

        if operator_hint == "temporal_compare" or re.search(
            r"\b(19|20)\d{2}\b", matched_values[0][0]
        ):
            outcome = evaluate_date_compare(
                matched_values[0][0], matched_values[1][0], comparator
            )
        else:
            outcome = evaluate_numeric_compare(
                matched_values[0][0], matched_values[1][0], comparator
            )

        if outcome.result_value is None:
            return None
        winner = (
            entities[0].canonical_name
            if bool(outcome.result_value)
            else entities[1].canonical_name
        )
        outcome.operator_name = "compare_selection"
        outcome.result_value = winner
        outcome.passed = True
        outcome.supporting_facts = [matched_values[0][1], matched_values[1][1]]
        outcome.summary = f"Compared values and selected '{winner}'."
        return outcome
