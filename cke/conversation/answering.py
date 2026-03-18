"""Grounded answer composition for conversational memory and RAG flows."""

from __future__ import annotations

import re

from cke.conversation.types import ConversationAnswer, RetrievalBundle
from cke.models import Statement
from cke.reasoning.operator_executor import OperatorExecutor
from cke.reasoning.operator_selector import OperatorSelector
from cke.router.query_plan import QueryPlan


class GroundedAnswerComposer:
    """Compose natural grounded responses from retrieved turns and facts."""

    def __init__(
        self,
        operator_selector: OperatorSelector | None = None,
        operator_executor: OperatorExecutor | None = None,
    ) -> None:
        self.operator_selector = operator_selector or OperatorSelector()
        self.operator_executor = operator_executor or OperatorExecutor()

    def compose(self, query: str, bundle: RetrievalBundle) -> ConversationAnswer:
        evidence_facts = bundle.retrieved_facts or bundle.graph_neighbors
        if not bundle.retrieved_turns:
            return ConversationAnswer(
                answer=(
                    (
                        "I don't have enough grounded conversation history "
                        "to answer that yet."
                    )
                ),
                confidence=0.0,
                grounded=False,
                metadata={"abstained": True, "reason": "no_retrieval"},
            )

        if self._retrieval_is_weak(bundle) or self._missing_relation_support(
            query, evidence_facts
        ):
            return ConversationAnswer(
                answer=(
                    (
                        "I found only weak matches in the conversation history, "
                        "so I can't answer that confidently."
                    )
                ),
                confidence=0.2,
                grounded=False,
                retrieved_turns=bundle.retrieved_turns,
                retrieved_facts=evidence_facts,
                graph_neighbors=bundle.graph_neighbors,
                metadata={
                    "abstained": True,
                    "reason": "weak_retrieval_or_missing_support",
                },
            )

        operator_answer = self._operator_answer(query, evidence_facts)
        if operator_answer is not None:
            return ConversationAnswer(
                answer=operator_answer,
                confidence=0.84,
                grounded=True,
                retrieved_turns=bundle.retrieved_turns,
                retrieved_facts=evidence_facts,
                graph_neighbors=bundle.graph_neighbors,
                metadata={"answer_mode": "operator_augmented"},
            )

        lowered = query.lower()
        if any(token in lowered for token in ["when", "what date", "when was"]):
            answer = self._when_answer(bundle)
        elif any(
            token in lowered
            for token in [
                "who hasn't replied",
                "who hasnt replied",
                "who has not replied",
            ]
        ):
            answer = self._pending_reply_answer(bundle)
        elif (
            "preferred backend" in lowered
            or "didn't i say i preferred" in lowered
            or "didnt i say" in lowered
        ):
            answer = self._preference_confirmation(bundle)
        elif "should i apply" in lowered:
            answer = self._recommendation_answer(bundle)
        else:
            answer = self._summary_answer(query, bundle)

        grounded = not answer.startswith("I don't") and not answer.startswith(
            "I found only weak"
        )
        confidence = 0.8 if grounded else 0.25
        return ConversationAnswer(
            answer=answer,
            confidence=confidence,
            grounded=grounded,
            retrieved_turns=bundle.retrieved_turns,
            retrieved_facts=evidence_facts,
            graph_neighbors=bundle.graph_neighbors,
            metadata={"answer_mode": "grounded_natural_language"},
        )

    def _detect_operator_hint(self, query: str) -> str | None:
        lowered = f" {query.lower()} "
        if " how many" in lowered:
            return "count"
        if any(token in lowered for token in [" same ", " equal ", " identical "]):
            return "equality"
        if any(
            token in lowered
            for token in [" before ", " after ", " later ", " earlier ", " when "]
        ):
            return "temporal_compare"
        if lowered.strip().startswith(("did ", "is ", "has ", "was ")):
            return "existence"
        return None

    def _operator_answer(
        self, query: str, evidence_facts: list[Statement]
    ) -> str | None:
        statements = evidence_facts or []
        if not statements:
            return None
        query_plan = QueryPlan(operator_hint=self._detect_operator_hint(query))
        operator = self.operator_selector.select(
            query=query,
            query_plan=query_plan,
            resolved_entities=[],
            statements=statements,
        )
        if not operator:
            return None
        outcome = self.operator_executor.execute(
            operator_hint=operator,
            query=query,
            evidence_facts=statements,
            resolved_entities=[],
        )
        if outcome is None:
            return None
        result = outcome.result_value
        if result is None:
            return None
        if isinstance(result, bool):
            return "yes" if result else "no"
        return str(result)

    def _when_answer(self, bundle: RetrievalBundle) -> str:
        for turn in bundle.retrieved_turns:
            date_match = re.search(
                (
                    r"\b(?:Monday|Tuesday|Wednesday|Thursday|Friday|Saturday|Sunday|"
                    r"next\s+week|next\s+month|January\s+\d{1,2}|"
                    r"February\s+\d{1,2}|March\s+\d{1,2}|April\s+\d{1,2}|"
                    r"May\s+\d{1,2}|June\s+\d{1,2}|July\s+\d{1,2}|"
                    r"August\s+\d{1,2}|September\s+\d{1,2}|October\s+\d{1,2}|"
                    r"November\s+\d{1,2}|December\s+\d{1,2}|\d{4}-\d{2}-\d{2})\b"
                ),
                turn.text,
                flags=re.IGNORECASE,
            )
            if date_match:
                return f"You said it was {date_match.group(0)}."
        return self._summary_answer("when", bundle)

    def _pending_reply_answer(self, bundle: RetrievalBundle) -> str:
        pending = {
            fact.subject
            for fact in bundle.retrieved_facts
            if fact.relation == "reply_status" and fact.object == "pending"
        }
        replied = {
            fact.subject
            for fact in bundle.retrieved_facts
            if fact.relation == "reply_status" and fact.object == "replied"
        }
        outstanding = sorted(pending - replied)
        if outstanding:
            return (
                "You said you're still waiting to hear back from "
                f"{', '.join(outstanding)}."
            )
        return (
            "Based on the retrieved conversation history, everyone mentioned "
            "there has already replied."
        )

    def _preference_confirmation(self, bundle: RetrievalBundle) -> str:
        preferences = [
            fact.object
            for fact in bundle.retrieved_facts
            if fact.subject == "user" and fact.relation == "prefers"
        ]
        if preferences:
            return f"Yes — you said you preferred {preferences[0]}."
        return (
            "I couldn't find a grounded preference statement about that in "
            "the retrieved history."
        )

    def _recommendation_answer(self, bundle: RetrievalBundle) -> str:
        preferences = [
            fact.object
            for fact in bundle.retrieved_facts
            if fact.subject == "user" and fact.relation == "prefers"
        ]
        role_focus = [
            fact.object
            for fact in bundle.retrieved_facts
            if fact.relation in {"role_focus", "has_role"}
        ]
        work_modes = [
            fact.object
            for fact in bundle.retrieved_facts
            if fact.relation == "work_mode"
        ]
        if preferences and role_focus:
            preferred = preferences[0].lower()
            focus_match = any(
                token in " ".join(role_focus).lower() for token in preferred.split()
            )
            if focus_match:
                detail = ", ".join(role_focus[:2])
                mode_text = f" and the role is {work_modes[0]}" if work_modes else ""
                return (
                    "Probably yes — the retrieved context says you're "
                    f"leaning toward {preferences[0]}, "
                    f"and this opportunity looks aligned ({detail}{mode_text})."
                )
            return (
                "I'd be cautious. You said you preferred "
                f"{preferences[0]}, but the retrieved details here point more toward "
                f"{', '.join(role_focus[:2]) or 'a different role profile'}."
            )
        return (
            "I don't have enough grounded detail about your preferences and "
            "this role to recommend one way or the other."
        )

    def _summary_answer(self, query: str, bundle: RetrievalBundle) -> str:
        top_turns = bundle.retrieved_turns[:2]
        if not top_turns:
            return "I don't have enough grounded context to answer that."
        if len(top_turns) == 1:
            return f"From what you told me: {top_turns[0].text}"
        return f"From what you told me: {top_turns[0].text} Also, {top_turns[1].text}"

    def _missing_relation_support(self, query: str, facts: list[Statement]) -> bool:
        lowered = query.lower()
        relation_terms = {fact.relation.lower() for fact in facts}
        if "salary" in lowered or "offer" in lowered or "comp" in lowered:
            return not any(
                term in relation_terms
                for term in {"compensation", "offer_amount", "salary"}
            )
        if "repl" in lowered:
            return not any(
                term in relation_terms
                for term in {"reply_status", "recruiter_for", "has_recruiter"}
            )
        return False

    def _retrieval_is_weak(self, bundle: RetrievalBundle) -> bool:
        best_score = bundle.retrieved_turns[0].score if bundle.retrieved_turns else 0.0
        evidence_count = len(bundle.retrieved_facts) + len(bundle.graph_neighbors)
        return best_score < 0.22 and evidence_count == 0
