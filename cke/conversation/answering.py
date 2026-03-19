"""Grounded answer composition for conversational memory and RAG flows."""

from __future__ import annotations

from cke.conversation.config import AnsweringConfig, RetrievalConfig
from cke.conversation.patterns import extract_date_phrase
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
        config: AnsweringConfig | None = None,
        retrieval_config: RetrievalConfig | None = None,
    ) -> None:
        self.operator_selector = operator_selector or OperatorSelector()
        self.operator_executor = operator_executor or OperatorExecutor()
        self.config = config or AnsweringConfig()
        self.retrieval_config = retrieval_config or RetrievalConfig()

    def compose(self, query: str, bundle: RetrievalBundle) -> ConversationAnswer:
        evidence_facts = bundle.retrieved_facts or bundle.graph_neighbors
        if not bundle.retrieved_turns:
            return ConversationAnswer(
                answer=(
                    "I don't have enough grounded conversation history "
                    "to answer that yet."
                ),
                confidence=self.config.no_history_confidence,
                grounded=False,
                metadata={
                    "abstained": True,
                    "reason": "no_retrieval",
                    "confidence_band": self._confidence_band(
                        self.config.no_history_confidence
                    ),
                },
            )

        if self._retrieval_is_weak(bundle) or self._missing_relation_support(
            query, evidence_facts
        ):
            return ConversationAnswer(
                answer=(
                    "I found only weak matches in the conversation history, "
                    "so I can't answer that confidently."
                ),
                confidence=self.config.weak_answer_confidence,
                grounded=False,
                retrieved_turns=bundle.retrieved_turns,
                retrieved_facts=evidence_facts,
                graph_neighbors=bundle.graph_neighbors,
                metadata={
                    "abstained": True,
                    "reason": "weak_retrieval_or_missing_support",
                    "confidence_band": self._confidence_band(
                        self.config.weak_answer_confidence
                    ),
                },
            )

        operator_answer = self._operator_answer(query, evidence_facts)
        if operator_answer is not None:
            confidence = self.config.operator_answer_confidence
            return ConversationAnswer(
                answer=operator_answer,
                confidence=confidence,
                grounded=True,
                retrieved_turns=bundle.retrieved_turns,
                retrieved_facts=evidence_facts,
                graph_neighbors=bundle.graph_neighbors,
                metadata={
                    "answer_mode": "operator_augmented",
                    "confidence_band": self._confidence_band(confidence),
                },
            )

        lowered = query.lower()
        if any(token in lowered for token in self.config.when_query_tokens):
            answer = self._when_answer(bundle)
        elif any(token in lowered for token in self.config.pending_reply_tokens):
            answer = self._pending_reply_answer(bundle)
        elif any(
            token in lowered for token in self.config.preference_confirmation_tokens
        ):
            answer = self._preference_confirmation(bundle)
        elif any(token in lowered for token in self.config.recommendation_tokens):
            answer = self._recommendation_answer(bundle)
        else:
            answer = self._summary_answer(query, bundle)

        grounded = not answer.startswith("I don't") and not answer.startswith(
            "I found only weak"
        )
        confidence = (
            self.config.grounded_answer_confidence
            if grounded
            else self.config.fallback_grounded_confidence
        )
        return ConversationAnswer(
            answer=answer,
            confidence=confidence,
            grounded=grounded,
            retrieved_turns=bundle.retrieved_turns,
            retrieved_facts=evidence_facts,
            graph_neighbors=bundle.graph_neighbors,
            metadata={
                "answer_mode": "grounded_natural_language",
                "confidence_band": self._confidence_band(confidence),
            },
        )

    def _detect_operator_hint(self, query: str) -> str | None:
        lowered = f" {query.lower()} "
        for operator_name, phrases in self.config.operator_hint_map.items():
            if any(phrase in lowered for phrase in phrases):
                return operator_name
        if lowered.strip().startswith(self.config.existence_prefixes):
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
        if outcome is None or outcome.result_value is None:
            return None
        result = outcome.result_value
        if isinstance(result, bool):
            return "yes" if result else "no"
        return str(result)

    def _when_answer(self, bundle: RetrievalBundle) -> str:
        for turn in bundle.retrieved_turns:
            date_value = extract_date_phrase(turn.text)
            if date_value:
                return f"You said it was {date_value}."
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
            return not any(term in relation_terms for term in {"compensation", "offer"})
        if "reply" in lowered or "replied" in lowered:
            return "reply_status" not in relation_terms
        if "when" in lowered or "date" in lowered:
            return not any(term in relation_terms for term in {"scheduled_for", "date"})
        return False

    def _retrieval_is_weak(self, bundle: RetrievalBundle) -> bool:
        best_score = max((item.score for item in bundle.retrieved_turns), default=0.0)
        return best_score < self.retrieval_config.weak_match_threshold

    def _confidence_band(self, confidence: float) -> str:
        if confidence >= 0.75:
            return "high"
        if confidence >= 0.4:
            return "medium"
        return "low"
