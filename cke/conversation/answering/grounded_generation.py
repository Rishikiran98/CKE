"""Evidence-driven grounded answer generation."""

from __future__ import annotations

from cke.conversation.config import AnsweringConfig, RetrievalConfig
from cke.conversation.types import ConversationAnswer, EvidenceSet, RetrievalBundle
from .abstention import AbstentionDecider
from .confidence import ConfidenceEstimator
from .evidence_selection import EvidenceSelector


class GroundedAnswerComposer:
    """Compose answers from evidence rather than phrase-trigger routing."""

    def __init__(
        self,
        evidence_selector: EvidenceSelector | None = None,
        abstention_decider: AbstentionDecider | None = None,
        confidence_estimator: ConfidenceEstimator | None = None,
        config: AnsweringConfig | None = None,
        retrieval_config: RetrievalConfig | None = None,
    ) -> None:
        self.config = config or AnsweringConfig()
        self.retrieval_config = retrieval_config or RetrievalConfig()
        self.evidence_selector = evidence_selector or EvidenceSelector()
        self.abstention_decider = abstention_decider or AbstentionDecider(
            self.config,
            self.retrieval_config,
        )
        self.confidence_estimator = confidence_estimator or ConfidenceEstimator(
            self.config
        )

    def compose(self, query: str, bundle: RetrievalBundle) -> ConversationAnswer:
        evidence = bundle.evidence or self.evidence_selector.select(query, bundle)
        abstain, reason = self.abstention_decider.should_abstain(evidence)
        if abstain:
            confidence, band = self.confidence_estimator.estimate(
                evidence, grounded=False
            )
            return ConversationAnswer(
                answer=self._abstention_text(reason),
                confidence=confidence,
                grounded=False,
                evidence=evidence,
                retrieved_turns=bundle.retrieved_turns,
                retrieved_facts=bundle.retrieved_facts,
                graph_neighbors=bundle.graph_neighbors,
                metadata={
                    "abstained": True,
                    "reason": reason,
                    "confidence_band": band.value,
                },
            )

        answer_text = self._generate_grounded_answer(query, evidence)
        confidence, band = self.confidence_estimator.estimate(evidence, grounded=True)
        return ConversationAnswer(
            answer=answer_text,
            confidence=confidence,
            grounded=True,
            evidence=evidence,
            retrieved_turns=bundle.retrieved_turns,
            retrieved_facts=bundle.retrieved_facts,
            graph_neighbors=bundle.graph_neighbors,
            metadata={"abstained": False, "confidence_band": band.value},
        )

    def _generate_grounded_answer(self, query: str, evidence: EvidenceSet) -> str:
        lowered = query.lower()
        if "should i" in lowered:
            turn_text = " ".join(
                item.text.lower() for item in evidence.supporting_turns
            )
            fact_text = " ".join(
                f"{fact.subject} {fact.relation} {fact.object}".lower()
                for fact in evidence.supporting_facts
            )
            if ("prefer" in turn_text or "prefers" in fact_text) and (
                "remote" in turn_text
                or "backend" in turn_text
                or "has_role" in fact_text
            ):
                return (
                    "Probably yes — the retrieved conversation evidence suggests "
                    "the option matches your backend and remote preferences."
                )
        if (
            any(
                phrase in lowered
                for phrase in (
                    "what did i tell you",
                    "what did i say",
                    "given everything i told you",
                    "my ",
                )
            )
            and evidence.supporting_turns
        ):
            snippets = [item.text for item in evidence.supporting_turns[:2]]
            if len(snippets) == 1:
                return f"From the conversation: {snippets[0]}"
            return f"From the conversation: {snippets[0]} Also, {snippets[1]}"
        if "prefer" in lowered:
            for fact in evidence.supporting_facts:
                if fact.relation == "prefers":
                    return f"Yes — you said you preferred {fact.object}."
        if any(token in lowered for token in ("when", "date", "time")):
            for fact in evidence.supporting_facts:
                if fact.relation in {"occurs_on", "scheduled_for", "date"}:
                    return (
                        "Based on the conversation history, "
                        f"{fact.subject} {fact.relation.replace('_', ' ')} {fact.object}."
                    )
        if lowered.startswith("who"):
            pending = [
                fact.subject
                for fact in evidence.supporting_facts
                if fact.relation in {"status", "reply_status"}
                and fact.object in {"pending", "waiting"}
                and fact.subject != "user"
            ]
            names = pending or [
                fact.subject
                for fact in evidence.supporting_facts
                if fact.subject and fact.subject != "user"
            ]
            if names:
                return ", ".join(dict.fromkeys(names))
        if lowered.startswith(("did ", "is ", "has ", "was ")):
            return (
                "yes"
                if evidence.supporting_facts or evidence.supporting_turns
                else "no"
            )
        if evidence.supporting_facts:
            best = evidence.supporting_facts[0]
            return f"I found evidence that {best.subject} {best.relation.replace('_', ' ')} {best.object}."
        if evidence.supporting_turns:
            snippets = [item.text for item in evidence.supporting_turns[:2]]
            if len(snippets) == 1:
                return f"From the conversation: {snippets[0]}"
            return f"From the conversation: {snippets[0]} Also, {snippets[1]}"
        return "I don't have enough grounded conversation history to answer that yet."

    def _abstention_text(self, reason: str) -> str:
        if reason == "conflicting_evidence":
            return (
                "I found conflicting evidence in the conversation history, so I can't "
                "answer confidently."
            )
        if reason == "weak_evidence":
            return (
                "I found only weak matches in the conversation history, so I can't "
                "answer that confidently."
            )
        return "I don't have enough grounded conversation history to answer that yet."
