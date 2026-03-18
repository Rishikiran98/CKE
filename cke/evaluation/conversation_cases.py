"""Conversation-first evaluation scenarios for realistic memory and RAG behavior."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(slots=True)
class ConversationScenario:
    scenario_id: str
    turns: list[tuple[str, str]]
    query: str
    expected_fragments: list[str] = field(default_factory=list)
    notes: str = ""


CONVERSATION_SCENARIOS: list[ConversationScenario] = [
    ConversationScenario(
        scenario_id="memory-recall-apple",
        turns=[
            (
                "user",
                "I have an Apple interview next Tuesday for a backend engineer role.",
            ),
            ("assistant", "Got it — Apple, next Tuesday, backend engineer."),
        ],
        query="What did I tell you about my Apple interview?",
        expected_fragments=["Apple", "next Tuesday", "backend engineer"],
        notes="Semantic recall over a prior interview turn.",
    ),
    ConversationScenario(
        scenario_id="follow-up-when",
        turns=[
            ("user", "The Apple interview is on March 14."),
            ("assistant", "Noted."),
        ],
        query="When was that again?",
        expected_fragments=["March 14"],
        notes="Reference resolution for temporal follow-up.",
    ),
    ConversationScenario(
        scenario_id="pending-replies",
        turns=[
            ("user", "The recruiter from Stripe still hasn't replied."),
            ("user", "Meta already replied yesterday."),
        ],
        query="Who hasn't replied to me yet?",
        expected_fragments=["Stripe"],
        notes="Communication tracking with explicit non-reply state.",
    ),
    ConversationScenario(
        scenario_id="preference-tracking",
        turns=[
            ("user", "I preferred backend roles over frontend roles."),
            ("assistant", "I'll remember that."),
        ],
        query="Didn't I say I preferred backend roles?",
        expected_fragments=["preferred backend roles"],
        notes="Preference memory and confirmation.",
    ),
    ConversationScenario(
        scenario_id="recommendation",
        turns=[
            ("user", "I preferred backend roles and remote work."),
            (
                "user",
                "The role at Stripe is a backend engineer position and it's remote.",
            ),
        ],
        query="Given everything I told you, should I apply to this?",
        expected_fragments=["Probably yes", "backend", "remote"],
        notes="Recommendation with grounded preference alignment.",
    ),
    ConversationScenario(
        scenario_id="abstain-no-grounding",
        turns=[
            ("user", "I talked to one recruiter last week."),
        ],
        query="What salary did they offer?",
        expected_fragments=["can't answer that confidently"],
        notes="Abstention when retrieval lacks grounded support.",
    ),
]


def get_conversation_scenarios() -> list[ConversationScenario]:
    return list(CONVERSATION_SCENARIOS)
