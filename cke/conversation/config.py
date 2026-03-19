"""Configuration objects for conversational memory heuristics."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True, slots=True)
class ExtractionConfig:
    """Configurable extraction patterns and lightweight domain hints."""

    extracted_roles: tuple[str, ...] = ("user",)
    statement_confidence: float = 0.88
    pronouns_to_ignore: frozenset[str] = frozenset(
        {"i", "we", "they", "it", "he", "she"}
    )
    nounish_entity_patterns: tuple[str, ...] = (
        r"\b(backend roles?|frontend roles?|platform roles?|infra roles?)\b",
        r"\b(Apple interview|Google interview|Meta interview|Stripe interview)\b",
        r"\b(recruiter|hiring manager|onsite|phone screen|take-home)\b",
    )
    interview_tokens: tuple[str, ...] = ("interview", "onsite", "screen")
    application_tokens: tuple[str, ...] = ("apply", "applying", "application")
    pending_reply_tokens: tuple[str, ...] = (
        "hasn't replied",
        "hasnt replied",
        "didn't reply",
        "didnt reply",
        "waiting on",
        "still waiting",
    )
    replied_tokens: tuple[str, ...] = ("replied", "got back", "responded")
    preference_tokens: tuple[str, ...] = (
        "prefer",
        "preferred",
        "i'm leaning",
        "i am leaning",
    )
    work_modes: tuple[str, ...] = ("remote", "hybrid", "onsite")
    role_flavors: tuple[str, ...] = (
        "backend",
        "frontend",
        "platform",
        "infra",
        "full stack",
        "full-stack",
    )
    compensation_tokens: tuple[str, ...] = ("salary", "comp", "pays", "pay")
    process_update_tokens: tuple[str, ...] = (
        "faster",
        "slower",
        "moved",
        "rescheduled",
    )
    company_leading_prepositions: tuple[str, ...] = ("at", "with", "from", "for")


@dataclass(frozen=True, slots=True)
class RetrievalConfig:
    """Heuristic retrieval weights for hybrid ranking."""

    recency_weight: float = 0.1
    entity_match_weight: float = 0.08
    keyword_overlap_weight: float = 0.25
    fact_topic_bonus: float = 0.05
    weak_match_threshold: float = 0.22
    stop_words: frozenset[str] = frozenset(
        {"the", "a", "an", "that", "again", "did", "i", "you"}
    )
    boosted_fact_topics: frozenset[str] = frozenset(
        {"preference", "timeline", "communication"}
    )


@dataclass(frozen=True, slots=True)
class AnsweringConfig:
    """Intent phrases and confidence bands for answer composition."""

    no_history_confidence: float = 0.0
    weak_answer_confidence: float = 0.2
    grounded_answer_confidence: float = 0.8
    operator_answer_confidence: float = 0.84
    fallback_grounded_confidence: float = 0.25
    when_query_tokens: tuple[str, ...] = ("when", "what date", "when was")
    pending_reply_tokens: tuple[str, ...] = (
        "who hasn't replied",
        "who hasnt replied",
        "who has not replied",
    )
    preference_confirmation_tokens: tuple[str, ...] = (
        "preferred backend",
        "didn't i say i preferred",
        "didnt i say",
    )
    recommendation_tokens: tuple[str, ...] = ("should i apply",)
    operator_hint_map: dict[str, tuple[str, ...]] = field(
        default_factory=lambda: {
            "count": (" how many",),
            "equality": (" same ", " equal ", " identical "),
            "temporal_compare": (
                " before ",
                " after ",
                " later ",
                " earlier ",
                " when ",
            ),
        }
    )
    existence_prefixes: tuple[str, ...] = ("did ", "is ", "has ", "was ")


@dataclass(frozen=True, slots=True)
class ReferenceResolutionConfig:
    """Configurable shorthand reference phrases and date trigger words."""

    explicit_reference_map: dict[str, str] = field(
        default_factory=lambda: {
            "that company": "company",
            "the company": "company",
            "that recruiter": "person",
            "the recruiter": "person",
            "that role": "role",
            "the role": "role",
        }
    )
    company_pronouns: tuple[str, ...] = ("it", "they", "them")
    date_context_tokens: tuple[str, ...] = ("when", "date", "again")
