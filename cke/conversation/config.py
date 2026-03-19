
"""Configuration objects for the conversational memory subsystem."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True, slots=True)
class RetentionPolicy:
    max_events_per_conversation: int = 1000
    max_canonical_memories_per_conversation: int = 500
    keep_superseded_memories: bool = True


@dataclass(frozen=True, slots=True)
class ValidationPolicy:
    min_confidence: float = 0.3
    min_span_chars: int = 3
    max_relation_tokens: int = 6
    allowed_memory_kinds: tuple[str, ...] = (
        "fact",
        "preference",
        "plan",
        "status",
        "temporal",
        "summary",
        "alias",
        "observation",
    )
    reject_placeholder_objects: tuple[str, ...] = ("something", "stuff", "thing")


@dataclass(frozen=True, slots=True)
class ConsolidationPolicy:
    duplicate_similarity_threshold: float = 0.9
    conflict_relation_families: dict[str, tuple[str, ...]] = field(
        default_factory=lambda: {
            "preference": ("prefers",),
            "status": ("status", "reply_status"),
            "temporal": ("occurs_on", "scheduled_for", "deadline"),
        }
    )
    ephemeral_kinds: tuple[str, ...] = ("observation",)
    update_relations: tuple[str, ...] = ("status", "reply_status", "occurs_on", "scheduled_for")


@dataclass(frozen=True, slots=True)
class IndexingConfig:
    enable_graph_projection: bool = True
    enable_vector_cache: bool = True


@dataclass(frozen=True, slots=True)
class RetrievalConfig:
    top_k_events: int = 5
    top_k_memories: int = 5
    top_k_graph_facts: int = 5
    recency_weight: float = 0.08
    lexical_weight: float = 0.32
    dense_weight: float = 0.6
    weak_match_threshold: float = 0.2
    stop_words: frozenset[str] = frozenset(
        {"the", "a", "an", "that", "again", "did", "i", "you", "to", "of"}
    )


@dataclass(frozen=True, slots=True)
class ResolutionConfig:
    pronouns: tuple[str, ...] = ("it", "they", "them", "he", "she", "this", "that")
    temporal_reference_tokens: tuple[str, ...] = ("when", "date", "time", "again", "then")
    generic_reference_heads: tuple[str, ...] = ("company", "person", "role", "place", "thing")


@dataclass(frozen=True, slots=True)
class SummarizationConfig:
    enable_summaries: bool = False
    summary_turn_window: int = 20


@dataclass(frozen=True, slots=True)
class AnsweringConfig:
    abstain_confidence: float = 0.1
    weak_confidence: float = 0.25
    grounded_confidence: float = 0.8
    conflict_penalty: float = 0.25


@dataclass(frozen=True, slots=True)
class ConversationConfig:
    retention: RetentionPolicy = field(default_factory=RetentionPolicy)
    validation: ValidationPolicy = field(default_factory=ValidationPolicy)
    consolidation: ConsolidationPolicy = field(default_factory=ConsolidationPolicy)
    indexing: IndexingConfig = field(default_factory=IndexingConfig)
    retrieval: RetrievalConfig = field(default_factory=RetrievalConfig)
    resolution: ResolutionConfig = field(default_factory=ResolutionConfig)
    summarization: SummarizationConfig = field(default_factory=SummarizationConfig)
    answering: AnsweringConfig = field(default_factory=AnsweringConfig)

