"""Generic normalization and parsing helpers for conversational memory."""

from __future__ import annotations

import re

from cke.models import Statement

MONTHS_PATTERN = (
    "january|february|march|april|may|june|july|august|"
    "september|october|november|december"
)
WEEKDAYS_PATTERN = "monday|tuesday|wednesday|thursday|friday|saturday|sunday"
DATE_PATTERNS: tuple[str, ...] = (
    rf"\b(?:{WEEKDAYS_PATTERN})\b",
    rf"\b(?:{MONTHS_PATTERN})\s+\d{{1,2}}(?:st|nd|rd|th)?\b",
    r"\b\d{4}-\d{2}-\d{2}\b",
    r"\bnext\s+(?:week|month|monday|tuesday|wednesday|thursday|friday|year)\b",
    r"\btomorrow\b|\byesterday\b|\btoday\b",
)


def normalize_text_token(text: str) -> str:
    return re.sub(r"\s+", " ", text.strip(" .,!?:;\"'()[]{}")).strip()


def normalize_relation_label(text: str) -> str:
    cleaned = normalize_text_token(text).lower()
    return "_".join(token for token in re.split(r"[^a-z0-9]+", cleaned) if token)


def normalize_fact_parts(
    subject: str, relation: str, object_: str
) -> tuple[str, str, str]:
    return (
        normalize_text_token(subject),
        normalize_relation_label(relation),
        normalize_text_token(object_),
    )


def normalized_fact_key(statement: Statement) -> tuple[str, str, str]:
    return normalize_fact_parts(statement.subject, statement.relation, statement.object)


def extract_date_phrases(text: str) -> list[tuple[int, int, str]]:
    matches: list[tuple[int, int, str]] = []
    for pattern in DATE_PATTERNS:
        for match in re.finditer(pattern, text, flags=re.IGNORECASE):
            matches.append((match.start(), match.end(), match.group(0)))
    matches.sort(key=lambda item: item[0])
    return matches


def extract_date_phrase(text: str) -> str | None:
    matches = extract_date_phrases(text)
    return matches[0][2] if matches else None


def tokenize(text: str) -> list[str]:
    return re.findall(r"[a-z0-9]+", text.lower())


def lexical_overlap(
    left: str, right: str, *, stop_words: set[str] | frozenset[str] | None = None
) -> float:
    ignored = set(stop_words or set())
    left_terms = {token for token in tokenize(left) if token not in ignored}
    right_terms = {token for token in tokenize(right) if token not in ignored}
    if not left_terms:
        return 0.0
    return len(left_terms & right_terms) / len(left_terms)
