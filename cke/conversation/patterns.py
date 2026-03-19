"""Shared regex patterns and normalization helpers for conversation modules."""

from __future__ import annotations

import re

from cke.models import Statement

MONTHS_PATTERN = (
    "january|february|march|april|may|june|july|august|"
    "september|october|november|december"
)
WEEKDAYS_PATTERN = "monday|tuesday|wednesday|thursday|friday|saturday|sunday"
ROLE_PATTERN = (
    r"\b((?:backend|frontend|platform|infra|full stack|full-stack|ml|data)"
    r"\s+(?:engineer|developer)(?:\s+role|\s+position)?|"
    r"(?:backend|frontend|platform|infra|full stack|full-stack|ml|data)"
    r"\s+roles?)\b"
)
DATE_PATTERNS: tuple[str, ...] = (
    rf"\b(?:{WEEKDAYS_PATTERN})\b",
    rf"\b(?:{MONTHS_PATTERN})\s+\d{{1,2}}(?:st|nd|rd|th)?\b",
    r"\b\d{4}-\d{2}-\d{2}\b",
    r"\bnext\s+(?:week|month|monday|tuesday|wednesday|thursday|friday)\b",
)


def extract_date_phrase(text: str) -> str | None:
    """Return the first supported date-like span from free text."""
    for pattern in DATE_PATTERNS:
        match = re.search(pattern, text, flags=re.IGNORECASE)
        if match:
            return match.group(0)
    return None


def normalize_text_token(text: str) -> str:
    """Normalize lightweight entity/value text for dedupe and graph insertion."""
    return re.sub(r"\s+", " ", text.strip(" .,!?:;\"'")).strip()


def normalize_fact_parts(
    subject: str, relation: str, object_: str
) -> tuple[str, str, str]:
    """Return normalized statement parts while preserving display casing."""
    return (
        normalize_text_token(subject),
        normalize_text_token(relation).lower(),
        normalize_text_token(object_),
    )


def normalized_fact_key(statement: Statement) -> tuple[str, str, str]:
    """Stable normalized key for fact deduplication."""
    return normalize_fact_parts(statement.subject, statement.relation, statement.object)
