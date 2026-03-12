"""Text preprocessing helpers shared by dataset loaders."""

from __future__ import annotations

import re

_WHITESPACE_RE = re.compile(r"\s+")


def normalize_whitespace(text: str) -> str:
    """Collapse repeated whitespace and trim edges."""
    return _WHITESPACE_RE.sub(" ", text).strip()


def merge_sentences(sentences: list[str]) -> str:
    """Merge sentence chunks into a single paragraph and normalize spacing."""
    return normalize_whitespace(" ".join(sentence.strip() for sentence in sentences))
