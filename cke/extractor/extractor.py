"""Semantic extraction layer.

The default implementation is rule-based but follows an interface that can be
replaced by an LLM extractor later.
"""

from __future__ import annotations

import re
from abc import ABC, abstractmethod
from typing import Iterable, List

from cke.models import Statement


class BaseExtractor(ABC):
    """Extractor interface for converting text into structured statements."""

    @abstractmethod
    def extract(self, text: str) -> List[Statement]:
        """Extract knowledge statements from raw text."""


class RuleBasedExtractor(BaseExtractor):
    """Simple pattern-based extractor for prototype graph ingestion."""

    RELATION_PATTERNS: tuple[tuple[re.Pattern[str], str], ...] = (
        (
            re.compile(
                r"(?P<s>[A-Za-z0-9_\-/ ]+)\s+supports\s+"
                r"(?P<o>[A-Za-z0-9_\-/ ]+)",
                re.I,
            ),
            "supports",
        ),
        (
            re.compile(
                r"(?P<s>[A-Za-z0-9_\-/ ]+)\s+uses\s+(?P<o>[A-Za-z0-9_\-/ ]+)",
                re.I,
            ),
            "uses",
        ),
        (
            re.compile(
                r"(?P<s>[A-Za-z0-9_\-/ ]+)\s+implemented[_ ]via\s+"
                r"(?P<o>[A-Za-z0-9_\-/ ]+)",
                re.I,
            ),
            "implemented_via",
        ),
    )

    def extract(self, text: str) -> List[Statement]:
        statements: list[Statement] = []
        for sentence in self._split_sentences(text):
            for pattern, relation in self.RELATION_PATTERNS:
                match = pattern.search(sentence)
                if not match:
                    continue
                subject = self._normalize_token(match.group("s"))
                obj = self._normalize_token(match.group("o"))
                if subject and obj:
                    statements.append(Statement(subject, relation, obj))
        return statements

    @staticmethod
    def _split_sentences(text: str) -> Iterable[str]:
        return [
            seg.strip() for seg in re.split(r"[.?!]\s*", text) if seg.strip()
        ]

    @staticmethod
    def _normalize_token(token: str) -> str:
        cleaned = re.sub(r"\s+", " ", token.strip(" ,;"))
        cleaned = re.sub(
            r"\b(protocol|messaging)\b$", "", cleaned, flags=re.I
        ).strip()
        return cleaned
