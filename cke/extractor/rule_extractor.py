"""Regex-based extraction of simple relations."""

from __future__ import annotations

import re

from cke.models import Statement


class RuleExtractor:
    """Extract simple assertion patterns from free text."""

    MAX_OBJECT_LENGTH = 80
    GENERIC_RELATIONS: frozenset[str] = frozenset({"is_a"})

    PATTERNS: tuple[tuple[re.Pattern[str], str], ...] = (
        (re.compile(r"(?P<s>[^.]+?)\s+is\s+a\s+(?P<o>[^.]+)", re.I), "is_a"),
        (re.compile(r"(?P<s>[^.]+?)\s+uses\s+(?P<o>[^.]+)", re.I), "uses"),
        (re.compile(r"(?P<s>[^.]+?)\s+developed\s+(?P<o>[^.]+)", re.I), "developed"),
        (
            re.compile(r"(?P<s>[^.]+?)\s+located\s+in\s+(?P<o>[^.]+)", re.I),
            "located_in",
        ),
    )

    def extract(self, text: str) -> list[Statement]:
        assertions: list[Statement] = []
        for sentence in self._split_sentences(text):
            for pattern, relation in self.PATTERNS:
                match = pattern.search(sentence)
                if not match:
                    continue
                subject = self._normalize_entity(match.group("s"))
                obj = self._normalize_object(match.group("o"))
                if not self._is_valid_statement(subject, relation, obj):
                    continue
                assertions.append(
                    Statement(
                        subject=subject,
                        relation=relation,
                        object=obj,
                        confidence=1.0,
                    )
                )
        return assertions

    @staticmethod
    def _split_sentences(text: str) -> list[str]:
        return [s.strip() for s in re.split(r"[.!?]\s*", text) if s.strip()]

    @staticmethod
    def _clean(token: str) -> str:
        return re.sub(r"\s+", " ", token.strip(" ,;\n\t"))

    def _normalize_entity(self, token: str) -> str:
        cleaned = self._clean(token)
        return cleaned.strip("\"'()[]")

    def _normalize_object(self, token: str) -> str:
        cleaned = self._clean(token)
        cleaned = cleaned.strip("\"'()[]")
        return cleaned[: self.MAX_OBJECT_LENGTH].strip()

    def _is_valid_statement(self, subject: str, relation: str, obj: str) -> bool:
        if not subject or not obj:
            return False
        if relation in self.GENERIC_RELATIONS:
            return False
        return True
