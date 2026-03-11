"""Regex-based extraction of simple relations."""

from __future__ import annotations

import re

from cke.models import Statement


class RuleExtractor:
    """Extract simple assertion patterns from free text."""

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
                subject = self._clean(match.group("s"))
                obj = self._clean(match.group("o"))
                if subject and obj:
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
