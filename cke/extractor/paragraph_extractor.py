"""Sliding-window paragraph extraction helpers."""

from __future__ import annotations

import re


class ParagraphExtractor:
    """Generate overlapping sentence windows for document extraction."""

    def __init__(self, window_size: int = 3) -> None:
        self.window_size = max(1, window_size)

    def sentence_windows(self, document: str) -> list[str]:
        sentences = [
            s.strip() for s in re.split(r"(?<=[.!?])\s+", document) if s.strip()
        ]
        if not sentences:
            return []
        if len(sentences) <= self.window_size:
            return [" ".join(sentences)]

        windows: list[str] = []
        for idx in range(0, len(sentences) - self.window_size + 1):
            windows.append(" ".join(sentences[idx : idx + self.window_size]))
        return windows
