"""2WikiMultiHopQA dataset loader."""

from __future__ import annotations

import json
from typing import Any

from cke.datasets.base_loader import DatasetLoader
from cke.utils.text_cleaning import merge_sentences, normalize_whitespace


def flatten_contexts(contexts: list[list[Any]]) -> list[str]:
    """Convert nested context entries into extraction-ready paragraphs."""
    paragraphs: list[str] = []
    for ctx in contexts or []:
        if len(ctx) != 2:
            continue
        title, text_or_sentences = ctx
        if isinstance(text_or_sentences, list):
            body = merge_sentences([str(s) for s in text_or_sentences])
        else:
            body = str(text_or_sentences)
        paragraph = normalize_whitespace(f"{title}: {body}")
        paragraphs.append(paragraph)
    return paragraphs


class WikiMultiHopDataset(DatasetLoader):
    """Loads 2WikiMultiHopQA JSON into a simple QA+context structure."""

    def load(self, path: str) -> "WikiMultiHopDataset":
        with open(path, "r", encoding="utf-8") as f:
            rows = json.load(f)

        self.items = []
        for row in rows:
            self.items.append(
                {
                    "question": str(row.get("question", "")),
                    "answer": str(row.get("answer", "")),
                    "contexts": flatten_contexts(row.get("context", [])),
                    "supporting_facts": row.get("supporting_facts", []),
                }
            )
        return self
