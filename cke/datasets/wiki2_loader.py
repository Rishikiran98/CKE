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
    """Loads 2WikiMultiHopQA JSON into the normalized CKE format."""

    def _context_to_documents(self, context: list[list[Any]]) -> list[dict[str, Any]]:
        documents: list[dict[str, Any]] = []
        for idx, ctx in enumerate(context or []):
            if len(ctx) != 2:
                continue
            title, text_or_sentences = ctx
            title_str = str(title)
            if isinstance(text_or_sentences, list):
                body = merge_sentences([str(s) for s in text_or_sentences])
            else:
                body = str(text_or_sentences)
            text = normalize_whitespace(body)
            documents.append(
                {
                    "doc_id": f"{title_str}_{idx}",
                    "title": title_str,
                    "text": text,
                }
            )
        return documents

    def load(self, path: str) -> "WikiMultiHopDataset":
        with open(path, "r", encoding="utf-8") as f:
            rows = json.load(f)

        self.items = []
        for idx, row in enumerate(rows):
            item_id = str(row.get("_id", f"wiki2_{idx}"))
            self.items.append(
                {
                    "id": item_id,
                    "question": str(row.get("question", "")),
                    "answer": str(row.get("answer", "")),
                    "documents": self._context_to_documents(row.get("context", [])),
                    "supporting_facts": row.get("supporting_facts", []),
                    "metadata": {
                        "type": row.get("type"),
                        "evidences": row.get("evidences"),
                    },
                }
            )
        return self
