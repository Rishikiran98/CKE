"""2WikiMultiHopQA dataset loader."""

from __future__ import annotations

import json
from typing import Any

from cke.datasets.base_loader import DatasetLoader
from cke.diagnostics import DegradationMixin
from cke.utils.text_cleaning import merge_sentences, normalize_whitespace


class WikiMultiHopDataset(DatasetLoader, DegradationMixin):
    """Loads 2WikiMultiHopQA JSON into the normalized CKE format.

    This carried no degradation contract while its two siblings did, so a
    malformed context entry was dropped in silence: an item could load with
    zero documents, be scored against them, and say nothing. The same fault
    in HotpotQA and MuSiQue is declared and refused under strict, and now so
    is this one.
    """

    def __init__(self, strict: bool = False) -> None:
        super().__init__()
        self._init_degradation(strict)

    def _context_to_documents(self, context: list[list[Any]]) -> list[dict[str, Any]]:
        documents: list[dict[str, Any]] = []
        malformed = 0
        for idx, ctx in enumerate(context or []):
            if len(ctx) != 2:
                malformed += 1
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
        if malformed:
            self._degrade(
                f"{malformed} of {len(context or [])} context entries were not "
                "[title, sentences] pairs and were dropped, so this item is "
                "evaluated against fewer documents than it carries"
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
