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

    def normalize_record(self, index: int, record: dict[str, Any]) -> dict[str, Any]:
        """Normalise one raw 2WikiMultiHopQA record into the CKE item shape.

        Public, and named as the other loaders name it, so a caller that
        evaluates part of a file can normalise only the records it evaluates.
        Normalising the rest declares degradations for context this run never
        looked at, which under strict refuses the run.
        """
        return {
            "id": str(record.get("_id", f"wiki2_{index}")),
            "question": str(record.get("question", "")),
            "answer": str(record.get("answer", "")),
            "documents": self._context_to_documents(record.get("context", [])),
            "supporting_facts": record.get("supporting_facts", []),
            "metadata": {
                "type": record.get("type"),
                "evidences": record.get("evidences"),
            },
        }

    def load(self, path: str) -> "WikiMultiHopDataset":
        with open(path, "r", encoding="utf-8") as f:
            rows = json.load(f)

        self.items = [self.normalize_record(idx, row) for idx, row in enumerate(rows)]
        return self
