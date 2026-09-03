"""HotpotQA dataset loader."""

from __future__ import annotations

import json
from typing import Any

from cke.diagnostics import DegradationMixin
from cke.datasets.base_loader import DatasetLoader
from cke.utils.text_cleaning import merge_sentences, normalize_whitespace


class HotpotDataset(DatasetLoader, DegradationMixin):
    """Loads official HotpotQA JSON into the normalized CKE format."""

    def __init__(
        self,
        merge_context_sentences: bool = True,
        normalize_text: bool = True,
        strict: bool = False,
    ) -> None:
        super().__init__()
        self._init_degradation(strict)
        self.merge_context_sentences = merge_context_sentences
        self.normalize_text = normalize_text

    def _clean_text(self, text: str) -> str:
        if self.normalize_text:
            return normalize_whitespace(text)
        return text.strip()

    def _context_to_documents(self, context: list[list[Any]]) -> list[dict[str, Any]]:
        documents: list[dict[str, Any]] = []
        malformed = 0
        for idx, ctx in enumerate(context):
            if len(ctx) != 2:
                # A dropped paragraph changes what the item was evaluated
                # against without changing anything a reader can see.
                malformed += 1
                continue
            title, sentences = ctx
            title_str = str(title)
            if not isinstance(sentences, list):
                sentences = [str(sentences)]
            sentence_list = [str(sentence) for sentence in sentences]
            text = (
                merge_sentences(sentence_list)
                if self.merge_context_sentences
                else "\n".join(sentence_list)
            )
            text = self._clean_text(text)
            documents.append(
                {
                    "doc_id": f"{title_str}_{idx}",
                    "title": title_str,
                    "text": text,
                }
            )
        if malformed:
            self._degrade(
                f"{malformed} of {len(context)} context entries were not "
                "[title, sentences] pairs and were dropped, so this item is "
                "evaluated against fewer documents than it carries"
            )
        return documents

    def normalize_record(self, index: int, record: dict[str, Any]) -> dict[str, Any]:
        """Normalise one raw HotpotQA record into the CKE item shape.

        Public so that a caller evaluating only a prefix of a file can stop
        normalising, and therefore stop declaring, at the last record it
        evaluates. A malformed context in a record past that point is not
        part of the evaluation and must not refuse a strict run.
        """
        question = record.get("question")
        answer = record.get("answer")
        return {
            "id": str(record.get("_id", f"hotpot_{index}")),
            "question": (
                self._clean_text(str(question)) if question is not None else None
            ),
            "answer": self._clean_text(str(answer)) if answer is not None else None,
            "documents": self._context_to_documents(record.get("context", [])),
            "supporting_facts": record.get("supporting_facts", []),
            "metadata": {
                "type": record.get("type"),
                "level": record.get("level"),
            },
        }

    def load(self, path: str) -> "HotpotDataset":
        with open(path, "r", encoding="utf-8") as f:
            raw_items = json.load(f)

        self.items = [
            self.normalize_record(idx, item) for idx, item in enumerate(raw_items)
        ]
        return self
