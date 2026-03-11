"""HotpotQA dataset loader."""

from __future__ import annotations

import json
from typing import Any

from cke.datasets.base_loader import DatasetLoader
from cke.utils.text_cleaning import merge_sentences, normalize_whitespace


class HotpotDataset(DatasetLoader):
    """Loads official HotpotQA JSON into the normalized CKE format."""

    def __init__(
        self,
        merge_context_sentences: bool = True,
        normalize_text: bool = True,
    ) -> None:
        super().__init__()
        self.merge_context_sentences = merge_context_sentences
        self.normalize_text = normalize_text

    def _clean_text(self, text: str) -> str:
        if self.normalize_text:
            return normalize_whitespace(text)
        return text.strip()

    def _context_to_documents(self, context: list[list[Any]]) -> list[dict[str, Any]]:
        documents: list[dict[str, Any]] = []
        for idx, ctx in enumerate(context):
            if len(ctx) != 2:
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
        return documents

    def load(self, path: str) -> "HotpotDataset":
        with open(path, "r", encoding="utf-8") as f:
            raw_items = json.load(f)

        self.items = []
        for idx, item in enumerate(raw_items):
            item_id = str(item.get("_id", f"hotpot_{idx}"))
            question = item.get("question")
            answer = item.get("answer")
            normalized = {
                "id": item_id,
                "question": (
                    self._clean_text(str(question)) if question is not None else None
                ),
                "answer": self._clean_text(str(answer)) if answer is not None else None,
                "documents": self._context_to_documents(item.get("context", [])),
                "supporting_facts": item.get("supporting_facts", []),
                "metadata": {
                    "type": item.get("type"),
                    "level": item.get("level"),
                },
            }
            self.items.append(normalized)
        return self
