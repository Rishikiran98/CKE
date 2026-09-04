"""MuSiQue multi-hop QA loader.

MuSiQue differs from HotpotQA and 2WikiMultiHopQA in a way that matters for
retrieval evaluation: its supporting labels are per paragraph, not per
sentence. Every item carries twenty paragraphs of which two to four are
marked ``is_supporting``, so a recall figure computed from them is a
document-level figure and is comparable across the three datasets only at
that granularity.

Answer aliases are carried in the item's metadata and are not scored against.
MuSiQue's own evaluation accepts an alias as correct; HotpotQA has no aliases
at all, so scoring one dataset with them and the others without would make a
cross-dataset accuracy column mean two different things. Whether to use them
is a scoring decision that belongs with the metric, not with the loader.
"""

from __future__ import annotations

import json
from typing import Any

from cke.datasets.base_loader import DatasetLoader
from cke.diagnostics import DegradationMixin
from cke.utils.text_cleaning import normalize_whitespace


class MuSiQueDataset(DatasetLoader, DegradationMixin):
    """Loads MuSiQue JSON into the normalized CKE format."""

    def __init__(self, normalize_text: bool = True, strict: bool = False) -> None:
        super().__init__()
        self._init_degradation(strict)
        self.normalize_text = normalize_text

    def _clean_text(self, text: str) -> str:
        if self.normalize_text:
            return normalize_whitespace(text)
        return text.strip()

    def _paragraphs_to_documents(
        self, paragraphs: list[dict[str, Any]]
    ) -> list[dict[str, Any]]:
        """One document per paragraph, keeping the paragraph's own index.

        The index is kept in the doc_id because MuSiQue titles repeat within
        an item: two paragraphs of the same article appear side by side, and
        a doc_id built from the title alone would collide.
        """
        documents: list[dict[str, Any]] = []
        malformed = 0
        for position, paragraph in enumerate(paragraphs or []):
            if not isinstance(paragraph, dict):
                malformed += 1
                continue
            text = self._clean_text(str(paragraph.get("paragraph_text", "")))
            if not text:
                malformed += 1
                continue
            title = str(paragraph.get("title", ""))
            idx = paragraph.get("idx", position)
            documents.append(
                {
                    "doc_id": f"{title}_{idx}",
                    "title": title,
                    "text": text,
                }
            )
        if malformed:
            self._degrade(
                f"{malformed} of {len(paragraphs or [])} paragraphs carried no "
                "text or were not objects and were dropped, so this item is "
                "evaluated against fewer documents than it carries"
            )
        return documents

    @staticmethod
    def _supporting_facts(paragraphs: list[dict[str, Any]]) -> list[list[Any]]:
        """The supporting paragraphs, as ``[title, idx]`` pairs.

        Shaped like HotpotQA's ``[title, sent_id]`` so that one recall
        computation reads all three datasets. The second element is a
        paragraph index here and a sentence index there; both name a span
        within the titled document, and recall is measured on the title.
        """
        return [
            [str(p.get("title", "")), p.get("idx", position)]
            for position, p in enumerate(paragraphs or [])
            if isinstance(p, dict) and p.get("is_supporting")
        ]

    def normalize_record(self, index: int, record: dict[str, Any]) -> dict[str, Any]:
        """Normalise one raw MuSiQue record into the CKE item shape."""
        paragraphs = record.get("paragraphs") or []
        answer = record.get("answer")
        question = record.get("question")
        return {
            "id": str(record.get("id", f"musique_{index}")),
            "question": (
                self._clean_text(str(question)) if question is not None else None
            ),
            "answer": self._clean_text(str(answer)) if answer is not None else None,
            "documents": self._paragraphs_to_documents(paragraphs),
            "supporting_facts": self._supporting_facts(paragraphs),
            "metadata": {
                # Not scored against; see the module docstring.
                "answer_aliases": list(record.get("answer_aliases") or []),
                "answerable": record.get("answerable"),
                # "2hop__460946_294723" names the hop count in its id.
                "hops": str(record.get("id", "")).split("__")[0] or None,
            },
        }

    def load(self, path: str) -> "MuSiQueDataset":
        with open(path, "r", encoding="utf-8") as f:
            raw_items = json.load(f)

        self.items = [
            self.normalize_record(idx, item) for idx, item in enumerate(raw_items)
        ]
        return self
