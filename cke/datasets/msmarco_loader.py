"""MS MARCO full-document TSV loader."""

from __future__ import annotations

from typing import Iterator

from cke.datasets.base_loader import DatasetLoader
from cke.utils.text_cleaning import normalize_whitespace


class MSMarcoDocumentDataset(DatasetLoader):
    """Load MS MARCO document corpora from ``doc_id\ttext`` rows."""

    def __init__(self, normalize_text: bool = True) -> None:
        super().__init__()
        self.normalize_text = normalize_text

    def _clean_text(self, text: str) -> str:
        if self.normalize_text:
            return normalize_whitespace(text)
        return text.strip()

    def load(self, path: str) -> "MSMarcoDocumentDataset":
        self.items = []
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                row = line.rstrip("\n")
                if not row:
                    continue
                parts = row.split("\t", 1)
                if len(parts) != 2:
                    continue
                doc_id, text = parts
                cleaned = self._clean_text(text)
                self.items.append(
                    {
                        "id": doc_id,
                        "question": None,
                        "answer": None,
                        "documents": [
                            {
                                "doc_id": doc_id,
                                "title": None,
                                "text": cleaned,
                            }
                        ],
                        "supporting_facts": None,
                        "metadata": {},
                    }
                )
        return self

    def iter_batches(self, batch_size: int) -> Iterator[list[dict]]:
        """Yield loaded items in batches for large corpora processing."""
        if batch_size <= 0:
            raise ValueError("batch_size must be greater than 0")
        for start in range(0, len(self.items), batch_size):
            yield self.items[start : start + batch_size]
