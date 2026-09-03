"""LoCoMo conversational dataset loader."""

from __future__ import annotations

import json
from typing import Any

from cke.diagnostics import DegradationMixin
from cke.datasets.base_loader import DatasetLoader
from cke.utils.text_cleaning import normalize_whitespace


#: Keys a LoCoMo record may use for its turn list.
_TURN_KEYS = ("turns", "conversation", "messages", "dialogue")


class LoCoMoDataset(DatasetLoader, DegradationMixin):
    """Load conversation-style JSON records into normalized CKE documents."""

    def __init__(self, normalize_text: bool = True, strict: bool = False) -> None:
        self._init_degradation(strict)
        super().__init__()
        self.normalize_text = normalize_text

    def _clean_text(self, text: str) -> str:
        if self.normalize_text:
            return normalize_whitespace(text)
        return text.strip()

    def _extract_turns(self, item: dict[str, Any]) -> list[dict[str, Any]]:
        """Find the turn list, or declare that this record has none.

        A record using none of the recognised keys used to load as a valid
        item with zero documents, so it counted toward the dataset size while
        contributing nothing to retrieve from.
        """
        for key in _TURN_KEYS:
            value = item.get(key)
            if isinstance(value, list):
                return value

        self._degrade(
            f"a record uses none of the recognised turn keys "
            f"({', '.join(_TURN_KEYS)}); it loads with zero documents and "
            "still counts toward the dataset size"
        )
        return []

    def load(self, path: str) -> "LoCoMoDataset":
        with open(path, "r", encoding="utf-8") as f:
            raw_items = json.load(f)

        self.items = []
        for idx, item in enumerate(raw_items):
            conversation_id = str(
                item.get("conversation_id", item.get("id", f"conversation_{idx}"))
            )
            turns = self._extract_turns(item)

            documents: list[dict[str, Any]] = []
            turn_metadata: list[dict[str, Any]] = []
            for turn_index, turn in enumerate(turns):
                if not isinstance(turn, dict):
                    continue
                speaker = str(turn.get("speaker", turn.get("role", "unknown")))
                utterance = str(turn.get("text", turn.get("content", "")))
                text = self._clean_text(f"{speaker}: {utterance}")
                documents.append(
                    {
                        "doc_id": f"{conversation_id}_turn_{turn_index}",
                        "title": "conversation",
                        "text": text,
                    }
                )
                turn_metadata.append(
                    {
                        "conversation_id": conversation_id,
                        "turn_index": turn_index,
                        "speaker": speaker,
                    }
                )

            self.items.append(
                {
                    "id": conversation_id,
                    "question": item.get("question"),
                    "answer": item.get("answer"),
                    "documents": documents,
                    "supporting_facts": item.get("supporting_facts"),
                    "metadata": {
                        "conversation_id": conversation_id,
                        "turns": turn_metadata,
                    },
                }
            )
        return self
