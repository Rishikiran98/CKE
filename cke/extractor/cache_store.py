"""JSONL-backed cache for extraction outputs."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any


class ExtractionCache:
    """Disk-backed extraction cache keyed by SHA256 text hash."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._store: dict[str, list[dict[str, Any]]] = {}
        if self.path.exists():
            self._load_existing()

    @staticmethod
    def hash_text(text: str) -> str:
        return hashlib.sha256(text.encode("utf-8")).hexdigest()

    def get(self, text_hash: str) -> list[dict[str, Any]] | None:
        return self._store.get(text_hash)

    def set(self, text_hash: str, assertions: list[dict[str, Any]]) -> None:
        self._store[text_hash] = assertions
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(
                json.dumps({"text_hash": text_hash, "assertions": assertions}) + "\n"
            )

    def _load_existing(self) -> None:
        with self.path.open("r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                payload = json.loads(line)
                text_hash = str(payload.get("text_hash", ""))
                assertions = payload.get("assertions", [])
                if text_hash:
                    self._store[text_hash] = assertions
