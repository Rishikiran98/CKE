"""Base dataset loader interface for normalized dataset ingestion."""

from __future__ import annotations

import random
from abc import ABC, abstractmethod
from typing import Any


_SECURE_RANDOM = random.SystemRandom()


class DatasetLoader(ABC):
    """Abstract base class for dataset loaders.

    All concrete dataset loaders normalize raw dataset records into the
    internal CKE schema.
    """

    def __init__(self) -> None:
        self.items: list[dict[str, Any]] = []

    @abstractmethod
    def load(self, path: str) -> "DatasetLoader":
        """Load and normalize records from ``path`` into ``self.items``."""

    def sample(self, n: int) -> list[dict[str, Any]]:
        """Return up to ``n`` random normalized items from the dataset."""
        if n <= 0 or not self.items:
            return []
        if n >= len(self.items):
            return list(self.items)
        return _SECURE_RANDOM.sample(self.items, n)

    def get_item(self, index: int) -> dict[str, Any]:
        """Return the normalized item at ``index``."""
        return self.items[index]

    def __len__(self) -> int:
        """Return count of loaded items."""
        return len(self.items)
