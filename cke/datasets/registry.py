"""Dataset loader registry for standardized dataset ingestion."""

from __future__ import annotations

from cke.datasets.hotpot_loader import HotpotDataset
from cke.datasets.locomo_loader import LoCoMoDataset
from cke.datasets.msmarco_loader import MSMarcoDocumentDataset

DATASET_REGISTRY = {
    "hotpotqa": HotpotDataset,
    "msmarco": MSMarcoDocumentDataset,
    "locomo": LoCoMoDataset,
}


def load_dataset(name: str, path: str):
    """Instantiate a loader by name and load records from ``path``."""
    key = name.lower()
    if key not in DATASET_REGISTRY:
        available = ", ".join(sorted(DATASET_REGISTRY))
        raise ValueError(f"Unknown dataset '{name}'. Available: {available}")
    loader = DATASET_REGISTRY[key]()
    return loader.load(path)
