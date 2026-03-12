"""Dataset loader registry for standardized dataset ingestion."""

from __future__ import annotations

from cke.datasets.hotpot_loader import HotpotDataset
from cke.datasets.locomo_loader import LoCoMoDataset
from cke.datasets.msmarco_loader import MSMarcoDocumentDataset
from cke.datasets.wiki2_loader import WikiMultiHopDataset

DATASET_REGISTRY = {
    "hotpotqa": HotpotDataset,
    "msmarco": MSMarcoDocumentDataset,
    "locomo": LoCoMoDataset,
    "2wikimultihopqa": WikiMultiHopDataset,
    "wiki2": WikiMultiHopDataset,
}


def load_dataset(name: str, path: str):
    """Instantiate a loader by name and load records from ``path``."""
    key = name.lower()
    if key not in DATASET_REGISTRY:
        available = ", ".join(sorted(DATASET_REGISTRY))
        raise ValueError(f"Unknown dataset '{name}'. Available: {available}")

    loader = DATASET_REGISTRY[key]()
    return loader.load(path)