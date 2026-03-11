"""Dataset interfaces for CKE."""

from cke.datasets.base_loader import DatasetLoader
from cke.datasets.hotpot_loader import HotpotDataset
from cke.datasets.locomo_loader import LoCoMoDataset
from cke.datasets.msmarco_loader import MSMarcoDocumentDataset
from cke.datasets.registry import DATASET_REGISTRY, load_dataset

__all__ = [
    "DatasetLoader",
    "HotpotDataset",
    "LoCoMoDataset",
    "MSMarcoDocumentDataset",
    "WikiMultiHopDataset",
    "flatten_contexts",
    "DATASET_REGISTRY",
    "load_dataset",
]
