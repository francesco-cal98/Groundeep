"""
DatasetManager: Clean dataset loading and feature extraction.

Replaces the dataset-handling parts of Embedding_analysis with:
- Lazy loading (only creates what you need)
- Clear API
- Separation of concerns
"""

import sys
from pathlib import Path
from typing import Dict, Optional, Literal
import numpy as np
from torch.utils.data import DataLoader

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.datasets.uniform_dataset import create_dataloaders_uniform, create_dataloaders_zipfian


class DatasetManager:
    """
    Manages dataset loading for uniform and/or zipfian distributions.

    Features:
    - Lazy loading: only creates dataloaders when needed
    - Flexible: load only what you need (uniform or zipfian)
    - Clean API: explicit method calls instead of side effects

    Example:
        >>> dm = DatasetManager("path/to/data", "dataset.npz")
        >>> val_loader = dm.get_dataloader("uniform", split="val")
        >>> features = dm.get_features("uniform", split="val")
        >>> labels = features["labels"]
    """

    def __init__(
        self,
        dataset_path: str,
        dataset_name: str,
        default_val_size: float = 0.05,
        default_test_size: float = 0.2,
        default_batch_size: int = 128,
        num_workers: int = 4,
    ):
        """
        Initialize the dataset manager.

        Args:
            dataset_path: Path to dataset directory
            dataset_name: Name of .npz file
            default_val_size: Default validation split size
            default_test_size: Default test split size
            default_batch_size: Default batch size for dataloaders
            num_workers: Number of worker processes
        """
        self.dataset_path = dataset_path
        self.dataset_name = dataset_name
        self.default_val_size = default_val_size
        self.default_test_size = default_test_size
        self.default_batch_size = default_batch_size
        self.num_workers = num_workers

        # Lazy storage
        self._dataloaders: Dict[str, Dict[str, DataLoader]] = {}
        self._datasets = {}

    def get_dataloader(
        self,
        distribution: Literal["uniform", "zipfian"],
        split: Literal["train", "val", "test"] = "val",
        batch_size: Optional[int] = None,
        val_size: Optional[float] = None,
        test_size: Optional[float] = None,
        full_batch: bool = False,  # For validation: single batch with all data
    ) -> DataLoader:
        """
        Get a specific dataloader.

        Args:
            distribution: 'uniform' or 'zipfian'
            split: 'train', 'val', or 'test'
            batch_size: Override default batch size
            val_size: Override default val split size
            test_size: Override default test split size
            full_batch: If True, return entire split as single batch (for val)

        Returns:
            DataLoader for the requested configuration
        """
        # Create dataloaders if not already loaded
        if distribution not in self._dataloaders:
            self._create_dataloaders(
                distribution,
                val_size=val_size or self.default_val_size,
                test_size=test_size or self.default_test_size,
                batch_size=batch_size or self.default_batch_size,
            )

        loader = self._dataloaders[distribution][split]

        # Convert to full batch if requested (common for validation)
        if full_batch and split in ("val", "test"):
            loader = DataLoader(
                loader.dataset,
                batch_size=len(loader.dataset),
                shuffle=False,
                num_workers=0,  # Single batch, no workers needed
            )

        return loader

    def get_features(
        self,
        distribution: Literal["uniform", "zipfian"],
        split: Literal["train", "val", "test"] = "val",
    ) -> Dict[str, np.ndarray]:
        """
        Extract features from a dataset split.

        Args:
            distribution: 'uniform' or 'zipfian'
            split: Which split to extract from

        Returns:
            Dictionary with keys: 'labels', 'cum_area', 'convex_hull', 'density' (optional)
        """
        # Ensure dataloaders are created
        if distribution not in self._dataloaders:
            self._create_dataloaders(distribution)

        # Get the subset and base dataset
        loader = self._dataloaders[distribution][split]
        subset = loader.dataset
        base_dataset = subset.dataset  # Unwrap Subset to get UniformDataset

        # Get indices for this split
        indices = subset.indices

        # Extract features
        features = {
            "labels": np.array([base_dataset.labels[i] for i in indices]),
            "cum_area": np.array([base_dataset.cumArea_list[i] for i in indices]),
            "convex_hull": np.array([base_dataset.CH_list[i] for i in indices]),
        }

        # Optional density feature
        if base_dataset.density_list is not None:
            features["density"] = np.array([base_dataset.density_list[i] for i in indices])

        # Optional mean/std item size
        if base_dataset.mean_item_size_list is not None:
            features["mean_item_size"] = np.array(
                [base_dataset.mean_item_size_list[i] for i in indices]
            )
        if base_dataset.std_item_size_list is not None:
            features["std_item_size"] = np.array(
                [base_dataset.std_item_size_list[i] for i in indices]
            )

        return features

    def get_raw_dataset(self, distribution: Literal["uniform", "zipfian"]) -> any:
        """
        Get the raw UniformDataset object (for advanced use).

        Args:
            distribution: 'uniform' or 'zipfian'

        Returns:
            The underlying UniformDataset instance
        """
        if distribution not in self._dataloaders:
            self._create_dataloaders(distribution)

        loader = self._dataloaders[distribution]["val"]
        return loader.dataset.dataset

    def _create_dataloaders(
        self,
        distribution: str,
        val_size: float = None,
        test_size: float = None,
        batch_size: int = None,
    ):
        """
        Create train/val/test dataloaders for a distribution.

        Args:
            distribution: 'uniform' or 'zipfian'
            val_size: Validation split size
            test_size: Test split size
            batch_size: Batch size
        """
        val_size = val_size or self.default_val_size
        test_size = test_size or self.default_test_size
        batch_size = batch_size or self.default_batch_size

        if distribution == "uniform":
            train_loader, val_loader, test_loader = create_dataloaders_uniform(
                data_path=self.dataset_path,
                data_name=self.dataset_name,
                batch_size=batch_size,
                num_workers=self.num_workers,
                val_size=val_size,
                test_size=test_size,
            )
        elif distribution == "zipfian":
            train_loader, val_loader, test_loader = create_dataloaders_zipfian(
                data_path=self.dataset_path,
                data_name=self.dataset_name,
                batch_size=batch_size,
                num_workers=self.num_workers,
                val_size=val_size,
                test_size=test_size,
            )
        else:
            raise ValueError(f"Unknown distribution: {distribution}. Use 'uniform' or 'zipfian'.")

        # Store
        self._dataloaders[distribution] = {
            "train": train_loader,
            "val": val_loader,
            "test": test_loader,
        }

    def get_info(self, distribution: Literal["uniform", "zipfian"] = "uniform") -> Dict[str, any]:
        """
        Get information about the dataset.

        Args:
            distribution: Which distribution to query

        Returns:
            Dictionary with dataset statistics
        """
        features = self.get_features(distribution, split="val")
        labels = features["labels"]

        return {
            "n_samples_val": len(labels),
            "n_classes": len(np.unique(labels)),
            "label_range": (labels.min(), labels.max()),
            "distribution": distribution,
            "has_density": "density" in features,
            "has_mean_item_size": "mean_item_size" in features,
            "dataset_path": self.dataset_path,
            "dataset_name": self.dataset_name,
        }

    def __repr__(self) -> str:
        loaded = list(self._dataloaders.keys())
        return (
            f"DatasetManager(path='{self.dataset_path}', "
            f"dataset='{self.dataset_name}', "
            f"loaded={loaded})"
        )
