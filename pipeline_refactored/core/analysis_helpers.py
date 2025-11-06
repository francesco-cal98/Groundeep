"""Helper functions for analysis pipeline.

Functions previously scattered in analyze.py, now centralized here.
"""

from typing import Optional, Tuple, Any
import torch
import numpy as np
from dataclasses import dataclass


@dataclass
class EmbeddingBundle:
    """Container for embeddings and associated features.

    Attributes:
        embeddings: Embedding array
        labels: Numerosity labels
        cum_area: Cumulative area feature
        convex_hull: Convex hull feature
        inputs: Original input tensors
        mean_item_size: Mean item size (optional)
        density: Density feature (optional)
    """
    embeddings: np.ndarray
    labels: np.ndarray
    cum_area: np.ndarray
    convex_hull: np.ndarray
    inputs: torch.Tensor
    mean_item_size: Optional[np.ndarray] = None
    density: Optional[np.ndarray] = None


def maybe_init_wandb(
    use_wandb: bool,
    project: Optional[str] = None,
    run_name: Optional[str] = None
) -> Optional[Any]:
    """Initialize WandB if enabled.

    Args:
        use_wandb: Whether to use WandB
        project: WandB project name
        run_name: WandB run name

    Returns:
        WandB run object or None
    """
    if not use_wandb:
        return None

    try:
        import wandb
        return wandb.init(project=project, name=run_name)
    except ImportError:
        print("⚠️  WandB not installed, skipping logging")
        return None


def infer_chw_from_input(batch: torch.Tensor) -> Tuple[int, int, int]:
    """Infer channels, height, width from batch tensor.

    Args:
        batch: Input tensor [N, C, H, W] or [N, H, W]

    Returns:
        Tuple of (channels, height, width)
    """
    if batch.ndim == 4:
        # [N, C, H, W]
        _, c, h, w = batch.shape
        return int(c), int(h), int(w)
    elif batch.ndim == 3:
        # [N, H, W] - grayscale
        _, h, w = batch.shape
        return 1, int(h), int(w)
    elif batch.ndim == 2:
        # [N, D] - flatten, assume square
        _, d = batch.shape
        size = int(np.sqrt(d))
        if size * size == d:
            return 1, size, size
        else:
            return 1, 1, int(d)
    else:
        raise ValueError(f"Unexpected batch shape: {batch.shape}")


def infer_hw_from_batch(batch: torch.Tensor) -> Tuple[int, int]:
    """Infer height and width from batch tensor.

    Args:
        batch: Input tensor

    Returns:
        Tuple of (height, width)
    """
    _, h, w = infer_chw_from_input(batch)
    return h, w
