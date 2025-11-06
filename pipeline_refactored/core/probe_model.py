"""Probe-ready model wrapper.

This module contains ProbeReadyModel, previously in analyze.py.
Now it's clean with zero dependencies on analyze.py.
"""

from pathlib import Path
from typing import Dict, Any, Optional
import torch
import numpy as np


class ProbeReadyModel:
    """Adapter for offline probe training.

    Wraps a raw model and provides a clean interface for extracting
    representations at different layers.

    Attributes:
        raw: The underlying model (iDBN, gDBN, etc.)
        val_loader: Validation dataloader
        features: Dictionary of features (labels, cumArea, CH, etc.)
        wandb_run: Optional WandB run for logging
        arch_dir: Output directory path
        text_flag: Whether model processes text (always False for iDBN)
        device: Device where model is located
    """

    def __init__(
        self,
        raw_model: Any,
        val_loader: Any,
        features_dict: Dict[str, np.ndarray],
        out_dir: Path,
        wandb_run: Optional[Any] = None,
    ):
        """Initialize ProbeReadyModel.

        Args:
            raw_model: The model to wrap
            val_loader: Validation dataloader
            features_dict: Dictionary of features for probing
            out_dir: Output directory
            wandb_run: Optional WandB run
        """
        self.raw = raw_model
        self.val_loader = val_loader
        self.features = features_dict
        self.wandb_run = wandb_run
        self.arch_dir = str(out_dir)
        self.text_flag = False
        self.device = self._get_model_device(raw_model)

    def _get_model_device(self, model: Any) -> torch.device:
        """Infer device from model parameters.

        Args:
            model: The model

        Returns:
            Device (cuda or cpu)
        """
        try:
            # Try to get device from model layers (for RBM stack)
            layers = getattr(model, "layers", [])
            if layers:
                first_rbm = layers[0]
                for attr_name in ("W", "c", "b", "weights"):
                    attr_val = getattr(first_rbm, attr_name, None)
                    if isinstance(attr_val, torch.Tensor):
                        return attr_val.device
        except Exception:
            pass

        # Fallback to CPU
        return torch.device("cpu")

    @torch.no_grad()
    def represent(
        self,
        x: torch.Tensor,
        upto_layer: Optional[int] = None
    ) -> torch.Tensor:
        """Extract representations from the model.

        Args:
            x: Input tensor
            upto_layer: Layer index to stop at (None = all layers)

        Returns:
            Representations at specified layer
        """
        # Check if model has built-in represent method
        if hasattr(self.raw, "represent"):
            if upto_layer is not None:
                return self.raw.represent(x, upto_layer=upto_layer)
            else:
                return self.raw.represent(x)

        # Fallback: manually forward through RBM stack
        xt = x
        layers = getattr(self.raw, "layers", [])
        upto = len(layers) if upto_layer is None else min(upto_layer, len(layers))

        for rbm in layers[:upto]:
            xt = rbm.forward(xt)

        return xt
