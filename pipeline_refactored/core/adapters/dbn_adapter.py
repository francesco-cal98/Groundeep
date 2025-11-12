"""
Adapter for Deep Belief Networks (DBN) and Restricted Boltzmann Machines (RBM).

Compatible with:
- iDBN (Improved DBN) - your models
- gDBN (Generative DBN)
- Standard RBM stacks
- Any model with sequential .layers attribute
"""

from .base import BaseAdapter
import torch
from typing import List, Optional, Dict, Any


class DBNAdapter(BaseAdapter):
    """
    Adapter for Deep Belief Networks with sequential RBM layers.

    This adapter works with models that have a .layers attribute containing
    a list of RBM objects. Each RBM must implement .forward(x) and .backward(x).

    Attributes:
        layers: List of RBM layers extracted from model
        _has_represent: Whether model has optimized .represent() method

    Example:
        >>> # Your iDBN model
        >>> idbn = iDBN(layer_sizes=[10000, 1500, 500], ...)
        >>> adapter = DBNAdapter(idbn)
        >>> embeddings = adapter.encode(batch)
        >>> reconstructed = adapter.decode(embeddings)
    """

    def __init__(self, model):
        """
        Initialize DBN adapter.

        Args:
            model: DBN model object with .layers attribute, OR
                   dict with {"layers": [...], "params": {...}}

        Raises:
            ValueError: If model doesn't have .layers attribute
        """
        super().__init__(model)

        # Extract layers (support both direct .layers and dict format)
        if isinstance(model, dict) and "layers" in model:
            self.layers = model["layers"]
            self.params = model.get("params", {})
        else:
            self.layers = getattr(model, "layers", [])
            self.params = getattr(model, "params", {})

        if not self.layers:
            raise ValueError(
                "DBNAdapter requires model with .layers attribute "
                "containing RBM objects. Got model with no layers."
            )

        # Check if model has optimized .represent() method (e.g., iDBN)
        self._has_represent = (
            hasattr(model, "represent") and
            not isinstance(model, dict)
        )

        # Validate that layers have required methods
        self._validate_layers()

    def _validate_layers(self):
        """Validate that all layers have required forward/backward methods."""
        for i, rbm in enumerate(self.layers):
            if not hasattr(rbm, "forward"):
                raise ValueError(
                    f"Layer {i} does not have .forward() method. "
                    "All RBM layers must implement .forward(x)."
                )
            if not hasattr(rbm, "backward"):
                # Warning, not error (backward is only needed for reconstruction)
                pass

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """
        Extract top-level embedding by passing through all RBM layers.

        Uses optimized .represent() if available, otherwise does manual forward pass.

        Args:
            x: Input tensor [batch, ...] (will be flattened automatically)

        Returns:
            Top-level embedding [batch, embedding_dim]

        Example:
            >>> x = torch.randn(32, 1, 100, 100)  # Batch of images
            >>> z = adapter.encode(x)  # Shape: [32, 500] (top layer size)
        """
        # Use optimized path if available (faster for iDBN)
        if self._has_represent:
            return self.model.represent(x)

        # Manual forward pass through RBM stack
        x = self.prepare_input(x)
        cur = x.view(x.size(0), -1)  # Flatten input to [batch, features]

        for rbm in self.layers:
            cur = rbm.forward(cur)

        return cur

    def encode_layerwise(
        self,
        x: torch.Tensor,
        layers: Optional[List[int]] = None
    ) -> List[torch.Tensor]:
        """
        Extract embeddings at each RBM layer.

        Args:
            x: Input tensor [batch, ...]
            layers: Layer indices to extract (1-indexed). None = all layers.
                   Example: [1, 3] extracts layers 1 and 3.

        Returns:
            List of embeddings, one per requested layer

        Example:
            >>> # Extract all layers
            >>> embeddings = adapter.encode_layerwise(x)
            >>> len(embeddings)  # = number of RBM layers
            >>> embeddings[0].shape  # First layer output
            >>> embeddings[-1].shape  # Top layer output
            >>>
            >>> # Extract specific layers
            >>> top_only = adapter.encode_layerwise(x, layers=[3])
        """
        if layers is None:
            layers = list(range(1, len(self.layers) + 1))

        x = self.prepare_input(x)
        cur = x.view(x.size(0), -1)

        embeddings = []
        for li, rbm in enumerate(self.layers, start=1):
            cur = rbm.forward(cur)
            if li in layers:
                embeddings.append(cur.detach())

        return embeddings

    def decode(
        self,
        z: torch.Tensor,
        from_layer: Optional[int] = None
    ) -> torch.Tensor:
        """
        Reconstruct input by backward pass through RBM stack.

        Args:
            z: Embedding tensor [batch, embedding_dim]
            from_layer: Layer to decode from (1-indexed). None = decode from top.
                       Example: from_layer=2 decodes through layers 2 and 1.

        Returns:
            Reconstructed input [batch, input_dim]

        Example:
            >>> z = adapter.encode(x)  # Encode
            >>> x_recon = adapter.decode(z)  # Full reconstruction
            >>> x_partial = adapter.decode(z, from_layer=2)  # Partial
        """
        if from_layer is None:
            from_layer = len(self.layers)

        cur = z.to(self.get_device())

        # Backward through specified layers
        for rbm in reversed(self.layers[:from_layer]):
            if not hasattr(rbm, 'backward'):
                raise NotImplementedError(
                    f"RBM layer does not have .backward() method. "
                    "Reconstruction not supported for this model."
                )
            cur = rbm.backward(cur)

        return cur

    def get_device(self) -> torch.device:
        """
        Infer device from first RBM's weights.

        Checks common RBM parameter names: W, hid_bias, vis_bias, weight, bias.

        Returns:
            Device where model parameters are stored
        """
        if self._device is not None:
            return self._device

        if not self.layers:
            self._device = torch.device('cpu')
            return self._device

        # Check common RBM parameter names
        first_rbm = self.layers[0]
        for attr_name in ("W", "hid_bias", "vis_bias", "weight", "bias", "hbias", "vbias"):
            attr_val = getattr(first_rbm, attr_name, None)
            if isinstance(attr_val, torch.Tensor):
                self._device = attr_val.device
                return self._device

        # Fallback to parent method (tries model.parameters())
        return super().get_device()

    def get_num_layers(self) -> int:
        """Return number of RBM layers."""
        return len(self.layers)

    def get_layer_sizes(self) -> List[tuple]:
        """
        Get (input_size, output_size) for each layer.

        Returns:
            List of (num_visible, num_hidden) tuples

        Example:
            >>> adapter.get_layer_sizes()
            [(10000, 1500), (1500, 500)]
        """
        sizes = []
        for rbm in self.layers:
            if hasattr(rbm, 'num_visible') and hasattr(rbm, 'num_hidden'):
                sizes.append((rbm.num_visible, rbm.num_hidden))
            else:
                # Try to infer from weight matrix shape
                W = getattr(rbm, 'W', None) or getattr(rbm, 'weight', None)
                if W is not None:
                    sizes.append(tuple(W.shape))
                else:
                    sizes.append((None, None))
        return sizes

    def get_metadata(self) -> Dict[str, Any]:
        """
        Extended metadata with DBN-specific information.

        Returns:
            Dictionary with adapter and model metadata
        """
        meta = super().get_metadata()

        # Add DBN-specific info
        layer_sizes = self.get_layer_sizes()
        meta.update({
            "layer_sizes": layer_sizes,
            "num_rbm_layers": len(self.layers),
            "has_represent_method": self._has_represent,
            "architecture": self._infer_architecture_name(),
        })

        # Add training params if available
        if self.params:
            meta["training_params"] = {
                k: v for k, v in self.params.items()
                if k in ("LEARNING_RATE", "CD", "EPOCHS", "SPARSITY")
            }

        return meta

    def _infer_architecture_name(self) -> str:
        """Infer architecture name from layer sizes."""
        sizes = self.get_layer_sizes()
        if all(h is not None for _, h in sizes):
            hidden_sizes = [h for _, h in sizes]
            return "_".join(map(str, hidden_sizes))
        return "unknown"

    def get_rbm_layer(self, layer_idx: int):
        """
        Get specific RBM layer object.

        Args:
            layer_idx: Layer index (1-indexed)

        Returns:
            RBM layer object

        Example:
            >>> rbm1 = adapter.get_rbm_layer(1)
            >>> rbm1.W.shape
        """
        if layer_idx < 1 or layer_idx > len(self.layers):
            raise IndexError(
                f"Layer index {layer_idx} out of range. "
                f"Model has {len(self.layers)} layers (1-indexed)."
            )
        return self.layers[layer_idx - 1]

    def __repr__(self) -> str:
        """String representation."""
        return (
            f"DBNAdapter("
            f"layers={self.get_num_layers()}, "
            f"arch={self._infer_architecture_name()}, "
            f"device={self.get_device()})"
        )
