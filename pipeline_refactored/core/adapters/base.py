"""
Base adapter for model-agnostic analysis pipeline.

This module provides the abstract base class that all model-specific adapters
must inherit from. It defines a standard interface for:
- Encoding (forward pass to extract embeddings)
- Decoding (reconstruction, optional)
- Layer-wise extraction (optional)
- Device management
"""

from abc import ABC, abstractmethod
from typing import List, Optional, Dict, Any
import torch
import numpy as np


class BaseAdapter(ABC):
    """
    Abstract base adapter that all model-specific adapters must inherit from.

    Provides default implementations for optional methods and requires
    subclasses to implement only the essential encode() method.

    Design Philosophy:
    - Minimal required interface (only encode() is mandatory)
    - Sensible defaults for optional features
    - Easy to extend for new model types
    - Graceful degradation (missing features raise NotImplementedError)

    Example:
        >>> class MyAdapter(BaseAdapter):
        ...     def encode(self, x):
        ...         return self.model(x)
        >>>
        >>> adapter = MyAdapter(my_model)
        >>> embeddings = adapter.encode(batch)
    """

    def __init__(self, model: Any):
        """
        Initialize adapter with a model.

        Args:
            model: The raw model object (can be any type: nn.Module, dict, custom class)
        """
        self.model = model
        self._device = None  # Cache device after first inference
        self._metadata_cache = None

    # =========================================================================
    # ABSTRACT METHODS (must be implemented by subclasses)
    # =========================================================================

    @abstractmethod
    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """
        Extract top-level embedding from input.

        This is the ONLY method that MUST be implemented by all adapters.
        All analysis stages can work with just this method.

        Args:
            x: Input tensor [batch, ...] (can be any shape)

        Returns:
            Embedding tensor [batch, embedding_dim]

        Note:
            - Input should be preprocessed using self.prepare_input() first
            - Output should be a 2D tensor (flatten if necessary)

        Example:
            >>> def encode(self, x):
            ...     x = self.prepare_input(x)
            ...     return self.model(x)
        """
        raise NotImplementedError(
            f"{self.__class__.__name__} must implement encode() method"
        )

    # =========================================================================
    # OPTIONAL METHODS (have default implementations, can be overridden)
    # =========================================================================

    def encode_layerwise(
        self,
        x: torch.Tensor,
        layers: Optional[List[int]] = None
    ) -> List[torch.Tensor]:
        """
        Extract embeddings at intermediate layers.

        Default: Returns only top-level embedding (for models without layers).
        Override: For models with extractable intermediate representations.

        Args:
            x: Input tensor [batch, ...]
            layers: Layer indices to extract (None = all layers, 1-indexed)

        Returns:
            List of embeddings [batch, dim_i] for each requested layer

        Note:
            If not overridden, geometry and dimensionality analyses will
            only use the top-level representation.

        Example:
            >>> def encode_layerwise(self, x, layers=None):
            ...     activations = []
            ...     for i, layer in enumerate(self.model.layers, 1):
            ...         x = layer(x)
            ...         if layers is None or i in layers:
            ...             activations.append(x.detach())
            ...     return activations
        """
        # Default: just return top embedding as single-element list
        return [self.encode(x)]

    def decode(
        self,
        z: torch.Tensor,
        from_layer: Optional[int] = None
    ) -> torch.Tensor:
        """
        Reconstruct input from embedding.

        Default: Raises NotImplementedError (for non-generative models).
        Override: For autoencoders, VAEs, GANs, etc.

        Args:
            z: Embedding tensor [batch, embedding_dim]
            from_layer: Layer to decode from (None = top layer, 1-indexed)

        Returns:
            Reconstructed input [batch, ...] (same shape as original input)

        Raises:
            NotImplementedError: If model doesn't support reconstruction

        Note:
            If not overridden, reconstruction-based analyses (MSE, AFP, SSIM)
            will be automatically skipped with a warning.

        Example:
            >>> def decode(self, z, from_layer=None):
            ...     return self.model.decoder(z)
        """
        raise NotImplementedError(
            f"{self.__class__.__name__} does not support reconstruction. "
            "Reconstruction-based analyses will be skipped."
        )

    def get_device(self) -> torch.device:
        """
        Get device where model parameters are stored.

        Default: Uses standard PyTorch parameter iteration.
        Override: For non-standard models or performance optimization.

        Returns:
            torch.device object (e.g., cpu, cuda:0)

        Note:
            Result is cached after first call for performance.

        Example:
            >>> def get_device(self):
            ...     return self.model.device  # If model has .device attribute
        """
        if self._device is not None:
            return self._device

        # Try standard PyTorch nn.Module approach
        if hasattr(self.model, 'parameters'):
            try:
                self._device = next(self.model.parameters()).device
                return self._device
            except StopIteration:
                pass

        # Fallback: CPU
        self._device = torch.device('cpu')
        return self._device

    def get_num_layers(self) -> int:
        """
        Get number of extractable layers.

        Default: Returns 1 (only top-level representation).
        Override: For models with multiple extractable layers.

        Returns:
            Number of layers available for extraction (>= 1)

        Example:
            >>> def get_num_layers(self):
            ...     return len(self.model.layers)
        """
        return 1

    def prepare_input(self, x: torch.Tensor) -> torch.Tensor:
        """
        Prepare input tensor for model (preprocessing, device transfer, etc.).

        Default: Moves to model device and converts to float32.
        Override: For models requiring specific preprocessing (normalization, etc.).

        Args:
            x: Raw input tensor

        Returns:
            Preprocessed tensor ready for model

        Example:
            >>> def prepare_input(self, x):
            ...     x = super().prepare_input(x)  # Device + float32
            ...     return (x - self.mean) / self.std  # Normalize
        """
        device = self.get_device()
        return x.to(device).float()

    def get_metadata(self) -> Dict[str, Any]:
        """
        Get adapter metadata for logging/debugging.

        Default: Returns adapter class name and basic capabilities.
        Override: To include model-specific information.

        Returns:
            Dictionary with adapter metadata

        Example:
            >>> def get_metadata(self):
            ...     meta = super().get_metadata()
            ...     meta["latent_dim"] = self.model.latent_dim
            ...     return meta
        """
        if self._metadata_cache is not None:
            return self._metadata_cache

        self._metadata_cache = {
            "adapter_type": self.__class__.__name__,
            "model_type": type(self.model).__name__,
            "num_layers": self.get_num_layers(),
            "supports_reconstruction": self.supports_reconstruction(),
            "supports_layerwise": self.supports_layerwise_extraction(),
            "device": str(self.get_device()),
        }
        return self._metadata_cache

    # =========================================================================
    # UTILITY METHODS (no need to override, use as-is)
    # =========================================================================

    def supports_reconstruction(self) -> bool:
        """
        Check if model supports reconstruction.

        Returns:
            True if decode() is implemented, False otherwise
        """
        try:
            # Try to call decode with dummy input
            dummy = torch.randn(1, 10).to(self.get_device())
            self.decode(dummy)
            return True
        except NotImplementedError:
            return False
        except Exception:
            # Other errors mean decode() exists but failed
            return True

    def supports_layerwise_extraction(self) -> bool:
        """
        Check if model supports layerwise extraction.

        Returns:
            True if model has multiple extractable layers, False otherwise
        """
        return self.get_num_layers() > 1

    def validate(self) -> Dict[str, bool]:
        """
        Run validation checks on adapter implementation.

        Returns:
            Dictionary with validation results

        Example:
            >>> adapter.validate()
            {'encode': True, 'decode': False, 'layerwise': True, 'device': True}
        """
        results = {}

        # Test encode
        try:
            dummy = torch.randn(2, 10).to(self.get_device())
            output = self.encode(dummy)
            results['encode'] = (
                isinstance(output, torch.Tensor) and
                output.shape[0] == 2
            )
        except Exception as e:
            results['encode'] = False
            results['encode_error'] = str(e)

        # Test decode
        try:
            dummy = torch.randn(2, 10).to(self.get_device())
            output = self.decode(dummy)
            results['decode'] = isinstance(output, torch.Tensor)
        except NotImplementedError:
            results['decode'] = False
        except Exception as e:
            results['decode'] = False
            results['decode_error'] = str(e)

        # Test layerwise
        try:
            dummy = torch.randn(2, 10).to(self.get_device())
            outputs = self.encode_layerwise(dummy)
            results['layerwise'] = (
                isinstance(outputs, list) and
                len(outputs) > 0 and
                all(isinstance(o, torch.Tensor) for o in outputs)
            )
        except Exception as e:
            results['layerwise'] = False
            results['layerwise_error'] = str(e)

        # Test device
        try:
            device = self.get_device()
            results['device'] = isinstance(device, torch.device)
        except Exception as e:
            results['device'] = False
            results['device_error'] = str(e)

        return results

    def __repr__(self) -> str:
        """String representation of adapter."""
        return (
            f"{self.__class__.__name__}("
            f"model={type(self.model).__name__}, "
            f"layers={self.get_num_layers()}, "
            f"device={self.get_device()})"
        )

    def __str__(self) -> str:
        """Human-readable string."""
        return self.__repr__()
