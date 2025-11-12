"""
Adapter system for model-agnostic analysis pipeline.

This module provides a unified interface for working with different model types:
- DBN/RBM models (iDBN, gDBN)
- VAE models
- Generic PyTorch models (CNNs, Transformers, etc.)

Quick Start:
    >>> from pipeline_refactored.core.adapters import create_adapter
    >>>
    >>> # Auto-detect model type
    >>> adapter = create_adapter(my_model)
    >>>
    >>> # Or specify explicitly
    >>> adapter = create_adapter(my_vae, adapter_type="vae")
    >>>
    >>> # Use adapter
    >>> embeddings = adapter.encode(batch)
    >>> reconstructed = adapter.decode(embeddings)  # If supported
    >>> print(adapter.get_metadata())

Available Adapters:
    - BaseAdapter: Abstract base class (inherit from this for custom adapters)
    - DBNAdapter: For Deep Belief Networks with sequential RBM layers
    - VAEAdapter: For Variational Autoencoders
    - PyTorchAdapter: Universal fallback for any torch.nn.Module

Factory Function:
    - create_adapter(): Auto-detect or manually specify adapter type
"""

from .base import BaseAdapter
from .dbn_adapter import DBNAdapter
from .vae_adapter import VAEAdapter
from .pytorch_adapter import PyTorchAdapter

import torch
from typing import Optional, Any, Type

# Export all adapters
__all__ = [
    "BaseAdapter",
    "DBNAdapter",
    "VAEAdapter",
    "PyTorchAdapter",
    "create_adapter",
]


def create_adapter(
    model: Any,
    adapter_type: str = "auto",
    **kwargs
) -> BaseAdapter:
    """
    Factory function to create appropriate adapter for a model.

    This function automatically detects the model type and returns the
    appropriate adapter, or you can specify the adapter type manually.

    Args:
        model: Model object (any type)
        adapter_type: One of:
            - "auto": Auto-detect model type (default)
            - "dbn": Use DBNAdapter
            - "vae": Use VAEAdapter
            - "pytorch": Use PyTorchAdapter
            - Custom adapter class: Pass BaseAdapter subclass
        **kwargs: Additional arguments passed to adapter constructor

    Returns:
        Appropriate adapter instance

    Raises:
        ValueError: If model type cannot be detected or adapter_type is invalid

    Examples:
        >>> # Auto-detection (recommended)
        >>> adapter = create_adapter(my_dbn_model)
        >>> type(adapter)
        <class 'DBNAdapter'>
        >>>
        >>> # Explicit type
        >>> adapter = create_adapter(my_vae, adapter_type="vae")
        >>>
        >>> # With options
        >>> adapter = create_adapter(
        ...     resnet,
        ...     adapter_type="pytorch",
        ...     output_layer="layer4",
        ...     flatten_output=True
        ... )
        >>>
        >>> # Custom adapter class
        >>> class MyAdapter(BaseAdapter):
        ...     def encode(self, x):
        ...         return self.model.custom_encode(x)
        >>> adapter = create_adapter(my_model, adapter_type=MyAdapter)
    """
    if adapter_type == "auto":
        return _auto_detect_adapter(model, **kwargs)

    elif adapter_type == "dbn":
        return DBNAdapter(model, **kwargs)

    elif adapter_type == "vae":
        return VAEAdapter(model, **kwargs)

    elif adapter_type == "pytorch":
        return PyTorchAdapter(model, **kwargs)

    elif isinstance(adapter_type, type) and issubclass(adapter_type, BaseAdapter):
        # Custom adapter class provided
        return adapter_type(model, **kwargs)

    else:
        raise ValueError(
            f"Unknown adapter_type: {adapter_type}. "
            f"Valid options: 'auto', 'dbn', 'vae', 'pytorch', or BaseAdapter subclass."
        )


def _auto_detect_adapter(model: Any, **kwargs) -> BaseAdapter:
    """
    Auto-detect model type and return appropriate adapter.

    Detection heuristics (in order):
    1. Check for .layers attribute (DBN/RBM)
    2. Check for dict with "layers" key (serialized DBN)
    3. Check for .encoder and .decoder (VAE)
    4. Check if torch.nn.Module (generic PyTorch)
    5. Raise error if none match

    Args:
        model: Model object
        **kwargs: Additional arguments for adapter

    Returns:
        Appropriate adapter instance

    Raises:
        ValueError: If model type cannot be detected
    """
    # 1. Check for DBN/RBM with .layers attribute
    if hasattr(model, "layers"):
        layers = getattr(model, "layers", [])
        if layers:
            # Verify it looks like RBM layers
            first_layer = layers[0]
            if hasattr(first_layer, "forward"):
                print(f"[Adapter] Auto-detected DBN/RBM model with {len(layers)} layers")
                return DBNAdapter(model, **kwargs)

    # 2. Check for dict-based DBN (serialized format)
    if isinstance(model, dict) and "layers" in model:
        layers = model["layers"]
        if layers and hasattr(layers[0], "forward"):
            print(f"[Adapter] Auto-detected serialized DBN with {len(layers)} layers")
            return DBNAdapter(model, **kwargs)

    # 3. Check for VAE (encoder + decoder)
    if hasattr(model, "encoder") and hasattr(model, "decoder"):
        print("[Adapter] Auto-detected VAE model")
        return VAEAdapter(model, **kwargs)

    # 4. Check for generic PyTorch model
    if isinstance(model, torch.nn.Module):
        model_name = type(model).__name__
        print(f"[Adapter] Auto-detected PyTorch model ({model_name}), using generic adapter")
        return PyTorchAdapter(model, **kwargs)

    # 5. Could not detect
    raise ValueError(
        f"Could not auto-detect adapter for model type {type(model).__name__}. "
        f"Please specify adapter_type explicitly: 'dbn', 'vae', or 'pytorch'.\n"
        f"Or create a custom adapter by subclassing BaseAdapter."
    )


def list_available_adapters() -> dict:
    """
    List all available adapter types.

    Returns:
        Dictionary mapping adapter names to classes

    Example:
        >>> adapters = list_available_adapters()
        >>> for name, cls in adapters.items():
        ...     print(f"{name}: {cls.__doc__.split(chr(10))[0]}")
        dbn: Adapter for Deep Belief Networks with sequential RBM layers
        vae: Adapter for Variational Autoencoders
        pytorch: Generic adapter for any PyTorch nn.Module
    """
    return {
        "dbn": DBNAdapter,
        "vae": VAEAdapter,
        "pytorch": PyTorchAdapter,
    }


def validate_adapter(adapter: BaseAdapter, verbose: bool = True) -> bool:
    """
    Validate adapter implementation.

    Args:
        adapter: Adapter instance to validate
        verbose: If True, print validation results

    Returns:
        True if adapter passes all checks, False otherwise

    Example:
        >>> adapter = create_adapter(my_model)
        >>> if not validate_adapter(adapter):
        ...     print("Warning: Adapter has issues")
    """
    results = adapter.validate()

    if verbose:
        print(f"\n{adapter.__class__.__name__} Validation:")
        print("=" * 50)
        for check, passed in results.items():
            if check.endswith("_error"):
                continue
            status = "✓" if passed else "✗"
            print(f"{status} {check}: {passed}")
            if not passed and f"{check}_error" in results:
                print(f"  Error: {results[f'{check}_error']}")
        print("=" * 50)

    # Check if all core methods work
    essential_checks = ["encode", "device"]
    return all(results.get(check, False) for check in essential_checks)
