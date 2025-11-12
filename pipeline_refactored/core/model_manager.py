"""
ModelManager: Clean model loading and device management with adapter support.

Replaces the model-handling parts of Embedding_analysis with:
- Lazy loading (load models on demand)
- Device inference and management
- Support for both dict-based and object-based models
- Adapter system for model-agnostic analysis
"""

import sys
from pathlib import Path
from typing import Any, Dict, Optional
import pickle as pkl
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Import adapter system
from .adapters import BaseAdapter, create_adapter


class ModelWrapper:
    """
    Wrapper for dict-based pickled models.

    Handles legacy format: {'layers': [...], 'params': {...}}
    """

    def __init__(self, model_dict: Dict):
        self.layers = model_dict.get("layers", [])
        self.params = model_dict.get("params", {})

    def decode(self, top: torch.Tensor) -> torch.Tensor:
        """Decode from top layer back to input space."""
        with torch.no_grad():
            cur = top
            for rbm in reversed(self.layers):
                cur = rbm.backward(cur)
            return cur

    def __repr__(self):
        return f"ModelWrapper(n_layers={len(self.layers)})"


class ModelManager:
    """
    Manages model loading, device placement, and adapter wrapping.

    Features:
    - Lazy loading: models loaded on demand
    - Device inference: automatically detects model device
    - Flexible: supports both dict-based and object-based models
    - Caching: loads each model only once
    - Adapter system: wraps models for uniform interface

    Example:
        >>> mm = ModelManager()
        >>> model = mm.load_model("path/to/model.pkl", label="uniform")
        >>> adapter = mm.get_adapter("uniform")
        >>> embeddings = adapter.encode(batch)
        >>> device = mm.get_device("uniform")
        >>> print(mm.get_info("uniform"))
    """

    def __init__(self, adapter_type: str = "auto"):
        """
        Initialize ModelManager.

        Args:
            adapter_type: Default adapter type for all models.
                         Options: "auto", "dbn", "vae", "pytorch"
        """
        self._models: Dict[str, Any] = {}
        self._adapters: Dict[str, BaseAdapter] = {}
        self._devices: Dict[str, torch.device] = {}
        self._paths: Dict[str, str] = {}
        self._default_adapter_type = adapter_type

    def load_model(
        self,
        model_path: str,
        label: Optional[str] = None,
        force_reload: bool = False,
        adapter_type: Optional[str] = None,
        **adapter_kwargs
    ) -> Any:
        """
        Load a model from disk and wrap it with an adapter.

        Args:
            model_path: Path to .pkl file
            label: Identifier for this model (defaults to filename)
            force_reload: If True, reload even if already cached
            adapter_type: Adapter type for this model (overrides default).
                         Options: "auto", "dbn", "vae", "pytorch", or None
            **adapter_kwargs: Additional arguments passed to adapter

        Returns:
            Loaded model object (raw, not adapter)

        Raises:
            FileNotFoundError: If model file doesn't exist

        Note:
            Use get_adapter(label) to get the wrapped adapter for analysis.
        """
        model_path = str(model_path)
        label = label or Path(model_path).stem

        # Return cached model unless force reload
        if label in self._models and not force_reload:
            return self._models[label]

        # Check file exists
        if not Path(model_path).exists():
            raise FileNotFoundError(f"Model file not found: {model_path}")

        # Load model
        print(f"[ModelManager] Loading model: {label} from {model_path}")
        model = self._load_model_file(model_path)

        # Create adapter
        adapter_type = adapter_type or self._default_adapter_type
        try:
            adapter = create_adapter(model, adapter_type=adapter_type, **adapter_kwargs)
            device = adapter.get_device()
        except Exception as e:
            print(f"[ModelManager] Warning: Failed to create adapter: {e}")
            print(f"[ModelManager] Falling back to legacy device inference")
            adapter = None
            device = self._infer_device(model)

        # Cache
        self._models[label] = model
        self._adapters[label] = adapter
        self._devices[label] = device
        self._paths[label] = model_path

        adapter_info = f" ({adapter.__class__.__name__})" if adapter else ""
        print(f"[ModelManager] Loaded {label} on {device}{adapter_info}")
        return model

    def get_model(self, label: str) -> Any:
        """
        Get a loaded model by label (raw model, not adapter).

        Args:
            label: Model identifier

        Returns:
            Raw model object

        Raises:
            KeyError: If model not loaded

        Note:
            For analysis, use get_adapter(label) instead for uniform interface.
        """
        if label not in self._models:
            raise KeyError(
                f"Model '{label}' not loaded. Available: {list(self._models.keys())}"
            )
        return self._models[label]

    def get_adapter(self, label: str) -> BaseAdapter:
        """
        Get the adapter for a loaded model.

        This is the preferred method for analysis - use adapter interface
        instead of raw model for model-agnostic code.

        Args:
            label: Model identifier

        Returns:
            Adapter instance wrapping the model

        Raises:
            KeyError: If model not loaded
            ValueError: If adapter creation failed

        Example:
            >>> adapter = mm.get_adapter("uniform")
            >>> embeddings = adapter.encode(batch)
            >>> reconstructed = adapter.decode(embeddings)
        """
        if label not in self._adapters:
            raise KeyError(
                f"Model '{label}' not loaded. Available: {list(self._adapters.keys())}"
            )

        adapter = self._adapters[label]
        if adapter is None:
            raise ValueError(
                f"Adapter for '{label}' is None (creation may have failed). "
                "Check loading warnings."
            )

        return adapter

    def get_device(self, label: str) -> torch.device:
        """
        Get the device for a loaded model.

        Args:
            label: Model identifier

        Returns:
            torch.device

        Raises:
            KeyError: If model not loaded
        """
        if label not in self._devices:
            raise KeyError(
                f"Model '{label}' not loaded. Available: {list(self._devices.keys())}"
            )
        return self._devices[label]

    def get_layers(self, label: str) -> list:
        """
        Get the RBM layers from a model.

        Args:
            label: Model identifier

        Returns:
            List of RBM layers
        """
        model = self.get_model(label)
        return getattr(model, "layers", [])

    def get_n_layers(self, label: str) -> int:
        """
        Get number of layers in a model.

        Args:
            label: Model identifier

        Returns:
            Number of layers
        """
        return len(self.get_layers(label))

    def get_info(self, label: str) -> Dict[str, Any]:
        """
        Get information about a loaded model and its adapter.

        Args:
            label: Model identifier

        Returns:
            Dictionary with model and adapter info
        """
        if label not in self._models:
            return {"loaded": False}

        model = self._models[label]
        device = self._devices[label]
        adapter = self._adapters.get(label)

        info = {
            "loaded": True,
            "label": label,
            "path": self._paths.get(label),
            "device": str(device),
            "is_wrapper": isinstance(model, ModelWrapper),
        }

        # Add adapter information
        if adapter is not None:
            try:
                adapter_meta = adapter.get_metadata()
                info["adapter"] = adapter_meta
                info["n_layers"] = adapter.get_num_layers()
                info["supports_reconstruction"] = adapter.supports_reconstruction()
                info["supports_layerwise"] = adapter.supports_layerwise_extraction()
            except Exception as e:
                info["adapter"] = {"error": str(e)}
        else:
            # Fallback to legacy layer extraction
            layers = getattr(model, "layers", [])
            info["n_layers"] = len(layers)
            info["adapter"] = None

            # Try to get layer sizes (legacy)
            try:
                layer_sizes = []
                for layer in layers:
                    for attr in ("n_hidden", "hidden_size", "hid_bias"):
                        if hasattr(layer, attr):
                            if attr == "hid_bias":
                                size = getattr(layer, attr).numel()
                            else:
                                size = getattr(layer, attr)
                            layer_sizes.append(size)
                            break
                if layer_sizes:
                    info["layer_sizes"] = layer_sizes
            except Exception:
                pass

        return info

    def is_loaded(self, label: str) -> bool:
        """Check if a model is loaded."""
        return label in self._models

    def list_models(self) -> list:
        """Get list of loaded model labels."""
        return list(self._models.keys())

    def unload(self, label: str):
        """
        Unload a model to free memory.

        Args:
            label: Model identifier
        """
        if label in self._models:
            del self._models[label]
            del self._devices[label]
            if label in self._paths:
                del self._paths[label]
            torch.cuda.empty_cache()
            print(f"[ModelManager] Unloaded model: {label}")

    def unload_all(self):
        """Unload all models to free memory."""
        labels = list(self._models.keys())
        for label in labels:
            self.unload(label)

    @staticmethod
    def _load_model_file(path: str) -> Any:
        """
        Load a model from pickle file.

        Supports both:
        - Dict format: {'layers': [...], 'params': {...}}
        - Object format: model object with .layers attribute

        Args:
            path: Path to .pkl file

        Returns:
            Model object or ModelWrapper
        """
        # Setup device mapping for loading
        target_device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        # Save original restore location
        orig_restore = getattr(torch.serialization, "default_restore_location", None)

        # Set custom restore location
        if orig_restore is not None:
            torch.serialization.default_restore_location = (
                lambda storage, loc: storage.cuda()
                if target_device.type == "cuda"
                else storage.cpu()
            )

        try:
            with open(path, "rb") as f:
                model = pkl.load(f)
        finally:
            # Restore original
            if orig_restore is not None:
                torch.serialization.default_restore_location = orig_restore

        # Wrap dict-based models
        if isinstance(model, dict) and "layers" in model:
            return ModelWrapper(model)

        return model

    @staticmethod
    def _infer_device(model: Any) -> torch.device:
        """
        Infer device from model parameters.

        Checks layer tensors (W, hid_bias, vis_bias, etc.) to determine device.

        Args:
            model: Model object

        Returns:
            torch.device
        """
        layers = getattr(model, "layers", [])

        for layer in layers:
            # Check various possible tensor attributes
            for attr in ("W", "hid_bias", "vis_bias", "hbias", "vbias", "weight"):
                tensor = getattr(layer, attr, None)
                if isinstance(tensor, torch.Tensor):
                    return tensor.device

        # Default to cuda if available, else cpu
        return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    def __repr__(self) -> str:
        models = list(self._models.keys())
        return f"ModelManager(loaded_models={models})"

    def __len__(self) -> int:
        """Return number of loaded models."""
        return len(self._models)
