"""
EmbeddingExtractor: Clean embedding extraction from models with adapter support.

Replaces the embedding extraction parts of Embedding_analysis with:
- Single responsibility: only extracts embeddings
- Layer-wise extraction support
- Aligned pair extraction (for model comparison)
- Reconstruction support
- Adapter system for model-agnostic extraction
"""

import sys
from pathlib import Path
from typing import Dict, Optional, Tuple, List
import numpy as np
import torch
from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from pipeline_refactored.core.model_manager import ModelManager
from pipeline_refactored.core.adapters import BaseAdapter


class EmbeddingExtractor:
    """
    Extracts embeddings from models.

    Features:
    - Layer-wise extraction (specify which layer)
    - Batch processing with memory management
    - Aligned pair extraction (same inputs → two models)
    - Reconstruction support (encode + decode)

    Example:
        >>> extractor = EmbeddingExtractor(model_manager)
        >>> # Single model
        >>> Z = extractor.extract("uniform", dataloader)
        >>> # Aligned pair (for comparison)
        >>> Z_u, Z_z = extractor.extract_aligned_pair("uniform", "zipfian", dataloader)
        >>> # Layer-wise
        >>> Z_layer2 = extractor.extract("uniform", dataloader, layer=2)
    """

    def __init__(self, model_manager: ModelManager, use_adapters: bool = True):
        """
        Initialize the extractor.

        Args:
            model_manager: ModelManager instance with loaded models
            use_adapters: If True, use adapter interface (recommended).
                         If False, use legacy direct model access.
        """
        self.model_manager = model_manager
        self.use_adapters = use_adapters

    def extract(
        self,
        model_label: str,
        dataloader: DataLoader,
        layer: Optional[int] = None,
        verbose: bool = False,
    ) -> np.ndarray:
        """
        Extract embeddings from a single model.

        Args:
            model_label: Identifier of loaded model
            dataloader: DataLoader to extract from
            layer: Layer index (None = top layer, 1 = first layer, etc.)
            verbose: Print progress

        Returns:
            Embeddings as numpy array of shape (N, D)
        """
        # Use adapter if available and enabled
        if self.use_adapters:
            try:
                return self._extract_with_adapter(model_label, dataloader, layer, verbose)
            except Exception as e:
                if verbose:
                    print(f"[EmbeddingExtractor] Adapter extraction failed: {e}")
                    print("[EmbeddingExtractor] Falling back to legacy extraction")

        # Fallback to legacy extraction
        return self._extract_legacy(model_label, dataloader, layer, verbose)

    def _extract_with_adapter(
        self,
        model_label: str,
        dataloader: DataLoader,
        layer: Optional[int] = None,
        verbose: bool = False,
    ) -> np.ndarray:
        """Extract embeddings using adapter interface (new method)."""
        adapter = self.model_manager.get_adapter(model_label)

        embeddings = []
        n_batches = len(dataloader)

        with torch.no_grad():
            for i, batch in enumerate(dataloader):
                if verbose and (i % 10 == 0):
                    print(f"  Batch {i+1}/{n_batches}")

                # Get input (handle both (x,) and (x, y) formats)
                if isinstance(batch, (list, tuple)):
                    x = batch[0]
                else:
                    x = batch

                # Extract embeddings using adapter
                if layer is None:
                    # Top layer
                    emb = adapter.encode(x)
                else:
                    # Specific layer
                    layer_embs = adapter.encode_layerwise(x, layers=[layer])
                    emb = layer_embs[0] if layer_embs else adapter.encode(x)

                embeddings.append(emb.detach().cpu().numpy())

        return np.concatenate(embeddings, axis=0)

    def _extract_legacy(
        self,
        model_label: str,
        dataloader: DataLoader,
        layer: Optional[int] = None,
        verbose: bool = False,
    ) -> np.ndarray:
        """Extract embeddings using legacy direct model access."""
        model = self.model_manager.get_model(model_label)
        device = self.model_manager.get_device(model_label)

        embeddings = []
        n_batches = len(dataloader)

        with torch.no_grad():
            for i, batch in enumerate(dataloader):
                if verbose and i % 10 == 0:
                    print(f"[EmbeddingExtractor] Processing batch {i+1}/{n_batches}")

                # Extract data from batch
                if isinstance(batch, (tuple, list)) and len(batch) >= 1:
                    batch_data = batch[0]
                else:
                    batch_data = batch

                # Move to device and flatten if needed
                x = batch_data.to(device).float()
                if x.dim() > 2:  # (B, C, H, W) → (B, C*H*W)
                    x = x.view(x.size(0), -1)

                # Forward through layers
                z = self._forward_to_layer(model, x, layer)

                # Store
                embeddings.append(z.cpu().numpy())

        # Concatenate all batches
        result = np.concatenate(embeddings, axis=0)

        if verbose:
            print(f"[EmbeddingExtractor] Extracted embeddings: {result.shape}")

        return result

    def extract_aligned_pair(
        self,
        model_label_a: str,
        model_label_b: str,
        dataloader: DataLoader,
        layer: Optional[int] = None,
        verbose: bool = False,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Extract aligned embeddings from two models using the same inputs.

        This is crucial for fair model comparison (uniform vs zipfian):
        both models process exactly the same stimuli.

        Args:
            model_label_a: First model identifier
            model_label_b: Second model identifier
            dataloader: DataLoader (same inputs for both models)
            layer: Layer index
            verbose: Print progress

        Returns:
            Tuple of (embeddings_a, embeddings_b) as numpy arrays
        """
        model_a = self.model_manager.get_model(model_label_a)
        model_b = self.model_manager.get_model(model_label_b)
        device_a = self.model_manager.get_device(model_label_a)
        device_b = self.model_manager.get_device(model_label_b)

        embeddings_a = []
        embeddings_b = []
        n_batches = len(dataloader)

        with torch.no_grad():
            for i, batch in enumerate(dataloader):
                if verbose and i % 10 == 0:
                    print(f"[EmbeddingExtractor] Processing batch {i+1}/{n_batches}")

                # Extract data
                if isinstance(batch, (tuple, list)) and len(batch) >= 1:
                    batch_data = batch[0]
                else:
                    batch_data = batch

                # Process with model A
                x_a = batch_data.to(device_a).float()
                if x_a.dim() > 2:
                    x_a = x_a.view(x_a.size(0), -1)
                z_a = self._forward_to_layer(model_a, x_a, layer)
                embeddings_a.append(z_a.cpu().numpy())

                # Process with model B (same inputs!)
                x_b = batch_data.to(device_b).float()
                if x_b.dim() > 2:
                    x_b = x_b.view(x_b.size(0), -1)
                z_b = self._forward_to_layer(model_b, x_b, layer)
                embeddings_b.append(z_b.cpu().numpy())

        # Concatenate
        result_a = np.concatenate(embeddings_a, axis=0)
        result_b = np.concatenate(embeddings_b, axis=0)

        if verbose:
            print(f"[EmbeddingExtractor] Extracted aligned pair:")
            print(f"  {model_label_a}: {result_a.shape}")
            print(f"  {model_label_b}: {result_b.shape}")

        return result_a, result_b

    def extract_layerwise(
        self,
        model_label: str,
        dataloader: DataLoader,
        layers: Optional[List[int]] = None,  # None = all layers
        verbose: bool = False,
    ) -> Dict[int, np.ndarray]:
        """
        Extract embeddings from multiple layers.

        Args:
            model_label: Model identifier
            dataloader: DataLoader
            layers: List of layer indices (None = all layers)
            verbose: Print progress

        Returns:
            Dictionary mapping layer index to embeddings
        """
        n_layers = self.model_manager.get_n_layers(model_label)

        if layers is None:
            layers = list(range(1, n_layers + 1))

        results = {}
        for layer_idx in layers:
            if verbose:
                print(f"[EmbeddingExtractor] Extracting layer {layer_idx}/{n_layers}")

            results[layer_idx] = self.extract(
                model_label, dataloader, layer=layer_idx, verbose=False
            )

        if verbose:
            print(f"[EmbeddingExtractor] Extracted {len(results)} layers")

        return results

    def reconstruct(
        self,
        model_label: str,
        dataloader: DataLoader,
        n_samples: Optional[int] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Reconstruct inputs by encoding then decoding.

        Args:
            model_label: Model identifier
            dataloader: DataLoader
            n_samples: Limit number of samples (None = all)

        Returns:
            Tuple of (original, reconstructed) as numpy arrays
        """
        model = self.model_manager.get_model(model_label)
        device = self.model_manager.get_device(model_label)

        originals = []
        reconstructions = []
        total_samples = 0

        with torch.no_grad():
            for batch in dataloader:
                # Extract data
                if isinstance(batch, (tuple, list)) and len(batch) >= 1:
                    batch_data = batch[0]
                else:
                    batch_data = batch

                x = batch_data.to(device).float()
                if x.dim() > 2:
                    x = x.view(x.size(0), -1)

                # Encode
                z = self._forward_to_layer(model, x, layer=None)

                # Decode
                x_recon = self._backward_from_layer(model, z)

                # Store
                originals.append(x.cpu().numpy())
                reconstructions.append(x_recon.cpu().numpy())

                total_samples += x.size(0)
                if n_samples and total_samples >= n_samples:
                    break

        # Concatenate
        original = np.concatenate(originals, axis=0)
        reconstructed = np.concatenate(reconstructions, axis=0)

        if n_samples:
            original = original[:n_samples]
            reconstructed = reconstructed[:n_samples]

        return original, reconstructed

    @staticmethod
    def _forward_to_layer(model, x: torch.Tensor, layer: Optional[int] = None) -> torch.Tensor:
        """
        Forward pass through RBM stack up to specified layer.

        Args:
            model: Model object
            x: Input tensor
            layer: Layer index (None = all layers, 1 = first layer, etc.)

        Returns:
            Encoded representation
        """
        cur = x
        layers = getattr(model, "layers", [])

        # Determine how many layers to forward through
        if layer is None:
            max_layer = len(layers)
        else:
            max_layer = min(layer, len(layers))

        # Forward
        for rbm in layers[:max_layer]:
            cur = rbm.forward(cur)

        return cur

    @staticmethod
    def _backward_from_layer(model, z: torch.Tensor) -> torch.Tensor:
        """
        Backward pass through RBM stack (decode).

        Args:
            model: Model object
            z: Latent representation

        Returns:
            Reconstructed input
        """
        cur = z
        layers = getattr(model, "layers", [])

        # Backward through all layers in reverse
        for rbm in reversed(layers):
            cur = rbm.backward(cur)

        return cur

    def __repr__(self) -> str:
        loaded = self.model_manager.list_models()
        return f"EmbeddingExtractor(available_models={loaded})"
