"""
Generic adapter for any PyTorch nn.Module.

This is a universal fallback adapter that works with:
- CNNs (ResNet, VGG, EfficientNet, etc.)
- Transformers (BERT, GPT, ViT, etc.)
- Any custom torch.nn.Module

Limitations:
- Does NOT support reconstruction by default
- Layer-wise extraction requires explicit layer names
"""

from .base import BaseAdapter
import torch
import torch.nn as nn
from typing import List, Optional, Dict, Any


class PyTorchAdapter(BaseAdapter):
    """
    Generic adapter for any PyTorch nn.Module.

    This adapter provides a minimal interface for models that don't have
    custom architectures. It's designed as a universal fallback.

    Attributes:
        output_layer: Name of layer to extract features from
        flatten_output: Whether to flatten spatial dimensions

    Example:
        >>> # Use with pretrained ResNet
        >>> import torchvision.models as models
        >>> resnet = models.resnet50(pretrained=True)
        >>> adapter = PyTorchAdapter(resnet, output_layer="avgpool")
        >>> features = adapter.encode(images)
        >>>
        >>> # Use with custom model
        >>> model = MyCustomCNN()
        >>> adapter = PyTorchAdapter(model)  # Uses final output
    """

    def __init__(
        self,
        model: nn.Module,
        output_layer: Optional[str] = None,
        flatten_output: bool = True
    ):
        """
        Initialize PyTorch adapter.

        Args:
            model: Any PyTorch nn.Module
            output_layer: Name of layer to extract from (None = final output).
                         Use model.named_modules() to see available names.
            flatten_output: If True, flatten output to [batch, features].
                           If False, keep original shape.

        Raises:
            ValueError: If model is not a torch.nn.Module

        Example:
            >>> # Extract from specific layer
            >>> adapter = PyTorchAdapter(resnet, output_layer="layer4")
            >>>
            >>> # Extract final output without flattening
            >>> adapter = PyTorchAdapter(model, flatten_output=False)
        """
        super().__init__(model)

        if not isinstance(model, nn.Module):
            raise ValueError(
                f"PyTorchAdapter requires torch.nn.Module, got {type(model).__name__}"
            )

        self.output_layer = output_layer
        self.flatten_output = flatten_output

        # Validate output layer exists if specified
        if output_layer is not None:
            layer_names = dict(model.named_modules()).keys()
            if output_layer not in layer_names:
                raise ValueError(
                    f"Layer '{output_layer}' not found in model. "
                    f"Available layers: {list(layer_names)[:10]}..."
                )

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through model to extract features.

        Args:
            x: Input tensor [batch, ...]

        Returns:
            Feature tensor [batch, features] (if flatten_output=True)
            or [batch, ...] (if flatten_output=False)

        Example:
            >>> images = torch.randn(32, 3, 224, 224)
            >>> features = adapter.encode(images)
            >>> features.shape  # [32, 2048] for ResNet50
        """
        x = self.prepare_input(x)

        if self.output_layer is None:
            # Use final model output
            output = self.model(x)
        else:
            # Extract from specific layer using hook
            output = self._extract_from_layer(x, self.output_layer)

        # Handle different output formats
        if isinstance(output, tuple):
            output = output[0]  # Take first element of tuple
        elif isinstance(output, dict):
            # Some models return dicts (e.g., Transformers)
            output = output.get("last_hidden_state", output.get("logits", list(output.values())[0]))

        # Flatten if requested (for CNNs with spatial dimensions)
        if self.flatten_output and output.dim() > 2:
            output = output.reshape(output.size(0), -1)

        return output

    def _extract_from_layer(
        self,
        x: torch.Tensor,
        layer_name: str
    ) -> torch.Tensor:
        """
        Extract activation from specific named layer using forward hook.

        Args:
            x: Input tensor
            layer_name: Name of layer to extract from

        Returns:
            Activation tensor from specified layer
        """
        activation = None

        def hook_fn(module, input, output):
            nonlocal activation
            activation = output

        # Find and register hook
        for name, module in self.model.named_modules():
            if name == layer_name:
                handle = module.register_forward_hook(hook_fn)
                break
        else:
            # Should not happen (validated in __init__)
            raise ValueError(f"Layer '{layer_name}' not found")

        # Forward pass
        with torch.no_grad():
            self.model(x)

        # Cleanup
        handle.remove()

        if activation is None:
            raise RuntimeError(f"Failed to extract activation from layer '{layer_name}'")

        return activation

    def encode_layerwise(
        self,
        x: torch.Tensor,
        layers: Optional[List[str]] = None
    ) -> List[torch.Tensor]:
        """
        Extract activations from multiple named layers.

        Args:
            x: Input tensor [batch, ...]
            layers: List of layer names to extract from.
                   Example: ["layer1.0", "layer2.0", "layer3.0"]
                   If None, returns only final output.

        Returns:
            List of activation tensors, one per requested layer

        Example:
            >>> # Extract from multiple ResNet blocks
            >>> layers = ["layer1", "layer2", "layer3", "layer4"]
            >>> features = adapter.encode_layerwise(images, layers)
            >>> [f.shape for f in features]
            [[32, 256, 56, 56], [32, 512, 28, 28], [32, 1024, 14, 14], [32, 2048, 7, 7]]
        """
        if layers is None:
            # Return only final output
            return [self.encode(x)]

        x = self.prepare_input(x)

        # Storage for activations
        activations = {}
        hooks = []

        def make_hook(name):
            def hook_fn(module, input, output):
                activations[name] = output.detach()
            return hook_fn

        # Register hooks on requested layers
        layer_found = set()
        for name, module in self.model.named_modules():
            if name in layers:
                h = module.register_forward_hook(make_hook(name))
                hooks.append(h)
                layer_found.add(name)

        if len(layer_found) != len(layers):
            missing = set(layers) - layer_found
            print(f"Warning: Layers not found: {missing}")

        # Forward pass
        with torch.no_grad():
            self.model(x)

        # Cleanup hooks
        for h in hooks:
            h.remove()

        # Return in requested order (maintain order from layers list)
        result = []
        for name in layers:
            if name in activations:
                act = activations[name]
                if self.flatten_output and act.dim() > 2:
                    act = act.reshape(act.size(0), -1)
                result.append(act)

        return result

    def get_num_layers(self) -> int:
        """
        Count number of extractable layers (all named modules).

        Returns:
            Total number of named modules in model

        Note:
            This includes ALL modules (conv layers, linear layers, etc.).
            May be large for deep networks.
        """
        return len(list(self.model.named_modules()))

    def list_layers(self) -> List[str]:
        """
        List all extractable layer names.

        Returns:
            List of layer names that can be used with encode_layerwise()

        Example:
            >>> adapter.list_layers()
            ['conv1', 'bn1', 'relu', 'maxpool', 'layer1', 'layer1.0', ...]
        """
        return [name for name, _ in self.model.named_modules() if name]

    def get_metadata(self) -> Dict[str, Any]:
        """PyTorch model metadata."""
        meta = super().get_metadata()

        # Count parameters
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)

        meta.update({
            "pytorch_version": torch.__version__,
            "output_layer": self.output_layer,
            "flatten_output": self.flatten_output,
            "total_parameters": total_params,
            "trainable_parameters": trainable_params,
            "available_layers": self.list_layers()[:20],  # First 20 layers
        })

        return meta

    def __repr__(self) -> str:
        """String representation."""
        return (
            f"PyTorchAdapter("
            f"model={type(self.model).__name__}, "
            f"output_layer={self.output_layer}, "
            f"device={self.get_device()})"
        )
