"""
Adapter for Variational Autoencoders (VAE).

Compatible with:
- Standard VAEs with separate encoder/decoder
- Beta-VAEs
- Conditional VAEs (with appropriate modifications)
"""

from .base import BaseAdapter
import torch
from typing import List, Optional, Dict, Any, Tuple


class VAEAdapter(BaseAdapter):
    """
    Adapter for Variational Autoencoders.

    This adapter works with VAE models that have separate .encoder and .decoder
    networks. The encoder typically outputs (mu, logvar) for reparameterization.

    Attributes:
        encoder: Encoder network (extracts latent representation)
        decoder: Decoder network (reconstructs from latent)

    Example:
        >>> # Standard VAE
        >>> vae = VAE(input_dim=784, latent_dim=128)
        >>> adapter = VAEAdapter(vae)
        >>> mu = adapter.encode(batch)  # Returns mu, not sampled z
        >>> reconstructed = adapter.decode(mu)
    """

    def __init__(self, model, use_sampling: bool = False):
        """
        Initialize VAE adapter.

        Args:
            model: VAE model with .encoder and .decoder attributes
            use_sampling: If True, sample from latent distribution.
                         If False, use mean (mu) only (deterministic).

        Raises:
            ValueError: If model doesn't have encoder/decoder
        """
        super().__init__(model)

        if not (hasattr(model, "encoder") and hasattr(model, "decoder")):
            raise ValueError(
                "VAEAdapter requires model with .encoder and .decoder attributes. "
                f"Got model of type {type(model).__name__}"
            )

        self.encoder = model.encoder
        self.decoder = model.decoder
        self.use_sampling = use_sampling

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """
        Extract latent representation from encoder.

        By default, returns mu (mean) for deterministic analysis.
        Set use_sampling=True in constructor to sample from distribution.

        Args:
            x: Input tensor [batch, ...]

        Returns:
            Latent representation [batch, latent_dim]
            - If use_sampling=False: returns mu
            - If use_sampling=True: returns sampled z ~ N(mu, exp(logvar))

        Example:
            >>> x = torch.randn(32, 1, 28, 28)  # MNIST batch
            >>> mu = adapter.encode(x)  # Deterministic
            >>> mu.shape  # [32, 128]
        """
        x = self.prepare_input(x)

        # Forward through encoder
        output = self.encoder(x)

        # Handle different VAE output formats
        mu, logvar = self._parse_encoder_output(output)

        if self.use_sampling:
            # Sample from latent distribution
            return self.reparameterize(mu, logvar)
        else:
            # Return mean for deterministic analysis
            return mu

    def _parse_encoder_output(
        self,
        output: Any
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Parse encoder output into (mu, logvar).

        Handles multiple common VAE output formats:
        - Tuple: (mu, logvar)
        - Dict: {"mu": ..., "logvar": ...} or {"mean": ..., "logvar": ...}
        - Tensor: assumes mu only (no variance)

        Args:
            output: Encoder output

        Returns:
            (mu, logvar) tuple
        """
        if isinstance(output, tuple) and len(output) >= 2:
            # (mu, logvar) format
            return output[0], output[1]

        elif isinstance(output, dict):
            # Dictionary format
            mu = output.get("mu") or output.get("mean")
            logvar = output.get("logvar") or output.get("log_var")
            if mu is None:
                raise ValueError(
                    "Encoder dict output must contain 'mu' or 'mean' key"
                )
            return mu, logvar

        elif isinstance(output, torch.Tensor):
            # Direct tensor output (no variance)
            return output, None

        else:
            raise ValueError(
                f"Unsupported encoder output type: {type(output)}. "
                "Expected tuple, dict, or tensor."
            )

    def reparameterize(
        self,
        mu: torch.Tensor,
        logvar: Optional[torch.Tensor]
    ) -> torch.Tensor:
        """
        Reparameterization trick: z = mu + std * eps.

        Args:
            mu: Mean of latent distribution [batch, latent_dim]
            logvar: Log variance [batch, latent_dim] (optional)

        Returns:
            Sampled latent code [batch, latent_dim]
        """
        if logvar is None:
            # No variance provided, return mean
            return mu

        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def encode_layerwise(
        self,
        x: torch.Tensor,
        layers: Optional[List[str]] = None
    ) -> List[torch.Tensor]:
        """
        Extract intermediate encoder representations using forward hooks.

        Args:
            x: Input tensor [batch, ...]
            layers: List of layer names to extract from (e.g., ["conv1", "fc1"]).
                   If None, extracts from all encoder children.

        Returns:
            List of activation tensors

        Example:
            >>> # Extract from specific layers
            >>> acts = adapter.encode_layerwise(x, layers=["conv1", "conv2"])
            >>>
            >>> # Extract from all layers
            >>> acts = adapter.encode_layerwise(x)
        """
        x = self.prepare_input(x)

        # Get encoder modules (either by name or by children)
        if layers is not None:
            # Extract by layer names
            encoder_dict = dict(self.encoder.named_modules())
            target_modules = [(name, encoder_dict[name]) for name in layers if name in encoder_dict]
        else:
            # Extract from all children
            target_modules = [(str(i), module) for i, module in enumerate(self.encoder.children())]

        # Storage for activations
        activations = []
        hooks = []

        def make_hook(name):
            def hook_fn(module, input, output):
                # Handle tuple outputs (some layers return multiple values)
                if isinstance(output, tuple):
                    output = output[0]
                activations.append((name, output.detach()))
            return hook_fn

        # Register hooks
        for name, module in target_modules:
            h = module.register_forward_hook(make_hook(name))
            hooks.append(h)

        # Forward pass
        _ = self.encoder(x)

        # Cleanup hooks
        for h in hooks:
            h.remove()

        # Return tensors in registration order
        return [act for _, act in activations]

    def decode(
        self,
        z: torch.Tensor,
        from_layer: Optional[int] = None
    ) -> torch.Tensor:
        """
        Reconstruct from latent code using decoder.

        Args:
            z: Latent code [batch, latent_dim]
            from_layer: Not supported for VAE (partial decoding not standard).
                       Must be None.

        Returns:
            Reconstructed input [batch, ...]

        Raises:
            NotImplementedError: If from_layer is specified (not supported)

        Example:
            >>> mu = adapter.encode(x)
            >>> x_recon = adapter.decode(mu)
        """
        if from_layer is not None and from_layer != self.get_num_layers():
            raise NotImplementedError(
                "VAEAdapter does not support partial decoding from intermediate layers. "
                "Set from_layer=None to decode from latent space."
            )

        z = z.to(self.get_device())
        return self.decoder(z)

    def get_device(self) -> torch.device:
        """Get device from encoder parameters."""
        if self._device is not None:
            return self._device

        try:
            self._device = next(self.encoder.parameters()).device
        except StopIteration:
            self._device = torch.device('cpu')

        return self._device

    def get_num_layers(self) -> int:
        """Count encoder layers (children)."""
        return len(list(self.encoder.children()))

    def get_latent_dim(self) -> Optional[int]:
        """
        Infer latent dimension from encoder output.

        Returns:
            Latent dimension or None if inference fails

        Example:
            >>> adapter.get_latent_dim()
            128
        """
        try:
            dummy = torch.randn(1, 3, 64, 64).to(self.get_device())
            mu = self.encode(dummy)
            return mu.shape[1]
        except Exception:
            return None

    def get_metadata(self) -> Dict[str, Any]:
        """VAE-specific metadata."""
        meta = super().get_metadata()
        meta.update({
            "vae_type": "VAE",
            "encoder_layers": len(list(self.encoder.children())),
            "decoder_layers": len(list(self.decoder.children())),
            "latent_dim": self.get_latent_dim(),
            "use_sampling": self.use_sampling,
        })
        return meta

    def __repr__(self) -> str:
        """String representation."""
        latent_dim = self.get_latent_dim()
        return (
            f"VAEAdapter("
            f"latent_dim={latent_dim}, "
            f"sampling={self.use_sampling}, "
            f"device={self.get_device()})"
        )
