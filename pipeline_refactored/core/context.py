"""
Context and Result data structures for the analysis pipeline.

These dataclasses provide clean interfaces for passing data between stages.
"""

from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Dict, Any, Optional
import json
import numpy as np


@dataclass
class AnalysisContext:
    """
    Runtime context for analysis stages.

    Contains all data needed to run analyses:
    - Embeddings from one or more models
    - Features (labels, area, convex hull, etc.)
    - Models (for methods that need model access)
    - Configuration and metadata

    Example:
        >>> context = AnalysisContext(
        ...     embeddings={"uniform": Z_u, "zipfian": Z_z},
        ...     features={"labels": labels, "cum_area": areas},
        ...     output_dir=Path("results/model_name")
        ... )
    """

    # Core data
    embeddings: Dict[str, np.ndarray] = field(default_factory=dict)
    features: Dict[str, np.ndarray] = field(default_factory=dict)

    # Optional model access (for analyses that need it)
    models: Dict[str, Any] = field(default_factory=dict)

    # Metadata
    architecture: str = "unknown"
    distribution: str = "unknown"
    output_dir: Optional[Path] = None
    config: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def get_embedding(self, key: str = "uniform") -> np.ndarray:
        """
        Get embeddings for a specific model/layer.

        Args:
            key: Embedding identifier (e.g., 'uniform', 'zipfian', 'layer_1')

        Returns:
            Embedding array

        Raises:
            KeyError: If embedding not found
        """
        if key not in self.embeddings:
            available = list(self.embeddings.keys())
            raise KeyError(
                f"Embedding '{key}' not found. Available: {available}"
            )
        return self.embeddings[key]

    def get_feature(self, key: str) -> np.ndarray:
        """
        Get a feature array.

        Args:
            key: Feature name (e.g., 'labels', 'cum_area')

        Returns:
            Feature array

        Raises:
            KeyError: If feature not found
        """
        if key not in self.features:
            available = list(self.features.keys())
            raise KeyError(
                f"Feature '{key}' not found. Available: {available}"
            )
        return self.features[key]

    def get_labels(self) -> np.ndarray:
        """Convenience method to get labels."""
        return self.get_feature("labels")

    def has_embedding(self, key: str) -> bool:
        """Check if embedding exists."""
        return key in self.embeddings

    def has_feature(self, key: str) -> bool:
        """Check if feature exists."""
        return key in self.features

    def get_model(self, key: str) -> Any:
        """Get a model object."""
        if key not in self.models:
            raise KeyError(f"Model '{key}' not found. Available: {list(self.models.keys())}")
        return self.models[key]

    @property
    def model_label(self) -> str:
        """Human-readable model identifier."""
        return f"{self.distribution}_{self.architecture}"

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary (excludes large arrays)."""
        return {
            "architecture": self.architecture,
            "distribution": self.distribution,
            "output_dir": str(self.output_dir) if self.output_dir else None,
            "available_embeddings": list(self.embeddings.keys()),
            "available_features": list(self.features.keys()),
            "available_models": list(self.models.keys()),
            "metadata": self.metadata,
        }

    def __repr__(self) -> str:
        emb_keys = list(self.embeddings.keys())
        feat_keys = list(self.features.keys())
        return (
            f"AnalysisContext("
            f"model={self.model_label}, "
            f"embeddings={emb_keys}, "
            f"features={feat_keys})"
        )


@dataclass
class StageResult:
    """
    Standardized result from an analysis stage.

    Attributes:
        stage_name: Unique identifier for this stage
        success: Whether the stage completed successfully
        metrics: Scalar metrics (for reporting/logging)
        artifacts: Non-scalar results (arrays, dataframes, etc.)
        metadata: Additional information (timing, parameters, etc.)
        error: Error message if success=False
    """

    stage_name: str
    success: bool = True
    metrics: Dict[str, float] = field(default_factory=dict)
    artifacts: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)
    error: Optional[str] = None

    def add_metric(self, key: str, value: float):
        """Add a scalar metric."""
        self.metrics[key] = float(value)

    def add_artifact(self, key: str, value: Any):
        """Add a non-scalar artifact."""
        self.artifacts[key] = value

    def add_metadata(self, key: str, value: Any):
        """Add metadata."""
        self.metadata[key] = value

    def to_dict(self, include_artifacts: bool = False) -> Dict[str, Any]:
        """
        Convert to dictionary.

        Args:
            include_artifacts: If False, exclude large artifacts

        Returns:
            Dictionary representation
        """
        d = {
            "stage_name": self.stage_name,
            "success": self.success,
            "metrics": self.metrics.copy(),
            "metadata": self.metadata.copy(),
        }

        if self.error:
            d["error"] = self.error

        if include_artifacts:
            # Filter out numpy arrays and other non-serializable objects
            d["artifacts"] = {
                k: v for k, v in self.artifacts.items()
                if not isinstance(v, (np.ndarray, np.generic))
            }

        return d

    def save_json(self, path: Path):
        """
        Save result as JSON.

        Args:
            path: Output path
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        # Convert numpy types to native Python
        def convert(obj):
            if isinstance(obj, (np.integer, np.floating)):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, dict):
                return {k: convert(v) for k, v in obj.items()}
            elif isinstance(obj, (list, tuple)):
                return [convert(item) for item in obj]
            return obj

        data = convert(self.to_dict(include_artifacts=False))

        with open(path, 'w') as f:
            json.dump(data, f, indent=2)

    def __repr__(self) -> str:
        status = "✓" if self.success else "✗"
        n_metrics = len(self.metrics)
        n_artifacts = len(self.artifacts)
        return (
            f"StageResult({status} {self.stage_name}: "
            f"{n_metrics} metrics, {n_artifacts} artifacts)"
        )


def create_backward_compatible_bundle(context: AnalysisContext) -> Dict[str, Any]:
    """
    Create a bundle compatible with old Embedding_analysis.output_dict format.

    This helps migrate existing code that expects the old format.

    Args:
        context: AnalysisContext

    Returns:
        Dictionary in old format
    """
    bundle = {}

    # Embeddings
    if "uniform" in context.embeddings:
        bundle["Z_uniform"] = context.embeddings["uniform"]
    if "zipfian" in context.embeddings:
        bundle["Z_zipfian"] = context.embeddings["zipfian"]

    # Features (duplicate for uniform/zipfian keys)
    for key in ["labels", "cum_area", "convex_hull", "density"]:
        if key in context.features:
            # Map to old naming
            old_key = key
            if key == "convex_hull":
                old_key = "CH"
            elif key == "cum_area":
                old_key = "cumArea"

            bundle[f"{old_key}_uniform"] = context.features[key]
            bundle[f"{old_key}_zipfian"] = context.features[key]

    return bundle
