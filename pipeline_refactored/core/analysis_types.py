"""Core types for analysis pipeline.

This module contains dataclasses that were previously in analyze.py.
Now they're in the clean pipeline_refactored structure with zero dependencies.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Any, Optional


@dataclass
class ModelSpec:
    """Specification for a single model to analyze.

    Attributes:
        arch_name: Architecture name (e.g., "iDBN_1500_500")
        distribution: Training distribution ("uniform" or "zipfian")
        dataset_path: Path to dataset directory
        dataset_name: Dataset filename
        model_uniform: Path to uniform-trained model
        model_zipfian: Path to zipfian-trained model
        val_size: Validation split size
    """

    arch_name: str
    distribution: str
    dataset_path: Path
    dataset_name: str
    model_uniform: Path
    model_zipfian: Path
    val_size: float = 0.05

    @classmethod
    def from_config(cls, raw_cfg: Dict[str, Any], project_root: Path) -> "ModelSpec":
        """Create ModelSpec from Hydra config dict.

        Args:
            raw_cfg: Raw config dictionary from Hydra
            project_root: Project root path for resolving relative paths

        Returns:
            ModelSpec instance
        """
        def _abs(value: Optional[str]) -> Path:
            """Convert to absolute path."""
            if value is None:
                return project_root
            candidate = Path(value)
            return candidate if candidate.is_absolute() else (project_root / candidate).resolve()

        return cls(
            arch_name=str(raw_cfg["arch"]),
            distribution=str(raw_cfg["distribution"]),
            dataset_path=_abs(raw_cfg["dataset_path"]),
            dataset_name=str(raw_cfg.get("dataset_name", "stimuli_dataset.npz")),
            model_uniform=_abs(raw_cfg["model_uniform"]),
            model_zipfian=_abs(raw_cfg["model_zipfian"]),
            val_size=float(raw_cfg.get("val_size", 0.05)),
        )


@dataclass
class AnalysisSettings:
    """Configuration for all analysis stages.

    Each attribute corresponds to a stage or group of analyses.
    Values are dictionaries containing stage-specific settings.
    """

    # Core analyses
    probing: Dict[str, Any] = field(default_factory=dict)
    rsa: Dict[str, Any] = field(default_factory=dict)
    rdm: Dict[str, Any] = field(default_factory=dict)
    monotonicity: Dict[str, Any] = field(default_factory=dict)
    partial_rsa: Dict[str, Any] = field(default_factory=dict)

    # Reconstruction metrics
    mse: Dict[str, Any] = field(default_factory=dict)
    afp: Dict[str, Any] = field(default_factory=dict)
    ssim: Dict[str, Any] = field(default_factory=dict)

    # Dimensionality
    pca_geometry: Dict[str, Any] = field(default_factory=dict)
    pca_report: Dict[str, Any] = field(default_factory=dict)
    tsne: Dict[str, Any] = field(default_factory=dict)
    umap: Dict[str, Any] = field(default_factory=dict)
    reductions: Dict[str, Any] = field(default_factory=dict)  # Legacy
    traversal: Dict[str, Any] = field(default_factory=dict)   # Legacy

    # Comparison & behavioral
    cka: Dict[str, Any] = field(default_factory=dict)
    behavioral: Dict[str, Any] = field(default_factory=dict)

    # Global settings
    layers: Any = 'top'  # 'top' | 'all' | List[int]

    @classmethod
    def from_cfg(cls, cfg: Dict[str, Any]) -> "AnalysisSettings":
        """Create AnalysisSettings from config dict.

        Args:
            cfg: Configuration dictionary (from Hydra or YAML)

        Returns:
            AnalysisSettings instance
        """
        try:
            from omegaconf import DictConfig, OmegaConf

            def _to_dict(name: str) -> Dict[str, Any]:
                """Convert config section to dict."""
                section = cfg.get(name, {})
                if isinstance(section, DictConfig):
                    return OmegaConf.to_container(section, resolve=True)  # type: ignore
                return dict(section) if isinstance(section, dict) else {}
        except ImportError:
            # Fallback if omegaconf not available
            def _to_dict(name: str) -> Dict[str, Any]:
                section = cfg.get(name, {})
                return dict(section) if isinstance(section, dict) else {}

        return cls(
            probing=_to_dict("probing"),
            rsa=_to_dict("rsa"),
            rdm=_to_dict("rdm"),
            monotonicity=_to_dict("monotonicity"),
            partial_rsa=_to_dict("partial_rsa"),
            mse=_to_dict("mse"),
            afp=_to_dict("afp"),
            ssim=_to_dict("ssim"),
            pca_geometry=_to_dict("pca_geometry"),
            pca_report=_to_dict("pca_report"),
            tsne=_to_dict("tsne"),
            umap=_to_dict("umap"),
            reductions=_to_dict("reductions"),
            traversal=_to_dict("traversal"),
            cka=_to_dict("cka"),
            behavioral=_to_dict("behavioral"),
            layers=cfg.get("layers", "top"),
        )
