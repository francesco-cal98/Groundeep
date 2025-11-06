#!/usr/bin/env python3
"""
Modular analysis pipeline for GROUNDEEP.

This is a cleaner entry point that uses the stage system while
reusing all existing analysis functions from analyze.py.

Usage:
    python src/main_scripts/analyze_modular.py
    (or use Hydra: python src/main_scripts/analyze_modular.py --config-name=analysis)
"""

import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Any, Optional, Tuple

import hydra
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader

# Ensure project root in path
THIS_FILE = Path(__file__).resolve()
PROJECT_ROOT = THIS_FILE.parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Import from clean pipeline_refactored (ZERO dependency on analyze.py)
from pipeline_refactored.core.analysis_types import ModelSpec, AnalysisSettings
from pipeline_refactored.core.analysis_helpers import (
    EmbeddingBundle,
    maybe_init_wandb,
    infer_chw_from_input,
    infer_hw_from_batch,
)
from pipeline_refactored.core import (
    DatasetManager,
    ModelManager,
    EmbeddingExtractor,
    AnalysisContext,
)

# Keep only necessary imports from src
from src.datasets.uniform_dataset import plot_label_histogram

# Import stage system
from pipeline_refactored.stages import (
    PowerLawStage,
    LinearProbesStage,
    GeometryStage,
    ReconstructionStage,
    DimensionalityStage,
    CKAStage,
    BehavioralStage,
    PCADiagnosticsStage,
)


@dataclass
class ModularPipelineContext:
    """Aggregates all artefacts needed by modular stages without Embedding_analysis."""

    spec: ModelSpec
    dataset_mgr: DatasetManager
    model_mgr: ModelManager
    extractor: EmbeddingExtractor
    analysis: AnalysisContext
    bundle: EmbeddingBundle
    output_dir: Path
    seed: int
    wandb_run: Optional[Any]
    base_batch: torch.Tensor
    orig_flat: np.ndarray
    image_shape: Tuple[int, int, int]
    uniform_val_loader: DataLoader
    zipf_val_loader: Optional[DataLoader] = None

    def get_model(self, label: str) -> Any:
        return self.model_mgr.get_model(label)

    @property
    def models(self) -> Dict[str, Any]:
        return self.analysis.models


def _prepare_pipeline_context(
    spec: ModelSpec,
    output_root: Path | str,
    seed: int,
    *,
    wandb_run=None,
) -> ModularPipelineContext:
    """Instantiate dataset/model managers and precompute aligned embeddings."""
    output_root = Path(output_root)
    out_dir = output_root / spec.distribution / spec.arch_name
    out_dir.mkdir(parents=True, exist_ok=True)

    dataset_mgr = DatasetManager(
        dataset_path=str(spec.dataset_path),
        dataset_name=spec.dataset_name,
        default_val_size=spec.val_size,
    )
    # Ensure uniform split is materialised as full batch (baseline shared inputs)
    uniform_val_loader = dataset_mgr.get_dataloader("uniform", split="val", full_batch=True)
    zipf_val_loader: Optional[DataLoader] = None
    try:
        zipf_val_loader = dataset_mgr.get_dataloader("zipfian", split="val", full_batch=True)
    except Exception:
        zipf_val_loader = None

    # Extract one batch of uniform stimuli (reference inputs for all analyses)
    try:
        batch_uniform = next(iter(uniform_val_loader))
    except StopIteration as exc:
        raise RuntimeError("Uniform validation loader returned no batches") from exc

    inputs_uniform, labels_uniform = batch_uniform
    inputs_uniform = inputs_uniform.to(torch.float32)
    base_batch = inputs_uniform.detach().cpu()
    labels_np = labels_uniform.detach().cpu().numpy()

    model_mgr = ModelManager()
    model_mgr.load_model(str(spec.model_uniform), label="uniform")
    model_mgr.load_model(str(spec.model_zipfian), label="zipfian")
    extractor = EmbeddingExtractor(model_mgr)

    # Compute aligned embeddings (uniform inputs through both models)
    Z_uniform, Z_zipfian = extractor.extract_aligned_pair(
        "uniform",
        "zipfian",
        uniform_val_loader,
    )

    # Collect geometric features for probes/analysis from uniform dataset
    features_uniform = dataset_mgr.get_features("uniform", split="val")

    bundle = EmbeddingBundle(
        embeddings=Z_uniform if spec.distribution.lower() == "uniform" else Z_zipfian,
        labels=labels_np,
        cum_area=features_uniform.get("cum_area", np.array([])),
        convex_hull=features_uniform.get("convex_hull", np.array([])),
        inputs=base_batch,
        mean_item_size=features_uniform.get("mean_item_size"),
        density=features_uniform.get("density"),
    )

    orig_flat = base_batch.reshape(base_batch.shape[0], -1).numpy()
    try:
        channels, img_h, img_w = infer_chw_from_input(base_batch)
    except Exception:
        img_h, img_w = infer_hw_from_batch(base_batch)
        channels = 1

    analysis_ctx = AnalysisContext(
        embeddings={
            "uniform": Z_uniform,
            "zipfian": Z_zipfian,
        },
        features={
            "labels": labels_np,
            "cum_area": features_uniform.get("cum_area", np.array([])),
            "convex_hull": features_uniform.get("convex_hull", np.array([])),
            **({"density": features_uniform["density"]} if "density" in features_uniform else {}),
            **({"mean_item_size": features_uniform["mean_item_size"]} if "mean_item_size" in features_uniform else {}),
        },
        models={
            "uniform": model_mgr.get_model("uniform"),
            "zipfian": model_mgr.get_model("zipfian"),
        },
        architecture=spec.arch_name,
        distribution=spec.distribution,
        output_dir=out_dir,
        metadata={"seed": seed},
    )

    return ModularPipelineContext(
        spec=spec,
        dataset_mgr=dataset_mgr,
        model_mgr=model_mgr,
        extractor=extractor,
        analysis=analysis_ctx,
        bundle=bundle,
        output_dir=out_dir,
        seed=seed,
        wandb_run=wandb_run,
        base_batch=base_batch,
        orig_flat=orig_flat,
        image_shape=(channels, img_h, img_w),
        uniform_val_loader=uniform_val_loader,
        zipf_val_loader=zipf_val_loader,
    )


def _save_label_histograms_modular(ctx: ModularPipelineContext) -> None:
    """Persist label distributions using DatasetManager instead of Embedding_analysis."""
    hist_dir = ctx.output_dir / "label_histograms"
    hist_dir.mkdir(parents=True, exist_ok=True)

    def _dataset_labels(distribution: str) -> Optional[np.ndarray]:
        try:
            loader = ctx.dataset_mgr.get_dataloader(distribution, split="train")
        except Exception:
            return None
        subset = loader.dataset
        base = getattr(subset, "dataset", subset)
        labels = getattr(base, "labels", None)
        if labels is None:
            return None
        if hasattr(subset, "indices"):
            return np.asarray(labels)[np.asarray(subset.indices)]
        return np.asarray(labels)

    try:
        labels_uniform = _dataset_labels("uniform")
        if labels_uniform is not None and labels_uniform.size:
            plot_label_histogram(
                labels_uniform,
                title="Label Histogram (uniform)",
                save_path=hist_dir / "uniform.png",
            )
    except Exception as exc:
        print(f"[Labels] Impossibile generare histogramma uniform: {exc}")

    try:
        labels_zipf = _dataset_labels("zipfian")
        if labels_zipf is not None and labels_zipf.size:
            plot_label_histogram(
                labels_zipf,
                title="Label Histogram (zipfian)",
                save_path=hist_dir / "zipfian.png",
            )
    except Exception as exc:
        print(f"[Labels] Impossibile generare histogramma zipfian: {exc}")


def _build_probe_features_modular(
    ctx: ModularPipelineContext,
) -> Tuple[Dict[str, np.ndarray], Optional[np.ndarray], Optional[np.ndarray]]:
    """Collect features required by linear probes + correlation reports."""
    subset = ctx.uniform_val_loader.dataset
    idxs = getattr(subset, "indices", np.arange(len(subset)))
    base = getattr(subset, "dataset", subset)

    def _collect(attr: str) -> Optional[np.ndarray]:
        values = getattr(base, attr, None)
        if values is None:
            return None
        return np.asarray([values[i] for i in idxs])

    features_for_probes: Dict[str, np.ndarray] = {}
    cum_area = _collect("cumArea_list")
    convex_hull = _collect("CH_list")
    density = _collect("density_list")
    mean_size = _collect("mean_item_size_list")

    if cum_area is not None:
        features_for_probes["Cumulative Area"] = cum_area
    if convex_hull is not None:
        features_for_probes["Convex Hull"] = convex_hull
    if density is not None:
        features_for_probes["Density"] = density
    if mean_size is not None:
        features_for_probes["Mean Item Size"] = mean_size
    if ctx.bundle.labels is not None and ctx.bundle.labels.size:
        features_for_probes["Labels"] = ctx.bundle.labels

    feature_dir = ctx.output_dir / "feature_analysis"
    feature_dir.mkdir(parents=True, exist_ok=True)

    corr_features: Dict[str, np.ndarray] = {}
    if ctx.bundle.labels.size:
        corr_features["labels"] = ctx.bundle.labels.astype(float)
    if ctx.bundle.cum_area.size:
        corr_features["cumArea"] = ctx.bundle.cum_area.astype(float)
    if ctx.bundle.convex_hull.size:
        corr_features["CH"] = ctx.bundle.convex_hull.astype(float)
    if density is not None and density.size:
        corr_features["Density"] = density.astype(float)
    if mean_size is not None and mean_size.size:
        corr_features["mean_item_size"] = mean_size.astype(float)

    density_np: Optional[np.ndarray] = density.astype(float) if density is not None else None
    mean_arr: Optional[np.ndarray] = mean_size.astype(float) if mean_size is not None else None

    if corr_features:
        corr_df = pd.DataFrame(corr_features)
        rename_map = {
            "labels": "Numerosity",
            "cumArea": "Cumulative Area",
            "CH": "Convex Hull",
            "Density": "Density",
            "mean_item_size": "Mean Item Size",
        }
        corr_df.rename(columns=rename_map, inplace=True)
        if "Numerosity" in corr_df:
            mask = corr_df["Numerosity"] > 5
            corr_df = corr_df[mask]
        corr_matrix = corr_df.corr(method="kendall")
        corr_matrix.to_csv(feature_dir / f"feature_correlations_{ctx.spec.distribution}.csv")
        plt.figure(figsize=(6, 4))
        sns.heatmap(
            corr_matrix,
            annot=True,
            fmt=".2f",
            cmap="viridis",
            cbar=True,
            square=True,
        )
        plt.title(f"Feature Correlations — {ctx.spec.arch_name} ({ctx.spec.distribution})")
        plt.xticks(rotation=45, ha="right")
        plt.yticks(rotation=0)
        plt.tight_layout()
        plt.savefig(feature_dir / f"feature_correlations_{ctx.spec.distribution}.png", dpi=300)
        plt.close()

    return features_for_probes, density_np, mean_arr

def run_analysis_pipeline(
    spec: ModelSpec,
    settings: AnalysisSettings,
    output_root: Path,
    seed: int = 42,
    use_wandb: bool = False,
    wandb_project: str = None,
    wandb_run_name: str = None,
    anchor_idx: int = 0,
) -> None:
    """Run modular analysis pipeline using stages.

    This orchestrates the analysis using the stage system while
    reusing all existing functions from analyze.py.
    """

    wandb_run = maybe_init_wandb(use_wandb, wandb_project, wandb_run_name)
    print(f"\n{'='*70}")
    print(f"🔄 Analyzing: {spec.arch_name} | Distribution: {spec.distribution}")
    print(f"{'='*70}")

    try:
        # 1. Prepare context (loads models, datasets, embeddings)
        print("\n[1/11] Preparing model context...")
        ctx = _prepare_pipeline_context(spec, output_root, seed, wandb_run=wandb_run)

        # 2. Save label histograms
        print("\n[2/11] Saving label histograms...")
        _save_label_histograms_modular(ctx)

        # 3. Build probe features (needed by multiple stages)
        print("\n[3/11] Building probe features...")
        features_for_probes, density_np, mean_arr = _build_probe_features_modular(ctx)

        # 4. Run stages in order
        print("\n[4/11] Running Power-Law Scaling Stage...")
        if True:  # Always run power-law
            stage = PowerLawStage()
            stage.run(ctx, {}, ctx.output_dir)

        print("\n[5/11] Running Linear Probes Stage...")
        if settings.probing.get('enabled', False):
            stage = LinearProbesStage()
            # Need to pass features_for_probes to stage
            stage.run(
                ctx,
                {
                    'settings': settings.probing,
                    'features': features_for_probes,
                    'arch_name': spec.arch_name,
                    'dist_name': spec.distribution,
                    'seed': seed,
                    'wandb_run': wandb_run,
                },
                ctx.output_dir,
            )

        print("\n[6/11] Running Geometry Stage (RSA, RDM, Monotonicity, Partial RSA)...")
        geo_settings = {
            'layers': getattr(settings, 'layers', 'top'),
            'rsa': settings.rsa,
            'rdm': settings.rdm,
            'monotonicity': settings.monotonicity,
            'partial_rsa': settings.partial_rsa,
        }
        stage = GeometryStage()
        if stage.is_enabled(geo_settings):
            stage.run(ctx, geo_settings, ctx.output_dir)

        print("\n[7/11] Running Reconstruction Stage (MSE, AFP, SSIM)...")
        recon_settings = {
            'layers': getattr(settings, 'layers', 'top'),
            'mse': getattr(settings, 'mse', {}),
            'afp': getattr(settings, 'afp', {}),
            'ssim': getattr(settings, 'ssim', {}),
            'n_bins': getattr(settings, 'n_bins', 5),
        }
        stage = ReconstructionStage()
        if stage.is_enabled(recon_settings):
            stage.run(ctx, recon_settings, ctx.output_dir)

        print("\n[8/11] Running Dimensionality Stage (PCA, TSNE, UMAP)...")
        dim_settings = {
            'layers': getattr(settings, 'layers', 'top'),
            'pca_geometry': getattr(settings, 'pca_geometry', {}),
            'pca_report': getattr(settings, 'pca_report', {}),
            'tsne': getattr(settings, 'tsne', {}),
            'umap': getattr(settings, 'umap', {}),
        }
        stage = DimensionalityStage()
        if stage.is_enabled(dim_settings):
            stage.run(ctx, dim_settings, ctx.output_dir)

        print("\n[9/11] Running CKA Stage...")
        if settings.cka.get('enabled', False):
            stage = CKAStage()
            stage.run(ctx, settings.cka, ctx.output_dir)

        print("\n[10/11] Running Behavioral Suite Stage...")
        if settings.behavioral.get('enabled', False):
            stage = BehavioralStage()
            stage.run(ctx, settings.behavioral, ctx.output_dir)

        print("\n[11/11] Running PCA Diagnostics Stage (legacy)...")
        if settings.pca_geometry.get('enabled', False):
            stage = PCADiagnosticsStage()
            stage.run(ctx, settings.pca_geometry, ctx.output_dir)

        # TODO: Cross-model comparisons (uniform vs zipfian)
        # This feature (maybe_generate_monotonicity_comparisons) compares
        # monotonicity results across distributions. To enable it, the function
        # and its plotting helpers need to be moved from analyze.py to
        # pipeline_refactored/core/analysis_helpers.py
        # For now, this is disabled to achieve zero dependency on analyze.py

        print(f"\n{'='*70}")
        print(f"✅ Analysis complete: {ctx.output_dir}")
        print(f"{'='*70}\n")

    finally:
        if wandb_run:
            wandb_run.finish()


@hydra.main(version_base=None, config_path="../../src/configs", config_name="analysis")
def main(cfg: DictConfig) -> None:
    """Main entry point using Hydra config."""

    print("\n" + "="*70)
    print("GROUNDEEP Modular Analysis Pipeline")
    print("="*70)
    print("\nConfig:")
    print(OmegaConf.to_yaml(cfg, resolve=True))

    # Extract global settings
    seed = cfg.get('seed', 42)
    use_wandb = cfg.get('use_wandb', False)
    wandb_project = cfg.get('wandb_project', 'numerosity-offline')
    output_root = Path(cfg.get('output_root', 'results/analysis'))
    anchor_idx = cfg.get('anchor', 0)

    # Parse settings
    settings = AnalysisSettings.from_cfg(OmegaConf.to_container(cfg, resolve=True))

    # Run analysis for each model
    models = cfg.get('models', [])
    if not models:
        print("⚠️  No models specified in config!")
        return

    for model_cfg in models:
        spec = ModelSpec.from_config(
            OmegaConf.to_container(model_cfg, resolve=True),
            PROJECT_ROOT,
        )

        wandb_run_name = f"{spec.arch_name}_{spec.distribution}" if use_wandb else None

        run_analysis_pipeline(
            spec=spec,
            settings=settings,
            output_root=output_root,
            seed=seed,
            use_wandb=use_wandb,
            wandb_project=wandb_project,
            wandb_run_name=wandb_run_name,
            anchor_idx=anchor_idx,
        )


if __name__ == "__main__":
    main()
