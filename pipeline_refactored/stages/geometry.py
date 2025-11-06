"""Geometric analysis stage - RSA, RDM, monotonicity, partial RSA."""

from pathlib import Path
from typing import Dict, Any, List, Tuple
import sys
import numpy as np
import pandas as pd
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scipy.spatial.distance import pdist, squareform
from scipy.stats import spearmanr
from statsmodels.stats.multitest import multipletests

from src.analyses.monotonicity_viz import (
    compute_class_centroids,
    pairwise_centroid_distances,
    plot_distance_vs_deltaN,
    plot_violin_by_deltaN,
    plot_centroid_heatmap,
    plot_ordinal_trajectory_1d,
    plot_ordinal_trajectory_2d,
    save_deltaN_stats_csv,
    plot_outlier_pairs,
    save_outlier_pairs_csv,
)


class GeometryStage:
    """Stage for geometric analyses: RSA, RDM, monotonicity, partial RSA.

    This stage analyzes the geometric structure of embeddings across layers.
    """

    name = "geometry"

    def is_enabled(self, settings: Dict[str, Any]) -> bool:
        # Enabled if any sub-analysis is enabled
        return (settings.get('rsa', {}).get('enabled', False) or
                settings.get('rdm', {}).get('enabled', False) or
                settings.get('monotonicity', {}).get('enabled', False) or
                settings.get('partial_rsa', {}).get('enabled', False))

    def _get_model_device(self, model_obj) -> torch.device:
        """Infer device from model."""
        try:
            import torch.nn as nn
            if isinstance(model_obj, nn.Module):
                try:
                    return next(model_obj.parameters()).device
                except StopIteration:
                    pass
        except Exception:
            pass

        try:
            layers = getattr(model_obj, "layers", [])
            if layers:
                first_rbm = layers[0]
                for attr_name in ("W", "c", "b", "weights"):
                    attr_val = getattr(first_rbm, attr_name, None)
                    if isinstance(attr_val, torch.Tensor):
                        return attr_val.device
        except Exception:
            pass
        return torch.device("cpu")

    def _extract_layer_embeddings(self, ctx: Any, layers: List[int]) -> Dict[int, np.ndarray]:
        """Extract embeddings for specified layers."""
        dist_name = ctx.spec.distribution
        model_sel = (
            ctx.get_model("uniform")
            if dist_name == "uniform"
            else ctx.get_model("zipfian")
        )

        model_layers = getattr(model_sel, "layers", [])
        if not model_layers:
            return {}

        device = self._get_model_device(model_sel)
        inputs_cpu = ctx.base_batch
        layer_embeddings = {}

        with torch.no_grad():
            inputs_device = inputs_cpu.to(device).view(inputs_cpu.shape[0], -1)
            cur = inputs_device
            for li, rbm in enumerate(model_layers, start=1):
                cur = rbm.forward(cur)
                if li in layers:
                    layer_embeddings[li] = cur.detach().cpu().numpy()
            del inputs_device

        return layer_embeddings

    def run(self, ctx: Any, settings: Dict[str, Any], output_dir: Path) -> None:
        """Run geometric analyses on specified layers."""
        # Determine which layers to analyze
        layers_config = settings.get('layers', 'all')
        if layers_config == 'all':
            dist_name = ctx.spec.distribution
            model_sel = (
                ctx.get_model("uniform")
                if dist_name == "uniform"
                else ctx.get_model("zipfian")
            )
            model_layers = getattr(model_sel, "layers", [])
            layers = list(range(1, len(model_layers) + 1))
        elif layers_config == 'top':
            dist_name = ctx.spec.distribution
            model_sel = (
                ctx.get_model("uniform")
                if dist_name == "uniform"
                else ctx.get_model("zipfian")
            )
            model_layers = getattr(model_sel, "layers", [])
            layers = [len(model_layers)]
        elif isinstance(layers_config, list):
            layers = [int(l) for l in layers_config]
        else:
            layers = [int(layers_config)]

        print(f"[Geometry] Analyzing layers: {layers}")

        # Extract embeddings for all layers
        layer_embeddings = self._extract_layer_embeddings(ctx, layers)

        if not layer_embeddings:
            print("[Geometry] No layers to analyze")
            return

        # Get labels and features
        labels = ctx.bundle.labels
        cumArea = ctx.bundle.cum_area
        CH = ctx.bundle.convex_hull

        # Run analyses for each layer
        for li, Zl in layer_embeddings.items():
            layer_dir = output_dir / "layers" / f"layer{li}"
            layer_dir.mkdir(parents=True, exist_ok=True)

            # Monotonicity analysis
            if settings.get('monotonicity', {}).get('enabled', False):
                self._run_monotonicity(Zl, labels, layer_dir, li, ctx)

            # RSA analysis
            if settings.get('rsa', {}).get('enabled', False):
                self._run_rsa(Zl, labels, cumArea, CH, layer_dir, li, ctx, settings)

            # RDM analysis
            if settings.get('rdm', {}).get('enabled', False):
                self._run_rdm(Zl, labels, layer_dir, li, ctx, settings)

            # Partial RSA
            if settings.get('partial_rsa', {}).get('enabled', False):
                self._run_partial_rsa(Zl, labels, cumArea, CH, layer_dir, li, ctx, settings)

    def _run_monotonicity(self, Z: np.ndarray, labels: np.ndarray,
                         layer_dir: Path, layer_idx: int, ctx: Any):
        """Run monotonicity analysis for one layer."""
        mono_dir = layer_dir / "monotonicity"
        mono_dir.mkdir(parents=True, exist_ok=True)

        # Compute centroids and distances
        centroids, classes = compute_class_centroids(Z, labels)
        D = pairwise_centroid_distances(centroids, metric="euclidean")

        # Generate plots
        stats1 = plot_distance_vs_deltaN(D, classes, mono_dir / "deltaN_vs_distance.png")
        plot_violin_by_deltaN(D, classes, mono_dir / "violin_by_deltaN.png")
        plot_centroid_heatmap(D, classes, mono_dir / "centroid_heatmap.png")
        stats2 = plot_ordinal_trajectory_1d(centroids, classes, mono_dir / "ordinal_trajectory_1d.png")
        plot_ordinal_trajectory_2d(centroids, classes, mono_dir / "ordinal_trajectory_2d.png")

        # Save stats
        save_deltaN_stats_csv(D, classes, mono_dir / "deltaN_stats.csv")

        # Handle outliers
        outliers = stats1.get("outliers", []) if isinstance(stats1, dict) else []
        if outliers:
            dataset_uniform = None
            try:
                dataset_uniform = ctx.dataset_mgr.get_raw_dataset("uniform")
            except Exception:
                dataset_uniform = None
            plot_outlier_pairs(
                dataset_uniform, centroids, classes, outliers,
                mono_dir / "outlier_pairs.png"
            )
            save_outlier_pairs_csv(outliers, mono_dir / "outlier_pairs.csv")

        # Summary
        stats_summary = {k: v for k, v in stats1.items() if k != "outliers"} if isinstance(stats1, dict) else {}
        stats_summary.update(stats2 if isinstance(stats2, dict) else {})
        if stats_summary:
            pd.DataFrame([stats_summary]).to_csv(mono_dir / "monotonicity_summary.csv", index=False)

        # WandB logging
        if ctx.wandb_run:
            try:
                import wandb
                payload = {
                    f"monotonicity/layer{layer_idx}/deltaN_vs_distance": wandb.Image(str(mono_dir / "deltaN_vs_distance.png")),
                    f"monotonicity/layer{layer_idx}/violin": wandb.Image(str(mono_dir / "violin_by_deltaN.png")),
                    f"monotonicity/layer{layer_idx}/heatmap": wandb.Image(str(mono_dir / "centroid_heatmap.png")),
                }
                if outliers:
                    payload[f"monotonicity/layer{layer_idx}/outliers"] = wandb.Image(str(mono_dir / "outlier_pairs.png"))
                ctx.wandb_run.log(payload)
            except Exception:
                pass

        r2_val = stats_summary.get("r2")
        r2_msg = f"{r2_val:.3f}" if isinstance(r2_val, (int, float)) else "N/A"
        print(f"[Monotonicity] Layer {layer_idx}: R²={r2_msg}")

    def _run_rsa(self, Z: np.ndarray, labels: np.ndarray, cumArea: np.ndarray,
                CH: np.ndarray, layer_dir: Path, layer_idx: int, ctx: Any, settings: Dict):
        """Run RSA (Representational Similarity Analysis) for one layer."""
        rsa_cfg = settings.get('rsa', {})
        metric = rsa_cfg.get('metric', 'cosine')
        alpha = float(rsa_cfg.get('alpha', 0.01))
        dump_rdms = rsa_cfg.get('dump_rdms', False)

        rsa_dir = layer_dir / "rsa"
        rsa_dir.mkdir(parents=True, exist_ok=True)

        # Compute brain RDM (from embeddings)
        if metric == 'cosine':
            normed = Z / np.linalg.norm(Z, axis=1, keepdims=True)
            brain_rdm = pdist(normed, metric='cosine')
        else:
            brain_rdm = pdist(Z, metric=metric)

        # Compute model RDMs (from features)
        model_rdms = {
            'labels': pdist(labels.reshape(-1, 1), metric='euclidean'),
            'cumArea': pdist(cumArea.reshape(-1, 1), metric='euclidean'),
            'CH': pdist(CH.reshape(-1, 1), metric='euclidean'),
        }

        # Compute correlations
        results = []
        for feat_name, model_rdm in model_rdms.items():
            rho, pval = spearmanr(brain_rdm, model_rdm)
            results.append({
                'feature': feat_name,
                'spearman_rho': rho,
                'p_value': pval,
            })

        # Save results
        df_rsa = pd.DataFrame(results)

        # FDR correction
        if len(results) > 1:
            _, pvals_corrected, _, _ = multipletests(df_rsa['p_value'], alpha=alpha, method='fdr_bh')
            df_rsa['p_value_corrected'] = pvals_corrected
            df_rsa['significant'] = pvals_corrected < alpha

        df_rsa.to_csv(rsa_dir / "rsa_results.csv", index=False)

        # Optionally save RDMs
        if dump_rdms:
            np.save(rsa_dir / "brain_rdm.npy", brain_rdm)
            for feat_name, rdm in model_rdms.items():
                np.save(rsa_dir / f"model_rdm_{feat_name}.npy", rdm)

        print(f"[RSA] Layer {layer_idx}: labels={results[0]['spearman_rho']:.3f}, "
              f"cumArea={results[1]['spearman_rho']:.3f}, CH={results[2]['spearman_rho']:.3f}")

    def _run_rdm(self, Z: np.ndarray, labels: np.ndarray, layer_dir: Path,
                layer_idx: int, ctx: Any, settings: Dict):
        """Compute and save RDM matrices."""
        rdm_cfg = settings.get('rdm', {})
        metrics = rdm_cfg.get('metrics', ['cosine', 'euclidean'])

        rdm_dir = layer_dir / "rdm"
        rdm_dir.mkdir(parents=True, exist_ok=True)

        for metric in metrics:
            if metric == 'cosine':
                normed = Z / (np.linalg.norm(Z, axis=1, keepdims=True) + 1e-10)
                rdm_vec = pdist(normed, metric='cosine')
            else:
                rdm_vec = pdist(Z, metric=metric)

            rdm_matrix = squareform(rdm_vec)
            np.save(rdm_dir / f"rdm_{metric}.npy", rdm_matrix)
            pd.DataFrame(rdm_matrix).to_csv(rdm_dir / f"rdm_{metric}.csv", index=False)

        print(f"[RDM] Layer {layer_idx}: Saved {len(metrics)} RDM matrices")

    def _run_partial_rsa(self, Z: np.ndarray, labels: np.ndarray,
                        cumArea: np.ndarray, CH: np.ndarray,
                        layer_dir: Path, layer_idx: int, ctx: Any, settings: Dict):
        """Run partial RSA (controlling for confounds)."""
        partial_dir = layer_dir / "partial_rsa"
        partial_dir.mkdir(parents=True, exist_ok=True)

        # Compute RDMs
        brain_rdm = pdist(Z, metric='euclidean')
        label_rdm = pdist(labels.reshape(-1, 1), metric='euclidean')
        area_rdm = pdist(cumArea.reshape(-1, 1), metric='euclidean')
        ch_rdm = pdist(CH.reshape(-1, 1), metric='euclidean')

        # Partial correlation: labels controlling for cumArea
        rho_raw, _ = spearmanr(brain_rdm, label_rdm)
        rho_confound, _ = spearmanr(brain_rdm, area_rdm)
        rho_between, _ = spearmanr(label_rdm, area_rdm)

        # Simple partial correlation formula
        partial_rho = (rho_raw - rho_confound * rho_between) / np.sqrt((1 - rho_confound**2) * (1 - rho_between**2))

        results = {
            'partial_rho_labels_control_area': partial_rho,
            'raw_rho_labels': rho_raw,
            'raw_rho_area': rho_confound,
        }

        pd.DataFrame([results]).to_csv(partial_dir / "partial_rsa_results.csv", index=False)
        print(f"[Partial RSA] Layer {layer_idx}: partial_rho={partial_rho:.3f}")
