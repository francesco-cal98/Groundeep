"""Reconstruction metrics stage - MSE, AFP, SSIM."""

from pathlib import Path
from typing import Dict, Any, List
import sys
import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.analyses.mse_viz import (
    compute_sample_mse,
    prepare_mse_dataframe,
    plot_mse_heatmap,
    plot_mse_vs_numerosity,
    save_regression_results as save_mse_regression,
)
from src.analyses.afp_viz import (
    compute_sample_afp,
    prepare_afp_dataframe,
    plot_afp_heatmap,
    plot_afp_vs_numerosity,
    save_afp_regression_results,
)
from src.analyses.ssim_viz import (
    compute_sample_ssim,
    prepare_ssim_dataframe,
    plot_ssim_heatmap,
    plot_ssim_vs_numerosity,
    save_ssim_regression_results,
)


class ReconstructionStage:
    """Stage for reconstruction quality metrics: MSE, AFP, SSIM.

    This stage analyzes how well the model reconstructs inputs after encoding.
    """

    name = "reconstruction"

    def is_enabled(self, settings: Dict[str, Any]) -> bool:
        # Enabled if any sub-metric is enabled
        return (settings.get('mse', {}).get('enabled', False) or
                settings.get('afp', {}).get('enabled', False) or
                settings.get('ssim', {}).get('enabled', False))

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

    def _reconstruct_from_layer(self, ctx: Any, layer_idx: int) -> np.ndarray:
        """Reconstruct inputs by encoding to specified layer and decoding back."""
        dist_name = ctx.spec.distribution
        model_sel = (
            ctx.get_model("uniform")
            if dist_name == "uniform"
            else ctx.get_model("zipfian")
        )

        model_layers = getattr(model_sel, "layers", [])
        if layer_idx > len(model_layers):
            raise ValueError(f"Layer {layer_idx} exceeds model depth {len(model_layers)}")

        device = self._get_model_device(model_sel)
        inputs_cpu = ctx.base_batch

        with torch.no_grad():
            # Encode
            inputs_device = inputs_cpu.to(device).view(inputs_cpu.shape[0], -1)
            cur = inputs_device
            for rbm in model_layers[:layer_idx]:
                cur = rbm.forward(cur)

            # Decode
            for rbm in reversed(model_layers[:layer_idx]):
                cur = rbm.backward(cur)

            reconstructed = cur.detach().cpu().numpy()

        return reconstructed

    def run(self, ctx: Any, settings: Dict[str, Any], output_dir: Path) -> None:
        """Run reconstruction metrics on specified layers."""
        # Determine which layers to analyze
        layers_config = settings.get('layers', 'top')
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

        print(f"[Reconstruction] Analyzing layers: {layers}")

        # Get features
        labels = ctx.bundle.labels
        cumArea = ctx.bundle.cum_area
        CH = ctx.bundle.convex_hull
        density = getattr(ctx.bundle, 'density', None)

        # Get original inputs (need 2D images for AFP/SSIM)
        originals_flat = ctx.base_batch.cpu().numpy()
        # Try to reshape to 2D images
        try:
            # Assume square images
            n_samples = originals_flat.shape[0]
            img_size = int(np.sqrt(originals_flat.shape[1]))
            if img_size * img_size == originals_flat.shape[1]:
                originals_2d = originals_flat.reshape(n_samples, img_size, img_size)
            else:
                originals_2d = originals_flat
        except Exception:
            originals_2d = originals_flat

        n_bins = settings.get('n_bins', 5)

        # Run metrics for each layer
        for li in layers:
            layer_dir = output_dir / "reconstruction" / f"layer{li}"
            layer_dir.mkdir(parents=True, exist_ok=True)

            # Reconstruct from this layer
            try:
                reconstructed_flat = self._reconstruct_from_layer(ctx, li)
            except Exception as exc:
                print(f"[Reconstruction] Layer {li}: reconstruction failed ({exc})")
                continue

            # Reshape reconstructed to match originals
            try:
                if originals_2d.ndim == 3:
                    n_samples = reconstructed_flat.shape[0]
                    img_size = originals_2d.shape[1]
                    reconstructed_2d = reconstructed_flat.reshape(n_samples, img_size, img_size)
                else:
                    reconstructed_2d = reconstructed_flat
            except Exception:
                reconstructed_2d = reconstructed_flat

            # MSE
            if settings.get('mse', {}).get('enabled', False):
                self._run_mse(originals_flat, reconstructed_flat, labels, cumArea, CH,
                            density, layer_dir, li, ctx, n_bins)

            # AFP
            if settings.get('afp', {}).get('enabled', False):
                if originals_2d.ndim >= 3:
                    self._run_afp(originals_2d, reconstructed_2d, labels, cumArea, CH,
                                density, layer_dir, li, ctx, n_bins)
                else:
                    print(f"[AFP] Layer {li}: skipped (need 2D images)")

            # SSIM
            if settings.get('ssim', {}).get('enabled', False):
                if originals_2d.ndim >= 3:
                    self._run_ssim(originals_2d, reconstructed_2d, labels, cumArea, CH,
                                 density, layer_dir, li, ctx, n_bins)
                else:
                    print(f"[SSIM] Layer {li}: skipped (need 2D images)")

    def _run_mse(self, originals: np.ndarray, reconstructed: np.ndarray,
                labels: np.ndarray, cumArea: np.ndarray, CH: np.ndarray,
                density: np.ndarray, layer_dir: Path, layer_idx: int, ctx: Any, n_bins: int):
        """Run MSE analysis for one layer."""
        mse_dir = layer_dir / "mse"
        mse_dir.mkdir(parents=True, exist_ok=True)

        # Compute MSE per sample
        mse_values = compute_sample_mse(originals, reconstructed)

        # Prepare dataframe
        df_mse, info = prepare_mse_dataframe(
            mse_values, labels, cumArea, CH, density, n_bins=n_bins
        )
        df_mse.to_csv(mse_dir / "mse_data.csv", index=False)

        # Heatmap: numerosity x cumarea_bin
        plot_mse_heatmap(
            df_mse,
            row_col="numerosity",
            col_col="cumarea_bin",
            out_path=mse_dir / "mse_heatmap_cumarea.png",
            title=f"MSE — {ctx.spec.arch_name} ({ctx.spec.distribution}) [Layer {layer_idx}]",
            row_label="Numerosity",
            col_label="Cumulative Area Bin",
        )

        # Heatmap: numerosity x convex_hull_bin
        plot_mse_heatmap(
            df_mse,
            row_col="numerosity",
            col_col="convex_hull_bin",
            out_path=mse_dir / "mse_heatmap_hull.png",
            title=f"MSE — {ctx.spec.arch_name} ({ctx.spec.distribution}) [Layer {layer_idx}]",
            row_label="Numerosity",
            col_label="Convex Hull Bin",
        )

        # Line plot: MSE vs numerosity by cumarea_bin
        plot_mse_vs_numerosity(
            df_mse,
            feature_col="cumarea_bin",
            feature_label="Cumulative Area",
            out_path=mse_dir / "mse_vs_numerosity_cumarea.png",
            title=f"MSE vs Numerosity — {ctx.spec.arch_name} [Layer {layer_idx}]",
        )

        # Regression
        save_mse_regression(df_mse, mse_dir)

        # Summary stats
        mean_mse = float(np.mean(mse_values))
        print(f"[MSE] Layer {layer_idx}: mean={mean_mse:.5f}")

        # WandB logging
        if ctx.wandb_run:
            try:
                import wandb
                ctx.wandb_run.log({
                    f"reconstruction/layer{layer_idx}/mse_mean": mean_mse,
                    f"reconstruction/layer{layer_idx}/mse_heatmap_cumarea": wandb.Image(str(mse_dir / "mse_heatmap_cumarea.png")),
                    f"reconstruction/layer{layer_idx}/mse_heatmap_hull": wandb.Image(str(mse_dir / "mse_heatmap_hull.png")),
                })
            except Exception:
                pass

    def _run_afp(self, originals: np.ndarray, reconstructed: np.ndarray,
                labels: np.ndarray, cumArea: np.ndarray, CH: np.ndarray,
                density: np.ndarray, layer_dir: Path, layer_idx: int, ctx: Any, n_bins: int):
        """Run AFP (Area Fraction Preserved) analysis for one layer."""
        afp_dir = layer_dir / "afp"
        afp_dir.mkdir(parents=True, exist_ok=True)

        # Compute AFP per sample
        afp_values = compute_sample_afp(originals, reconstructed)

        # Prepare dataframe
        df_afp, info = prepare_afp_dataframe(
            afp_values, labels, cumArea, CH, density, n_bins=n_bins
        )
        df_afp.to_csv(afp_dir / "afp_data.csv", index=False)

        # Heatmap: numerosity x cumarea_bin
        plot_afp_heatmap(
            df_afp,
            row_col="numerosity",
            col_col="cumarea_bin",
            out_path=afp_dir / "afp_heatmap_cumarea.png",
            title=f"AFP — {ctx.spec.arch_name} ({ctx.spec.distribution}) [Layer {layer_idx}]",
            row_label="Numerosity",
            col_label="Cumulative Area Bin",
        )

        # Heatmap: numerosity x convex_hull_bin
        plot_afp_heatmap(
            df_afp,
            row_col="numerosity",
            col_col="convex_hull_bin",
            out_path=afp_dir / "afp_heatmap_hull.png",
            title=f"AFP — {ctx.spec.arch_name} ({ctx.spec.distribution}) [Layer {layer_idx}]",
            row_label="Numerosity",
            col_label="Convex Hull Bin",
        )

        # Line plot: AFP vs numerosity by cumarea_bin
        plot_afp_vs_numerosity(
            df_afp,
            feature_col="cumarea_bin",
            feature_label="Cumulative Area",
            out_path=afp_dir / "afp_vs_numerosity_cumarea.png",
            title=f"AFP vs Numerosity — {ctx.spec.arch_name} [Layer {layer_idx}]",
        )

        # Regression
        save_afp_regression_results(df_afp, afp_dir)

        # Summary stats
        mean_afp = float(np.mean(afp_values))
        print(f"[AFP] Layer {layer_idx}: mean={mean_afp:.5f}")

        # WandB logging
        if ctx.wandb_run:
            try:
                import wandb
                ctx.wandb_run.log({
                    f"reconstruction/layer{layer_idx}/afp_mean": mean_afp,
                    f"reconstruction/layer{layer_idx}/afp_heatmap_cumarea": wandb.Image(str(afp_dir / "afp_heatmap_cumarea.png")),
                    f"reconstruction/layer{layer_idx}/afp_heatmap_hull": wandb.Image(str(afp_dir / "afp_heatmap_hull.png")),
                })
            except Exception:
                pass

    def _run_ssim(self, originals: np.ndarray, reconstructed: np.ndarray,
                 labels: np.ndarray, cumArea: np.ndarray, CH: np.ndarray,
                 density: np.ndarray, layer_dir: Path, layer_idx: int, ctx: Any, n_bins: int):
        """Run SSIM (Structural Similarity Index) analysis for one layer."""
        ssim_dir = layer_dir / "ssim"
        ssim_dir.mkdir(parents=True, exist_ok=True)

        # Compute SSIM per sample
        ssim_values = compute_sample_ssim(originals, reconstructed)

        # Prepare dataframe
        df_ssim, info = prepare_ssim_dataframe(
            ssim_values, labels, cumArea, CH, density, n_bins=n_bins
        )
        df_ssim.to_csv(ssim_dir / "ssim_data.csv", index=False)

        # Heatmap: numerosity x cumarea_bin
        plot_ssim_heatmap(
            df_ssim,
            row_col="numerosity",
            col_col="cumarea_bin",
            out_path=ssim_dir / "ssim_heatmap_cumarea.png",
            title=f"SSIM — {ctx.spec.arch_name} ({ctx.spec.distribution}) [Layer {layer_idx}]",
            row_label="Numerosity",
            col_label="Cumulative Area Bin",
        )

        # Heatmap: numerosity x convex_hull_bin
        plot_ssim_heatmap(
            df_ssim,
            row_col="numerosity",
            col_col="convex_hull_bin",
            out_path=ssim_dir / "ssim_heatmap_hull.png",
            title=f"SSIM — {ctx.spec.arch_name} ({ctx.spec.distribution}) [Layer {layer_idx}]",
            row_label="Numerosity",
            col_label="Convex Hull Bin",
        )

        # Line plot: SSIM vs numerosity by cumarea_bin
        plot_ssim_vs_numerosity(
            df_ssim,
            feature_col="cumarea_bin",
            feature_label="Cumulative Area",
            out_path=ssim_dir / "ssim_vs_numerosity_cumarea.png",
            title=f"SSIM vs Numerosity — {ctx.spec.arch_name} [Layer {layer_idx}]",
        )

        # Regression
        save_ssim_regression_results(df_ssim, ssim_dir)

        # Summary stats
        mean_ssim = float(np.mean(ssim_values))
        print(f"[SSIM] Layer {layer_idx}: mean={mean_ssim:.5f}")

        # WandB logging
        if ctx.wandb_run:
            try:
                import wandb
                ctx.wandb_run.log({
                    f"reconstruction/layer{layer_idx}/ssim_mean": mean_ssim,
                    f"reconstruction/layer{layer_idx}/ssim_heatmap_cumarea": wandb.Image(str(ssim_dir / "ssim_heatmap_cumarea.png")),
                    f"reconstruction/layer{layer_idx}/ssim_heatmap_hull": wandb.Image(str(ssim_dir / "ssim_heatmap_hull.png")),
                })
            except Exception:
                pass
