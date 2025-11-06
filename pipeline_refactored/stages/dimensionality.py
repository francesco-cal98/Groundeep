"""Dimensionality reduction stage - PCA, TSNE, UMAP."""

from pathlib import Path
from typing import Dict, Any, List
import sys
import numpy as np
import torch
import matplotlib.pyplot as plt

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.analyses.pca_geometry import run_pca_geometry
from src.analyses.pca_report import generate_pca_decomposition_report


class DimensionalityStage:
    """Stage for dimensionality reduction: PCA geometry, TSNE, UMAP.

    This stage performs dimensionality reduction analysis on embeddings.
    """

    name = "dimensionality"

    def is_enabled(self, settings: Dict[str, Any]) -> bool:
        # Enabled if any sub-analysis is enabled
        return (settings.get('pca_geometry', {}).get('enabled', False) or
                settings.get('pca_report', {}).get('enabled', False) or
                settings.get('tsne', {}).get('enabled', False) or
                settings.get('umap', {}).get('enabled', False))

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
        """Run dimensionality reduction analyses on specified layers."""
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

        print(f"[Dimensionality] Analyzing layers: {layers}")

        # Extract embeddings for all layers
        layer_embeddings = self._extract_layer_embeddings(ctx, layers)

        if not layer_embeddings:
            print("[Dimensionality] No layers to analyze")
            return

        # Get labels
        labels = ctx.bundle.labels

        # Run analyses for each layer
        for li, Z in layer_embeddings.items():
            layer_dir = output_dir / "dimensionality" / f"layer{li}"
            layer_dir.mkdir(parents=True, exist_ok=True)

            # PCA geometry
            if settings.get('pca_geometry', {}).get('enabled', False):
                self._run_pca_geometry(Z, labels, layer_dir, li, ctx, settings)

            # PCA decomposition report
            if settings.get('pca_report', {}).get('enabled', False):
                self._run_pca_report(Z, labels, layer_dir, li, ctx, settings)

            # TSNE
            if settings.get('tsne', {}).get('enabled', False):
                self._run_tsne(Z, labels, layer_dir, li, ctx, settings)

            # UMAP
            if settings.get('umap', {}).get('enabled', False):
                self._run_umap(Z, labels, layer_dir, li, ctx, settings)

    def _run_pca_geometry(self, Z: np.ndarray, labels: np.ndarray,
                         layer_dir: Path, layer_idx: int, ctx: Any, settings: Dict):
        """Run PCA geometry analysis for one layer."""
        pca_geo_dir = layer_dir / "pca_geometry"
        pca_geo_dir.mkdir(parents=True, exist_ok=True)

        pca_cfg = settings.get('pca_geometry', {})
        per_class = pca_cfg.get('per_class', 200)
        run_isomap = pca_cfg.get('isomap', False)

        tag = f"{ctx.spec.arch_name}_{ctx.spec.distribution}_layer{layer_idx}"

        try:
            report = run_pca_geometry(
                X=Z,
                y=labels,
                outdir=pca_geo_dir,
                tag=tag,
                per_class=per_class,
                run_isomap=run_isomap,
            )

            print(f"[PCA Geometry] Layer {layer_idx}: "
                  f"anisotropy_within={report.variance.anisotropy_within:.3f}, "
                  f"anisotropy_between={report.variance.anisotropy_between:.3f}")

            # WandB logging
            if ctx.wandb_run:
                try:
                    import wandb
                    ctx.wandb_run.log({
                        f"dimensionality/layer{layer_idx}/pca_geo/anisotropy_within": report.variance.anisotropy_within,
                        f"dimensionality/layer{layer_idx}/pca_geo/anisotropy_between": report.variance.anisotropy_between,
                        f"dimensionality/layer{layer_idx}/pca_geo/angle_pc1_deg": report.angles.angle_pc1_deg,
                        f"dimensionality/layer{layer_idx}/pca_geo/rho_pc1": report.angles.rho_pc1,
                    })
                except Exception:
                    pass

        except Exception as exc:
            print(f"[PCA Geometry] Layer {layer_idx}: failed ({exc})")

    def _run_pca_report(self, Z: np.ndarray, labels: np.ndarray,
                       layer_dir: Path, layer_idx: int, ctx: Any, settings: Dict):
        """Run PCA decomposition report for one layer."""
        pca_rep_dir = layer_dir / "pca_report"
        pca_rep_dir.mkdir(parents=True, exist_ok=True)

        pca_cfg = settings.get('pca_report', {})
        random_state = pca_cfg.get('random_state', 42)

        regime_name = ctx.spec.distribution
        layer_tag = f"layer{layer_idx}"

        try:
            report_dict = generate_pca_decomposition_report(
                X=Z,
                y=labels,
                regime_name=regime_name,
                layer_tag=layer_tag,
                out_dir=pca_rep_dir,
                random_state=random_state,
            )

            if report_dict:
                samples_info = report_dict.get('samples', {})
                rho_pc1 = samples_info.get('rho', [0])[0]
                evr_pc1 = samples_info.get('evr', [0])[0]
                print(f"[PCA Report] Layer {layer_idx}: rho_pc1={rho_pc1:.3f}, evr_pc1={evr_pc1:.3f}")

                # WandB logging
                if ctx.wandb_run:
                    try:
                        import wandb
                        ctx.wandb_run.log({
                            f"dimensionality/layer{layer_idx}/pca_report/rho_pc1": rho_pc1,
                            f"dimensionality/layer{layer_idx}/pca_report/evr_pc1": evr_pc1,
                            f"dimensionality/layer{layer_idx}/pca_report/angle_deg": report_dict.get('angle_deg', 0),
                        })
                    except Exception:
                        pass

        except Exception as exc:
            print(f"[PCA Report] Layer {layer_idx}: failed ({exc})")

    def _run_tsne(self, Z: np.ndarray, labels: np.ndarray,
                 layer_dir: Path, layer_idx: int, ctx: Any, settings: Dict):
        """Run TSNE dimensionality reduction for one layer."""
        tsne_dir = layer_dir / "tsne"
        tsne_dir.mkdir(parents=True, exist_ok=True)

        tsne_cfg = settings.get('tsne', {})
        perplexity = tsne_cfg.get('perplexity', 30)
        n_iter = tsne_cfg.get('n_iter', 1000)
        random_state = tsne_cfg.get('random_state', 42)

        try:
            from sklearn.manifold import TSNE

            # Subsample if too large
            max_samples = tsne_cfg.get('max_samples', 5000)
            if Z.shape[0] > max_samples:
                rng = np.random.default_rng(random_state)
                idx = rng.choice(Z.shape[0], size=max_samples, replace=False)
                Z_sub = Z[idx]
                labels_sub = labels[idx]
            else:
                Z_sub = Z
                labels_sub = labels

            tsne = TSNE(n_components=2, perplexity=perplexity, n_iter=n_iter, random_state=random_state)
            Z_tsne = tsne.fit_transform(Z_sub)

            # Save embeddings
            np.save(tsne_dir / "tsne_embeddings.npy", Z_tsne)

            # Plot
            fig, ax = plt.subplots(figsize=(8, 6))
            scatter = ax.scatter(Z_tsne[:, 0], Z_tsne[:, 1], c=labels_sub, cmap='viridis', s=10, alpha=0.6)
            plt.colorbar(scatter, ax=ax, label='Numerosity')
            ax.set_title(f"TSNE — {ctx.spec.arch_name} ({ctx.spec.distribution}) [Layer {layer_idx}]")
            ax.set_xlabel("TSNE 1")
            ax.set_ylabel("TSNE 2")
            fig.tight_layout()
            fig.savefig(tsne_dir / "tsne_plot.png", dpi=300)
            plt.close(fig)

            print(f"[TSNE] Layer {layer_idx}: completed")

            # WandB logging
            if ctx.wandb_run:
                try:
                    import wandb
                    ctx.wandb_run.log({
                        f"dimensionality/layer{layer_idx}/tsne": wandb.Image(str(tsne_dir / "tsne_plot.png")),
                    })
                except Exception:
                    pass

        except ImportError:
            print(f"[TSNE] Layer {layer_idx}: skipped (sklearn not available)")
        except Exception as exc:
            print(f"[TSNE] Layer {layer_idx}: failed ({exc})")

    def _run_umap(self, Z: np.ndarray, labels: np.ndarray,
                 layer_dir: Path, layer_idx: int, ctx: Any, settings: Dict):
        """Run UMAP dimensionality reduction for one layer."""
        umap_dir = layer_dir / "umap"
        umap_dir.mkdir(parents=True, exist_ok=True)

        umap_cfg = settings.get('umap', {})
        n_neighbors = umap_cfg.get('n_neighbors', 15)
        min_dist = umap_cfg.get('min_dist', 0.1)
        random_state = umap_cfg.get('random_state', 42)

        try:
            import umap as umap_pkg

            # Subsample if too large
            max_samples = umap_cfg.get('max_samples', 5000)
            if Z.shape[0] > max_samples:
                rng = np.random.default_rng(random_state)
                idx = rng.choice(Z.shape[0], size=max_samples, replace=False)
                Z_sub = Z[idx]
                labels_sub = labels[idx]
            else:
                Z_sub = Z
                labels_sub = labels

            reducer = umap_pkg.UMAP(
                n_components=2,
                n_neighbors=n_neighbors,
                min_dist=min_dist,
                random_state=random_state
            )
            Z_umap = reducer.fit_transform(Z_sub)

            # Save embeddings
            np.save(umap_dir / "umap_embeddings.npy", Z_umap)

            # Plot
            fig, ax = plt.subplots(figsize=(8, 6))
            scatter = ax.scatter(Z_umap[:, 0], Z_umap[:, 1], c=labels_sub, cmap='viridis', s=10, alpha=0.6)
            plt.colorbar(scatter, ax=ax, label='Numerosity')
            ax.set_title(f"UMAP — {ctx.spec.arch_name} ({ctx.spec.distribution}) [Layer {layer_idx}]")
            ax.set_xlabel("UMAP 1")
            ax.set_ylabel("UMAP 2")
            fig.tight_layout()
            fig.savefig(umap_dir / "umap_plot.png", dpi=300)
            plt.close(fig)

            print(f"[UMAP] Layer {layer_idx}: completed")

            # WandB logging
            if ctx.wandb_run:
                try:
                    import wandb
                    ctx.wandb_run.log({
                        f"dimensionality/layer{layer_idx}/umap": wandb.Image(str(umap_dir / "umap_plot.png")),
                    })
                except Exception:
                    pass

        except ImportError:
            print(f"[UMAP] Layer {layer_idx}: skipped (umap-learn not installed)")
        except Exception as exc:
            print(f"[UMAP] Layer {layer_idx}: failed ({exc})")
