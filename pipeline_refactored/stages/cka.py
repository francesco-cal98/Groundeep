"""CKA comparison stage."""

from pathlib import Path
from typing import Dict, Any, Tuple, List
import sys
import time
import numpy as np
import pandas as pd
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.analyses.cka import (
    compute_layerwise_cka,
    plot_cka_heatmap,
    permutation_test_cka,
)

# Optional: cka_report module (if available)
try:
    from cka_report import generate_report
    HAS_CKA_REPORT = True
except ImportError:
    HAS_CKA_REPORT = False
    generate_report = None


class CKAStage:
    """Stage for CKA similarity analysis between uniform and zipfian models."""

    name = "cka"

    def is_enabled(self, settings: Dict[str, Any]) -> bool:
        return settings.get('enabled', False)

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

    def run(self, ctx: Any, settings: Dict[str, Any], output_dir: Path) -> None:
        """Compute CKA between uniform and zipfian models."""
        model_uniform = ctx.get_model("uniform")
        model_zipfian = ctx.get_model("zipfian")
        uniform_layers = getattr(model_uniform, "layers", [])
        zipf_layers = getattr(model_zipfian, "layers", [])
        max_layers = min(len(uniform_layers), len(zipf_layers))

        if max_layers <= 0:
            print(f"[CKA] {ctx.spec.arch_name}/{ctx.spec.distribution}: no layers available")
            return

        layers_for_cka = list(range(1, max_layers + 1))
        base_inputs = ctx.base_batch.view(ctx.base_batch.shape[0], -1).to(torch.float32)

        models_for_cka = {
            "uniform": model_uniform,
            "zipfian": model_zipfian,
        }
        repr_cache: Dict[Tuple[int, str], np.ndarray] = {}

        def _repr(layer_idx: int, model_tag: str) -> np.ndarray:
            key = (layer_idx, model_tag)
            if key in repr_cache:
                return repr_cache[key]
            model = models_for_cka[model_tag]
            layers_model = getattr(model, "layers", [])
            upto = min(layer_idx, len(layers_model))
            device = self._get_model_device(model)
            with torch.no_grad():
                cur = base_inputs.to(device)
                for rbm in layers_model[:upto]:
                    cur = rbm.forward(cur)
                arr = cur.detach().cpu().float().numpy()
            repr_cache[key] = arr
            return arr

        cka_dir = output_dir / "cka"
        cka_dir.mkdir(parents=True, exist_ok=True)

        rng = np.random.default_rng(ctx.seed)
        subset_idx = None
        n_max = settings.get("n_max")
        if n_max is not None:
            n_max = int(n_max)
            if base_inputs.shape[0] > n_max:
                subset_idx = np.sort(rng.choice(base_inputs.shape[0], size=n_max, replace=False))

        kernels_to_run = settings.get("kernels") or ["linear", "rbf"]
        do_linear = "linear" in kernels_to_run
        do_rbf = "rbf" in kernels_to_run

        # Linear CKA
        if do_linear:
            t0 = time.perf_counter()
            M_lin, namesA, namesB = compute_layerwise_cka(
                _repr, layers_for_cka, layers_for_cka,
                tag_A="uniform", tag_B="zipfian",
                indices=subset_idx, kind="linear", rng=rng,
            )
            t_lin = time.perf_counter() - t0
            lin_df = pd.DataFrame(M_lin, index=namesA, columns=namesB)
            lin_csv = cka_dir / f"cka_linear_{ctx.spec.arch_name}.csv"
            lin_df.to_csv(lin_csv)
            lin_png = cka_dir / f"cka_linear_{ctx.spec.arch_name}.png"
            plot_cka_heatmap(M_lin, namesA, namesB,
                           f"Linear CKA — {ctx.spec.arch_name} (uniform vs zipf)", lin_png)
        else:
            M_lin = lin_png = None
            t_lin = 0.0

        # RBF CKA
        if do_rbf:
            t0 = time.perf_counter()
            M_rbf, _, _ = compute_layerwise_cka(
                _repr, layers_for_cka, layers_for_cka,
                tag_A="uniform", tag_B="zipfian",
                indices=subset_idx, kind="rbf", rng=rng,
            )
            t_rbf = time.perf_counter() - t0
            rbf_df = pd.DataFrame(M_rbf, index=namesA, columns=namesB)
            rbf_csv = cka_dir / f"cka_rbf_{ctx.spec.arch_name}.csv"
            rbf_df.to_csv(rbf_csv)
            rbf_png = cka_dir / f"cka_rbf_{ctx.spec.arch_name}.png"
            plot_cka_heatmap(M_rbf, namesA, namesB,
                           f"RBF CKA — {ctx.spec.arch_name} (uniform vs zipf)", rbf_png)
        else:
            M_rbf = rbf_png = None
            t_rbf = 0.0

        msg = f"[CKA] {ctx.spec.arch_name}/{ctx.spec.distribution}:"
        if M_lin is not None:
            msg += f" diag linear={np.diag(M_lin).mean():.3f} (time {t_lin:.1f}s)"
        if M_rbf is not None:
            msg += f", diag rbf={np.diag(M_rbf).mean():.3f} (time {t_rbf:.1f}s)"
        print(msg)

        # WandB logging
        if ctx.wandb_run is not None:
            try:
                import wandb
                payload = {}
                if lin_png: payload["cka/linear"] = wandb.Image(str(lin_png))
                if rbf_png: payload["cka/rbf"] = wandb.Image(str(rbf_png))
                if payload: ctx.wandb_run.log(payload)
            except Exception:
                pass

        # Permutation test
        perm_cfg = settings.get("permutation") or {}
        if perm_cfg.get("enabled", False):
            n_perm = int(perm_cfg.get("n_perm", 200))
            perm_rng = np.random.default_rng(ctx.seed)
            records = []
            for layer in layers_for_cka:
                Xa = _repr(layer, "uniform")
                Yb = _repr(layer, "zipfian")
                if subset_idx is not None:
                    Xa = Xa[subset_idx]
                    Yb = Yb[subset_idx]
                p_lin = permutation_test_cka(Xa, Yb, n_perm=n_perm, kind="linear", rng=perm_rng)
                p_rbf = permutation_test_cka(Xa, Yb, n_perm=n_perm, kind="rbf", rng=perm_rng) if do_rbf else np.nan
                records.append({"layer": layer, "p_linear": p_lin, "p_rbf": p_rbf})
            pd.DataFrame(records).to_csv(cka_dir / f"cka_permutation_{ctx.spec.arch_name}.csv", index=False)

        # Rich report
        report_cfg = settings.get("report", {})
        if report_cfg.get("enabled", True):
            try:
                report_dir = cka_dir / "report"
                acts_uniform_report: List[np.ndarray] = []
                acts_zipf_report: List[np.ndarray] = []
                for layer in layers_for_cka:
                    Xa = _repr(layer, "uniform")
                    Yb = _repr(layer, "zipfian")
                    if subset_idx is not None:
                        Xa = Xa[subset_idx]
                        Yb = Yb[subset_idx]
                    acts_uniform_report.append(Xa)
                    acts_zipf_report.append(Yb)

                if HAS_CKA_REPORT:
                    generate_report(
                        acts_uniform_report, acts_zipf_report,
                        outdir=report_dir,
                        layer_names_uniform=namesA,
                        layer_names_zipf=namesB,
                        kernels=report_cfg.get("kernels", ["linear", "rbf"]),
                        gamma=report_cfg.get("gamma"),
                        bootstrap=int(report_cfg.get("bootstrap", 200)),
                        ridge_tau=float(report_cfg.get("ridge_tau", 0.1)),
                        null_permutations=int(report_cfg.get("null_permutations", 1000)),
                        seed=int(report_cfg.get("seed", ctx.seed)),
                    )
                else:
                    print("⚠️  cka_report module not available, skipping detailed report")
            except Exception as exc:
                print(f"[CKA report] Failed to generate report ({exc})")
