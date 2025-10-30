# src/main_scripts/analyze.py
# Unified offline analysis runner (Hydra) che RIUSA le tue funzioni.

import os, math, warnings, sys, json
from pathlib import Path
from typing import Dict, Any, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
import seaborn as sns

warnings.filterwarnings("ignore", category=UserWarning)
plt.switch_backend("Agg")

# ==== Hydra / OmegaConf ====
import hydra
from omegaconf import DictConfig, OmegaConf
from hydra.utils import to_absolute_path

# ==== Stable sys.path so 'src' is importable even if Hydra changes CWD ====
THIS_FILE = Path(__file__).resolve()
PROJECT_ROOT = THIS_FILE.parents[2]  # repo root containing 'src'
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from cka_report import generate_report

# ==== TUE FUNZIONI/CLASSI ====
# - Embedding_analysis: carica dataloader + modelli e produce Z + feature allineate
from src.analyses.embedding_analysis import Embedding_analysis

# - Tue funzioni di plotting embedding
from src.utils.wandb_utils import (
    plot_2d_embedding_and_correlations,
    plot_3d_embedding_and_correlations,
    log_reconstructions_to_wandb,  # opzionale
)

# - Tuoi probe (con i piccoli bugfix menzionati)
from src.utils.probe_utils import (
    log_linear_probe,
    # log_joint_linear_probe  # se vorrai usare probe multimodale
)

# ==== Riduzioni (sklearn / umap) ====
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE, MDS
try:
    import umap  # pip install umap-learn
    _HAS_UMAP = True
except Exception:
    _HAS_UMAP = False

# ==== Stat/RSA utils (dal tuo runner “robust” che avevi incollato) ====
from scipy.spatial.distance import pdist
from scipy.stats import kendalltau, spearmanr, linregress
from statsmodels.stats.multitest import multipletests
from sklearn.metrics import pairwise_distances

# Monotonicity visualizations
from src.analyses.monotonicity_viz import (
    compute_class_centroids,
    pairwise_centroid_distances,
    plot_distance_vs_deltaN,
    plot_violin_by_deltaN,
    plot_centroid_heatmap,
    plot_ordinal_trajectory_1d,
    plot_ordinal_trajectory_2d,
    plot_outlier_pairs,
    save_deltaN_stats_csv,
    save_outlier_pairs_csv,
)

from src.analyses.mse_viz import (
    compute_sample_mse,
    prepare_mse_dataframe,
    plot_mse_heatmap,
    plot_mse_vs_numerosity,
    save_regression_results,
)

from src.analyses.powerfit_pairs import (
    build_pairwise_xy,
    fit_power_loglog_pairs,
    plot_pairs_fit,
    plot_pairs_fit_loglog,
    save_pairs_fit,
)

from src.analyses.cka import (
    compute_layerwise_cka,
    plot_cka_heatmap,
    permutation_test_cka,
)
from src.analyses.behavioral_analysis import (
    load_behavioral_inputs,
    run_behavioral_analysis,
)
from src.analyses.task_comparison import run_task_comparison
from src.analyses.task_fixed_reference import (
    load_fixed_reference_inputs,
    run_task_fixed_reference,
)
from src.analyses.task_numerosity_estimation import (
    load_estimation_dataset,
    run_task_numerosity_estimation,
)
from src.datasets.uniform_dataset import compute_label_histogram, plot_label_histogram

# =========================
# Helper per device/model
# =========================
def _get_model_device(model_obj) -> torch.device:
    try:
        import torch.nn as nn
        if isinstance(model_obj, nn.Module):
            try:
                return next(model_obj.parameters()).device
            except StopIteration:
                pass
    except Exception:
        pass
    layers = getattr(model_obj, "layers", None)
    if layers:
        for rbm in layers:
            for name in ("W","weight","weights","U","V","b","hbias","vbias"):
                t = getattr(rbm, name, None)
                if hasattr(t, "device"):
                    return t.device
            for v in rbm.__dict__.values():
                if hasattr(v, "device"):
                    return v.device
    return torch.device("cpu")


def _extract_labels_from_subset(subset) -> np.ndarray | None:
    if subset is None:
        return None
    if hasattr(subset, "indices"):
        base = getattr(subset, "dataset", None)
        if base is not None and hasattr(base, "labels"):
            return np.asarray(base.labels)[subset.indices]
    if hasattr(subset, "labels"):
        return np.asarray(getattr(subset, "labels"))
    return None


def _report_label_coverage(name: str, subset) -> None:
    labels = _extract_labels_from_subset(subset)
    if labels is None or labels.size == 0:
        print(f"[Labels] {name}: unavailable")
        return
    hist_df = compute_label_histogram(labels)
    summary = {int(row.label): int(row.count) for row in hist_df.itertuples()}
    print(f"[Labels] {name}: {summary}")


def _aggregate_delta_curve(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame(columns=["deltaN", "distance"])
    data = df.dropna(subset=["deltaN", "distance"]).copy()
    if data.empty:
        return pd.DataFrame(columns=["deltaN", "distance"])
    data["deltaN"] = data["deltaN"].astype(float)
    data["distance"] = data["distance"].astype(float)
    grouped = data.groupby("deltaN", as_index=False)["distance"].median()
    return grouped.sort_values("deltaN").reset_index(drop=True)


def _safe_slope(x: np.ndarray, y: np.ndarray) -> float:
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if x.size < 2 or np.allclose(x, x.mean()):
        return float("nan")
    return float(linregress(x, y).slope)


def _plot_deltaN_slope_comparison(
    df_uniform: pd.DataFrame,
    df_zipf: pd.DataFrame,
    out_path: Path,
    *,
    window: tuple[int, int] = (1, 5),
    n_boot: int = 1000,
    seed: int = 0,
) -> Dict[str, float]:
    df_u = df_uniform.dropna(subset=["deltaN", "distance"])
    df_z = df_zipf.dropna(subset=["deltaN", "distance"])
    if df_u.empty or df_z.empty:
        return {}

    agg_u = _aggregate_delta_curve(df_u)
    agg_z = _aggregate_delta_curve(df_z)

    mask_u = (df_u["deltaN"] >= window[0]) & (df_u["deltaN"] <= window[1])
    mask_z = (df_z["deltaN"] >= window[0]) & (df_z["deltaN"] <= window[1])
    x_u_win = df_u.loc[mask_u, "deltaN"].to_numpy(dtype=float)
    y_u_win = df_u.loc[mask_u, "distance"].to_numpy(dtype=float)
    x_z_win = df_z.loc[mask_z, "deltaN"].to_numpy(dtype=float)
    y_z_win = df_z.loc[mask_z, "distance"].to_numpy(dtype=float)

    slope_u = _safe_slope(x_u_win, y_u_win)
    slope_z = _safe_slope(x_z_win, y_z_win)

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.scatter(df_u["deltaN"], df_u["distance"], color="royalblue", alpha=0.12, s=14)
    ax.scatter(df_z["deltaN"], df_z["distance"], color="darkorange", alpha=0.12, s=14)
    ax.plot(
        agg_u["deltaN"],
        agg_u["distance"],
        marker="o",
        color="royalblue",
        label=f"Uniform (β={slope_u:.3f})" if not math.isnan(slope_u) else "Uniform",
    )
    ax.plot(
        agg_z["deltaN"],
        agg_z["distance"],
        marker="s",
        color="darkorange",
        label=f"Zipfian (β={slope_z:.3f})" if not math.isnan(slope_z) else "Zipfian",
    )
    ax.axvspan(window[0], window[1], color="gray", alpha=0.1, label=f"Low ΔN [{window[0]}, {window[1]}]")
    ax.set_xlabel("ΔN")
    ax.set_ylabel("Centroid distance")
    ax.set_title("ΔN vs Representational Distance\nSharper low-ΔN slope for Zipfian")
    ax.legend()
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=300)
    plt.close(fig)

    stats: Dict[str, float] = {
        "window_lo": float(window[0]),
        "window_hi": float(window[1]),
        "slope_uniform": float(slope_u) if not math.isnan(slope_u) else math.nan,
        "slope_zipfian": float(slope_z) if not math.isnan(slope_z) else math.nan,
        "delta_beta": float(slope_z - slope_u) if not (math.isnan(slope_u) or math.isnan(slope_z)) else math.nan,
        "n_uniform_window": int(x_u_win.size),
        "n_zipf_window": int(x_z_win.size),
    }

    if x_u_win.size >= 2 and x_z_win.size >= 2 and n_boot > 0:
        rng = np.random.default_rng(seed)
        diffs = []
        for _ in range(n_boot):
            idx_u = rng.integers(0, x_u_win.size, size=x_u_win.size)
            idx_z = rng.integers(0, x_z_win.size, size=x_z_win.size)
            s_u = _safe_slope(x_u_win[idx_u], y_u_win[idx_u])
            s_z = _safe_slope(x_z_win[idx_z], y_z_win[idx_z])
            if math.isnan(s_u) or math.isnan(s_z):
                continue
            diffs.append(s_z - s_u)
        if diffs:
            diffs_arr = np.asarray(diffs, dtype=float)
            stats["delta_beta_ci_lo"] = float(np.percentile(diffs_arr, 2.5))
            stats["delta_beta_ci_hi"] = float(np.percentile(diffs_arr, 97.5))
            stats["delta_beta_mean"] = float(diffs_arr.mean())
            stats["n_boot_valid"] = int(diffs_arr.size)
    return stats


def _plot_powerlaw_fit_comparison(
    df_uniform: pd.DataFrame,
    df_zipf: pd.DataFrame,
    out_path: Path,
) -> Dict[str, float]:
    if df_uniform.empty or df_zipf.empty:
        return {}
    x_u = df_uniform["deltaN"].to_numpy(dtype=float)
    y_u = df_uniform["distance"].to_numpy(dtype=float)
    x_z = df_zipf["deltaN"].to_numpy(dtype=float)
    y_z = df_zipf["distance"].to_numpy(dtype=float)
    if x_u.size == 0 or x_z.size == 0:
        return {}

    fit_u = fit_power_loglog_pairs(x_u, y_u)
    fit_z = fit_power_loglog_pairs(x_z, y_z)

    min_x = max(1e-6, min(x_u.min(), x_z.min()))
    max_x = max(x_u.max(), x_z.max())
    xs_line = np.logspace(np.log10(min_x), np.log10(max_x), 200)
    ys_u = fit_u["a"] * np.power(xs_line, fit_u["b"])
    ys_z = fit_z["a"] * np.power(xs_line, fit_z["b"])

    fig, ax = plt.subplots(figsize=(6, 5))
    ax.scatter(x_u, y_u, color="royalblue", alpha=0.18, s=18, label="Uniform pairs")
    ax.scatter(x_z, y_z, color="darkorange", alpha=0.18, s=18, label="Zipfian pairs")
    ax.plot(xs_line, ys_u, color="royalblue", linewidth=2.0, label=f"Uniform fit b={fit_u['b']:.3f}, R²={fit_u['r2']:.3f}")
    ax.plot(xs_line, ys_z, color="darkorange", linewidth=2.0, label=f"Zipfian fit b={fit_z['b']:.3f}, R²={fit_z['r2']:.3f}")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("ΔN (log scale)")
    ax.set_ylabel("Centroid distance (log scale)")
    ax.set_title("Power-law Scaling — Weber-like slopes, smoother Zipfian")
    ax.legend()
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=300)
    plt.close(fig)

    resid_u = np.asarray(fit_u.get("residuals", []), dtype=float)
    resid_z = np.asarray(fit_z.get("residuals", []), dtype=float)
    var_u = float(np.var(resid_u, ddof=1)) if resid_u.size > 1 else float(np.var(resid_u, ddof=0))
    var_z = float(np.var(resid_z, ddof=1)) if resid_z.size > 1 else float(np.var(resid_z, ddof=0))

    return {
        "a_uniform": float(fit_u["a"]),
        "b_uniform": float(fit_u["b"]),
        "r2_uniform": float(fit_u["r2"]),
        "residual_variance_uniform": var_u,
        "a_zipfian": float(fit_z["a"]),
        "b_zipfian": float(fit_z["b"]),
        "r2_zipfian": float(fit_z["r2"]),
        "residual_variance_zipfian": var_z,
    }


def _plot_dispersion_per_deltaN(
    df_uniform: pd.DataFrame,
    df_zipf: pd.DataFrame,
    out_path: Path,
) -> tuple[Dict[str, float], pd.DataFrame]:
    if df_uniform.empty or df_zipf.empty:
        return {}, pd.DataFrame()
    std_u = df_uniform.groupby("deltaN")["distance"].std(ddof=0)
    std_z = df_zipf.groupby("deltaN")["distance"].std(ddof=0)
    df_agg = pd.concat([std_u.rename("Uniform"), std_z.rename("Zipfian")], axis=1).reset_index()
    df_long = df_agg.melt(id_vars="deltaN", value_name="std", var_name="distribution").dropna(subset=["std"])

    fig, ax = plt.subplots(figsize=(6, 4))
    sns.lineplot(data=df_long, x="deltaN", y="std", hue="distribution", marker="o", ax=ax)
    ax.set_xlabel("ΔN")
    ax.set_ylabel("Std. distance")
    ax.set_title("Dispersion per ΔN — lower variance = smoother encoding")
    ax.legend(title="")
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=300)
    plt.close(fig)

    stats = {
        "mean_std_uniform": float(np.nanmean(df_agg["Uniform"].to_numpy(dtype=float))),
        "mean_std_zipfian": float(np.nanmean(df_agg["Zipfian"].to_numpy(dtype=float))),
    }
    return stats, df_agg


def maybe_generate_monotonicity_comparisons(
    arch_name: str,
    out_root: Path | str,
    *,
    dist_name: str,
    monotonicity_cfg: Dict[str, Any] | None,
    seed: int,
    wandb_run=None,
) -> None:
    claims_cfg = (monotonicity_cfg or {}).get("comparisons", {})
    if claims_cfg.get("enabled", True) is False:
        return

    out_root = Path(out_root)
    uniform_csv = out_root / "uniform" / arch_name / "powerfit_pairs" / f"pairs_table_{arch_name}_uniform.csv"
    zipf_csv = out_root / "zipfian" / arch_name / "powerfit_pairs" / f"pairs_table_{arch_name}_zipfian.csv"
    if not (uniform_csv.exists() and zipf_csv.exists()):
        return

    df_uniform = pd.read_csv(uniform_csv)
    df_zipf = pd.read_csv(zipf_csv)
    if df_uniform.empty or df_zipf.empty:
        return

    comp_dir = out_root / "comparisons" / arch_name
    comp_dir.mkdir(parents=True, exist_ok=True)

    window = claims_cfg.get("window") or (1, 5)
    if not isinstance(window, (list, tuple)) or len(window) != 2:
        window = (1, 5)
    window = (int(window[0]), int(window[1]))
    n_boot = int(claims_cfg.get("bootstrap", 1000))

    slope_stats = _plot_deltaN_slope_comparison(
        df_uniform,
        df_zipf,
        comp_dir / "deltaN_slope_comparison.png",
        window=window,
        n_boot=n_boot,
        seed=seed,
    )
    if slope_stats:
        with open(comp_dir / "deltaN_slope_stats.json", "w", encoding="utf-8") as f:
            json.dump(slope_stats, f, indent=2)

    power_stats = _plot_powerlaw_fit_comparison(
        df_uniform,
        df_zipf,
        comp_dir / "powerlaw_fit_comparison.png",
    )
    if power_stats:
        with open(comp_dir / "powerlaw_stats.json", "w", encoding="utf-8") as f:
            json.dump(power_stats, f, indent=2)

    dispersion_stats, dispersion_df = _plot_dispersion_per_deltaN(
        df_uniform,
        df_zipf,
        comp_dir / "deltaN_dispersion.png",
    )
    if not dispersion_df.empty:
        dispersion_df.to_csv(comp_dir / "deltaN_dispersion.csv", index=False)
    if dispersion_stats:
        with open(comp_dir / "deltaN_dispersion_stats.json", "w", encoding="utf-8") as f:
            json.dump(dispersion_stats, f, indent=2)

    if wandb_run is not None and dist_name.lower() == "zipfian":
        try:
            import wandb

            log_payload = {}
            slope_img = comp_dir / "deltaN_slope_comparison.png"
            power_img = comp_dir / "powerlaw_fit_comparison.png"
            disp_img = comp_dir / "deltaN_dispersion.png"
            if slope_img.exists():
                log_payload["monotonicity_compare/deltaN_slope"] = wandb.Image(str(slope_img))
            if power_img.exists():
                log_payload["monotonicity_compare/powerlaw"] = wandb.Image(str(power_img))
            if disp_img.exists():
                log_payload["monotonicity_compare/dispersion"] = wandb.Image(str(disp_img))
            if slope_stats:
                log_payload["monotonicity_compare/delta_beta"] = slope_stats.get("delta_beta", math.nan)
            if power_stats:
                log_payload["monotonicity_compare/residual_var_delta"] = power_stats.get("residual_variance_zipfian", math.nan) - power_stats.get("residual_variance_uniform", math.nan)
            if dispersion_stats:
                log_payload["monotonicity_compare/mean_dispersion_delta"] = dispersion_stats.get("mean_std_zipfian", math.nan) - dispersion_stats.get("mean_std_uniform", math.nan)
            if log_payload:
                wandb_run.log(log_payload)
        except Exception:
            pass

# =========================
# Adapter per riusare i TUOI probe
# =========================
class ProbeReadyModel:
    """
    Adapter per usare i tuoi probe offline:
      - espone: device, val_loader, features, wandb_run, arch_dir, text_flag, represent(...)
    """
    def __init__(self, raw_model, val_loader, features_dict, out_dir: Path, wandb_run=None):
        self.raw = raw_model
        self.val_loader = val_loader
        self.features = features_dict
        self.wandb_run = wandb_run
        self.arch_dir = str(out_dir)
        self.text_flag = False
        self.device = _get_model_device(raw_model)

    @torch.no_grad()
    def represent(self, x, upto_layer: int | None = None):
        if hasattr(self.raw, "represent"):
            return self.raw.represent(x, upto_layer=upto_layer) if upto_layer is not None else self.raw.represent(x)
        # Stack RBM
        xt = x
        layers = getattr(self.raw, "layers", [])
        upto = len(layers) if upto_layer is None else min(upto_layer, len(layers))
        for rbm in layers[:upto]:
            xt = rbm.forward(xt)
        return xt

# =========================
# Riduzioni (compute)
# =========================
def compute_reductions(Z: np.ndarray, seed: int = 42,
                       enable_tsne: bool = True,
                       enable_mds: bool = True,
                       enable_umap: bool = True) -> Dict[str, np.ndarray]:
    outs = {}
    if Z.shape[0] >= 2 and Z.shape[1] >= 2:
        outs["PCA_2"]  = PCA(n_components=2, random_state=seed).fit_transform(Z)
        outs["PCA_3"]  = PCA(n_components=min(3, Z.shape[1]), random_state=seed).fit_transform(Z)
        if enable_tsne:
            try:
                outs["TSNE_2"] = TSNE(n_components=2, random_state=seed, init="pca", learning_rate="auto").fit_transform(Z)
                if Z.shape[1] >= 3:
                    outs["TSNE_3"] = TSNE(n_components=3, random_state=seed, init="pca", learning_rate="auto").fit_transform(Z)
            except Exception:
                pass
        if enable_mds:
            try:
                outs["MDS_2"] = MDS(n_components=2, random_state=seed, n_init=4, max_iter=300).fit_transform(Z)
            except Exception:
                pass
        if enable_umap and _HAS_UMAP:
            try:
                outs["UMAP_2"] = umap.UMAP(n_components=2, random_state=seed).fit_transform(Z)
                if Z.shape[1] >= 3:
                    outs["UMAP_3"] = umap.UMAP(n_components=3, random_state=seed).fit_transform(Z)
            except Exception:
                pass
    return outs

# =========================
# RSA & friends (copiate dal tuo runner)
# =========================
def _compute_brain_rdm(embeddings: np.ndarray, metric: str = "cosine") -> np.ndarray:
    if embeddings.shape[0] < 2:
        return np.array([])
    X = embeddings.copy()
    if metric == "cosine":
        X = X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-12)
        X = np.nan_to_num(X)
        return pdist(X, metric="cosine")
    if metric == "euclidean":
        X = (X - X.mean(0, keepdims=True)) / (X.std(0, keepdims=True) + 1e-8)
        return pdist(X, metric="euclidean")
    raise ValueError("metric not supported")

def _compute_model_rdm(values: np.ndarray, metric: str = "euclidean") -> np.ndarray:
    if values is None or len(values) < 2:
        return np.array([])
    x = np.asarray(values, dtype=np.float64).reshape(-1,1)
    return pdist(x, metric=metric)

def run_rsa_with_fdr(Z: np.ndarray, feats: Dict[str, np.ndarray], arch: str, dist: str, out_dir: Path, alpha=0.01):
    brain = _compute_brain_rdm(Z, metric="cosine")
    if brain.size == 0:
        return
    lbl = feats.get("labels", None)
    fdict = {
        "numerosity_linear": lbl,
        "numerosity_log1p": np.log1p(lbl) if lbl is not None else None,
        "cumArea": feats.get("cumArea"),
        "CH": feats.get("CH"),
    }
    rows = []
    for name, vals in fdict.items():
        if vals is None: continue
        model_rdm = _compute_model_rdm(vals)
        if model_rdm.size == 0 or model_rdm.shape != brain.shape: continue
        v = np.isfinite(brain) & np.isfinite(model_rdm)
        if v.sum() < 3: continue
        tau, p_two = kendalltau(brain[v], model_rdm[v])
        if np.isnan(tau): continue
        p_one = p_two/2 if tau > 0 else 1 - (p_two/2)
        rows.append({"Architecture": arch, "Distribution": dist, "Feature": name, "Kendall Tau": float(tau), "P-value (1-sided)": float(p_one)})
    if not rows: return
    df = pd.DataFrame(rows)
    rej, p_corr, _, _ = multipletests(df["P-value (1-sided)"], alpha=alpha, method="fdr_bh")
    df["Significant_FDR"] = rej; df["P-value FDR"] = p_corr
    rsa_dir = out_dir / "rsa"
    rsa_dir.mkdir(parents=True, exist_ok=True)
    out_xlsx = rsa_dir / f"rsa_results_{arch}_{dist}.xlsx"
    df.to_excel(out_xlsx, index=False)
    plt.figure(figsize=(8,5))
    ax = sns.barplot(data=df, x="Feature", y="Kendall Tau", palette="deep")
    for i, r in enumerate(df.itertuples()):
        if r.Significant_FDR:
            ax.text(i, r._4 + 0.02, "*", ha="center", color="red")  # _4 = Kendall Tau
    plt.title(f"RSA ({arch}, {dist})"); plt.tight_layout()
    plt.savefig(rsa_dir / f"rsa_bar_{arch}_{dist}.png", dpi=300, bbox_inches="tight"); plt.close()

def pairwise_class_rdm(Z: np.ndarray, labels: np.ndarray, arch: str, dist: str, out_dir: Path, metric="cosine"):
    if Z.shape[0] < 2 or labels is None or labels.shape[0] < 2: return
    ul = np.unique(labels)
    X = Z.copy()
    if metric == "euclidean":
        X = (X - X.mean(0, keepdims=True)) / (X.std(0, keepdims=True) + 1e-8)
    elif metric == "cosine":
        X = X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-12)
    C = np.vstack([X[labels == u].mean(axis=0) for u in ul])
    if metric == "cosine":
        C = C / (np.linalg.norm(C, axis=1, keepdims=True) + 1e-12)
    D = pairwise_distances(C, metric=metric)
    plt.figure(figsize=(7,6))
    sns.heatmap(D, xticklabels=ul.astype(int), yticklabels=ul.astype(int), cmap="viridis", square=True)
    plt.xlabel("Numerosity"); plt.ylabel("Numerosity"); plt.title(f"Class RDM ({metric})")
    plt.tight_layout()
    rdm_dir = out_dir / "rdm"
    rdm_dir.mkdir(parents=True, exist_ok=True)
    plt.savefig(rdm_dir / f"class_rdm_{arch}_{dist}_{metric}.png", dpi=300); plt.close()

def monotonicity_deltaN(Z: np.ndarray, labels: np.ndarray) -> Dict[str, float]:
    ul = np.unique(labels)
    if len(ul) < 2: return {"spearman_r": np.nan, "p": np.nan}
    C = np.vstack([Z[labels == u].mean(axis=0) for u in ul])
    D = pairwise_distances(C, metric="euclidean")
    xs, ys = [], []
    for i in range(len(ul)):
        for j in range(i+1, len(ul)):
            xs.append(abs(int(ul[i]) - int(ul[j])))
            ys.append(D[i,j])
    rho, p = spearmanr(xs, ys)
    return {"spearman_r": float(rho), "p": float(p), "pairs": len(xs)}

def partial_rsa_numerosity(Z: np.ndarray, labels: np.ndarray, conf: Dict[str, np.ndarray]) -> Dict[str, float]:
    brain = _compute_brain_rdm(Z, metric="cosine")
    if brain.size == 0 or labels is None or len(labels) < 2:
        return {"partial_spearman": np.nan, "p": np.nan}
    num_rdm = pdist(labels.reshape(-1,1), metric="euclidean")
    conf_rdms = []
    for k in ("cumArea", "CH"):
        v = conf.get(k, None)
        if v is not None and len(v) > 1:
            conf_rdms.append(pdist(v.reshape(-1,1), metric="euclidean"))
    def _resid(y, Xs):
        if not Xs: return y
        X = np.vstack(Xs).T
        Xa = np.column_stack([np.ones(len(X)), X])
        beta, *_ = np.linalg.lstsq(Xa, y, rcond=None)
        return y - Xa @ beta
    rb = _resid(brain, conf_rdms); rf = _resid(num_rdm, conf_rdms)
    r, p = spearmanr(rb, rf)
    return {"partial_spearman": float(r), "p": float(p)}

# ====== Latent traversal (dal tuo runner; grid su PCA) ======
def _infer_hw_from_batch(batch_tensor) -> Tuple[int, int]:
    arr = batch_tensor.detach().cpu().numpy()
    if arr.ndim == 4:  # N,C,H,W
        return int(arr.shape[2]), int(arr.shape[3])
    if arr.ndim == 3:  # N,H,W
        return int(arr.shape[1]), int(arr.shape[2])
    L = int(np.sqrt(arr.shape[-1])); return L, L

def latent_grid_on_pca(model, Z: np.ndarray, base_batch: torch.Tensor, out: Path,
                       anchor_idx=0, pcs=(0,1), steps=7, delta=2.0, seed=42,
                       start_layer: int | None = None):
    if Z.ndim != 2 or Z.shape[1] < 2: return
    device = _get_model_device(model)
    pca = PCA(n_components=max(pcs)+1, random_state=seed).fit(Z)
    mu = Z.mean(0, keepdims=True); C = pca.components_
    S = pca.transform(Z); s0 = S[anchor_idx].copy(); sd = S.std(0) + 1e-8
    gx = np.linspace(s0[pcs[0]]-delta*sd[pcs[0]], s0[pcs[0]]+delta*sd[pcs[0]], steps)
    gy = np.linspace(s0[pcs[1]]-delta*sd[pcs[1]], s0[pcs[1]]+delta*sd[pcs[1]], steps)
    grid_scores = np.tile(s0, (steps*steps,1))
    k=0
    for yv in gy:
        for xv in gx:
            grid_scores[k, pcs[0]] = xv
            grid_scores[k, pcs[1]] = yv
            k+=1
    Zg = (grid_scores @ C) + mu
    with torch.no_grad():
        zt = torch.tensor(Zg, dtype=torch.float32, device=device)
        layers = getattr(model, "layers", [])
        # If Z belongs to an intermediate layer, push forward to top before decoding back
        if start_layer is not None and isinstance(start_layer, int) and start_layer < len(layers):
            cur = zt
            for rbm in layers[start_layer:]:
                cur = rbm.forward(cur)
            top = cur
        else:
            top = zt
        if hasattr(model, "decode"):
            dec = model.decode(top).detach().cpu().numpy()
        else:
            cur = top
            for rbm in reversed(layers):
                cur = rbm.backward(cur)
            dec = cur.detach().cpu().numpy()
    dec = np.squeeze(dec)
    if dec.ndim == 4 and dec.shape[1] in (1,3): dec = dec[:,0,:,:]
    if dec.ndim != 3: return
    H, W = _infer_hw_from_batch(base_batch)
    def _norm01(x):
        mn, mx = x.min(), x.max()
        return np.zeros_like(x) if mx<=mn else (x-mn)/(mx-mn)
    tiles=[]; idx=0
    for _ in gy:
        row=[]
        for _ in gx:
            row.append(_norm01(dec[idx,:H,:W])); idx+=1
        tiles.append(np.concatenate(row,axis=1))
    grid_img = np.concatenate(tiles,axis=0)
    out.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(steps*1.6, steps*1.6))
    plt.imshow(grid_img, cmap="gray"); plt.axis("off")
    plt.savefig(out, dpi=300, bbox_inches="tight"); plt.close()

# =========================
# RUNNER per un modello (usa TUTTE le TUE funzioni)
# =========================
def analyze_one_model(
    dataset_path: str,
    dataset_name: str,
    arch_name: str,
    dist_name: str,
    model_uniform: str,
    model_zipfian: str,
    out_root: str,
    val_size: float = 0.05,
    reductions_cfg: Dict[str, Any] | None = None,
    probing_cfg: Dict[str, Any] | None = None,
    rsa_cfg: Dict[str, Any] | None = None,
    rdm_cfg: Dict[str, Any] | None = None,
    monotonicity_cfg: Dict[str, Any] | None = None,
    partial_rsa_cfg: Dict[str, Any] | None = None,
    traversal_cfg: Dict[str, Any] | None = None,
    cka_cfg: Dict[str, Any] | None = None,
    mse_cfg: Dict[str, Any] | None = None,
    behavioral_cfg: Dict[str, Any] | None = None,
    use_wandb: bool = False,
    wandb_project: Optional[str] = None,
    wandb_run_name: Optional[str] = None,
    anchor_idx: int = 0,
    seed: int = 42,
):
    # Init (opzionale) W&B — i tuoi probe lo useranno se presente
    wandb_run = None
    if use_wandb:
        try:
            import wandb
            wandb_run = wandb.init(project=wandb_project, name=wandb_run_name, reinit=True)
        except Exception:
            wandb_run = None

    print(f"🔄 Analyzing {arch_name} — dist={dist_name}")

    # 1) Usa la TUA classe per allineare embeddings & features
    analyzer = Embedding_analysis(
        path2data=dataset_path,
        data_name=dataset_name,
        model_uniform=model_uniform,
        model_zipfian=model_zipfian,
        arch_name=arch_name,
        val_size=val_size,
    )

    def _loader_dataset(loader):
        return getattr(loader, "dataset", None) if loader is not None else None

    if dist_name.lower() == "zipfian":
        _report_label_coverage("zipf/train", _loader_dataset(getattr(analyzer, "train_dataloader_zipfian", None)))
        _report_label_coverage("zipf/val", _loader_dataset(getattr(analyzer, "val_dataloader_zipfian", None)))
        _report_label_coverage("zipf/test", _loader_dataset(getattr(analyzer, "test_dataloader_zipfian", None)))
    else:
        _report_label_coverage("uniform/train", _loader_dataset(getattr(analyzer, "train_dataloader_uniform", None)))
        _report_label_coverage("uniform/val", _loader_dataset(getattr(analyzer, "val_dataloader_uniform", None)))
        _report_label_coverage("uniform/test", _loader_dataset(getattr(analyzer, "test_dataloader_uniform", None)))
    out = analyzer._get_encodings()

    # Sorgenti coerenti (usa dist_name come vista principale)
    Z = np.array(out.get(f"Z_{dist_name}", []), dtype=np.float64)
    labels = np.array(out.get(f"labels_{dist_name}", []))
    cumArea = np.array(out.get(f"cumArea_{dist_name}", []))
    CH = np.array(out.get(f"CH_{dist_name}", []))
    feats = {"labels": labels, "cumArea": cumArea, "CH": CH}

    inputs_cpu = analyzer.inputs_uniform
    orig_flat = inputs_cpu.reshape(inputs_cpu.shape[0], -1).numpy()

    out_dir = Path(out_root) / dist_name / arch_name
    out_dir.mkdir(parents=True, exist_ok=True)
    # Base batch for decoders/traversals
    base_batch = analyzer.inputs_uniform
    tr_cfg = traversal_cfg or {}

    # Power-fit on centroid distances (top-level)
    pf_dir = out_dir / "powerfit_pairs"
    pf_dir.mkdir(parents=True, exist_ok=True)
    x_pairs, y_pairs, pairs_df = build_pairwise_xy(Z, labels, metric="euclidean")
    pairs_df.to_csv(pf_dir / f"pairs_table_{arch_name}_{dist_name}.csv", index=False)
    if x_pairs.size > 0:
        fit = fit_power_loglog_pairs(x_pairs, y_pairs)
        save_pairs_fit(fit, pf_dir / f"params_{arch_name}_{dist_name}.csv")
        plot_pairs_fit(
            x_pairs,
            y_pairs,
            fit,
            pf_dir / f"fit_linear_{arch_name}_{dist_name}.png",
            f"{arch_name} ({dist_name})",
        )
        plot_pairs_fit_loglog(
            x_pairs,
            y_pairs,
            fit,
            pf_dir / f"fit_loglog_{arch_name}_{dist_name}.png",
            f"{arch_name} ({dist_name})",
        )
        print(
            f"[PowerFit] {arch_name}/{dist_name}: b={fit['b']:.3f}, R²={fit['r2']:.3f}"
        )
        if wandb_run is not None:
            try:
                import wandb

                wandb_run.log(
                    {
                        "powerfit_pairs/fit_linear": wandb.Image(
                            str(pf_dir / f"fit_linear_{arch_name}_{dist_name}.png")
                        ),
                        "powerfit_pairs/fit_loglog": wandb.Image(
                            str(pf_dir / f"fit_loglog_{arch_name}_{dist_name}.png")
                        ),
                        "powerfit_pairs/b": fit["b"],
                        "powerfit_pairs/r2": fit["r2"],
                    }
                )
            except Exception:
                pass
    else:
        print(f"[PowerFit] {arch_name}/{dist_name}: no valid centroid pairs for fitting.")

    # CKA between uniform and zipfian representations (layer-wise)
    if cka_cfg.get("enabled", True):
        uniform_layers = getattr(analyzer.model_uniform, "layers", [])
        zipf_layers = getattr(analyzer.model_zipfian, "layers", [])
        max_layers = min(len(uniform_layers), len(zipf_layers))
        if max_layers > 0:
            layers_for_cka = list(range(1, max_layers + 1))
            base_inputs_flat_tensor = analyzer.inputs_uniform.view(analyzer.inputs_uniform.shape[0], -1).to(torch.float32)
            models_for_cka = {
                "uniform": analyzer.model_uniform,
                "zipfian": analyzer.model_zipfian,
            }
            repr_cache: dict[tuple[int, str], np.ndarray] = {}

            def _repr(layer_idx: int, model_tag: str) -> np.ndarray:
                key = (layer_idx, model_tag)
                if key in repr_cache:
                    return repr_cache[key]
                model = models_for_cka[model_tag]
                layers_model = getattr(model, "layers", [])
                upto = min(layer_idx, len(layers_model))
                device = _get_model_device(model)
                with torch.no_grad():
                    cur = base_inputs_flat_tensor.to(device)
                    for rbm in layers_model[:upto]:
                        cur = rbm.forward(cur)
                    arr = cur.detach().cpu().numpy().astype(np.float64)
                repr_cache[key] = arr
                return arr

            cka_dir = out_dir / "cka"
            cka_dir.mkdir(parents=True, exist_ok=True)

            rng = np.random.default_rng(seed)
            subset_idx = None
            n_max = cka_cfg.get("n_max", None)
            if n_max is not None:
                n_max = int(n_max)
                if base_inputs_flat_tensor.shape[0] > n_max:
                    subset_idx = np.sort(rng.choice(base_inputs_flat_tensor.shape[0], size=n_max, replace=False))

            M_lin, namesA, namesB = compute_layerwise_cka(
                _repr,
                layers_for_cka,
                layers_for_cka,
                tag_A="uniform",
                tag_B="zipfian",
                indices=subset_idx,
                kind="linear",
                rng=rng,
            )
            lin_df = pd.DataFrame(M_lin, index=namesA, columns=namesB)
            lin_csv = cka_dir / f"cka_linear_{arch_name}.csv"
            lin_df.to_csv(lin_csv)
            lin_png = cka_dir / f"cka_linear_{arch_name}.png"
            plot_cka_heatmap(M_lin, namesA, namesB, f"Linear CKA — {arch_name} (uniform vs zipf)", lin_png)

            M_rbf, _, _ = compute_layerwise_cka(
                _repr,
                layers_for_cka,
                layers_for_cka,
                tag_A="uniform",
                tag_B="zipfian",
                indices=subset_idx,
                kind="rbf",
                rng=rng,
            )
            rbf_df = pd.DataFrame(M_rbf, index=namesA, columns=namesB)
            rbf_csv = cka_dir / f"cka_rbf_{arch_name}.csv"
            rbf_df.to_csv(rbf_csv)
            rbf_png = cka_dir / f"cka_rbf_{arch_name}.png"
            plot_cka_heatmap(M_rbf, namesA, namesB, f"RBF CKA — {arch_name} (uniform vs zipf)", rbf_png)

            print(
                f"[CKA] {arch_name}/{dist_name}: diag linear={np.diag(M_lin).mean():.3f}, "
                f"diag rbf={np.diag(M_rbf).mean():.3f}"
            )

            if wandb_run is not None:
                try:
                    import wandb

                    wandb_run.log(
                        {
                            "cka/linear": wandb.Image(str(lin_png)),
                            "cka/rbf": wandb.Image(str(rbf_png)),
                        }
                    )
                except Exception:
                    pass

            perm_cfg = cka_cfg.get("permutation") or {}
            if perm_cfg.get("enabled", False):
                n_perm = int(perm_cfg.get("n_perm", 200))
                perm_rng = np.random.default_rng(seed)
                records = []
                for layer in layers_for_cka:
                    Xa = _repr(layer, "uniform")
                    Yb = _repr(layer, "zipfian")
                    if subset_idx is not None:
                        Xa = Xa[subset_idx]
                        Yb = Yb[subset_idx]
                    p_lin = permutation_test_cka(Xa, Yb, n_perm=n_perm, kind="linear", rng=perm_rng)
                    p_rbf = permutation_test_cka(Xa, Yb, n_perm=n_perm, kind="rbf", rng=perm_rng)
                    records.append({
                        "layer": layer,
                        "p_linear": p_lin,
                        "p_rbf": p_rbf,
                    })
                pd.DataFrame(records).to_csv(cka_dir / f"cka_permutation_{arch_name}.csv", index=False)

            report_cfg = cka_cfg.get("report", {})
            if report_cfg.get("enabled", True):
                try:
                    report_dir = cka_dir / "report"
                    acts_uniform_report: list[np.ndarray] = []
                    acts_zipf_report: list[np.ndarray] = []
                    for layer in layers_for_cka:
                        Xa = _repr(layer, "uniform")
                        Yb = _repr(layer, "zipfian")
                        if subset_idx is not None:
                            Xa = Xa[subset_idx]
                            Yb = Yb[subset_idx]
                        acts_uniform_report.append(Xa)
                        acts_zipf_report.append(Yb)

                    generate_report(
                        acts_uniform_report,
                        acts_zipf_report,
                        outdir=report_dir,
                        layer_names_uniform=namesA,
                        layer_names_zipf=namesB,
                        kernels=report_cfg.get("kernels", ["linear", "rbf"]),
                        gamma=report_cfg.get("gamma", None),
                        bootstrap=int(report_cfg.get("bootstrap", 200)),
                        ridge_tau=float(report_cfg.get("ridge_tau", 0.1)),
                        null_permutations=int(report_cfg.get("null_permutations", 1000)),
                        seed=int(report_cfg.get("seed", seed)),
                    )
                except Exception as exc:
                    print(f"[CKA report] Impossibile generare gli asset slide-ready ({exc})")
        else:
            print(f"[CKA] {arch_name}/{dist_name}: no layers available for comparison.")
    else:
        print(f"[CKA] {arch_name}/{dist_name}: skipped (cka.enabled=False)")

    # Behavioral analysis (optional)
    behavioral_cfg = behavioral_cfg or {}
    tasks_cfg = behavioral_cfg.get("tasks", {}) if isinstance(behavioral_cfg, dict) else {}
    model_for_behavior = analyzer.model_uniform if dist_name == "uniform" else analyzer.model_zipfian
    behaviors_dir = out_dir / "behavioral"
    behavior_label = f"{arch_name}_{dist_name}"
    behavioral_inputs = None
    device_behavior: torch.device | None = None
    if behavioral_cfg.get("enabled", False):
        train_path = behavioral_cfg.get("train_pickle")
        test_path = behavioral_cfg.get("test_pickle")
        mat_path = behavioral_cfg.get("mat_file")
        if train_path and test_path and mat_path:
            try:
                device_behavior = _get_model_device(model_for_behavior)
                behavioral_inputs = load_behavioral_inputs(
                    Path(train_path),
                    Path(test_path),
                    Path(mat_path),
                    device_behavior,
                )
            except Exception as exc:
                print(f"[Behavioral] Impossibile caricare i dataset ({exc})")
        else:
            print("[Behavioral] Percorsi mancanti: train_pickle/test_pickle/mat_file")

    if behavioral_inputs is not None:
        results_beh = run_behavioral_analysis(
            model_for_behavior,
            behavioral_inputs,
            behaviors_dir,
            behavior_label,
            guess_rate=float(behavioral_cfg.get("guess_rate", 0.01)),
        )
        if wandb_run is not None:
            try:
                import wandb

                wandb_run.log(
                    {
                        "behavioral/accuracy_test": results_beh.get("accuracy_test"),
                        "behavioral/accuracy_train": results_beh.get("accuracy_train"),
                        "behavioral/beta_number": results_beh.get("beta_number"),
                        "behavioral/beta_size": results_beh.get("beta_size"),
                        "behavioral/beta_spacing": results_beh.get("beta_spacing"),
                        "behavioral/weber_fraction": results_beh.get("weber_fraction"),
                    }
                )
            except Exception:
                pass

        comparison_cfg = tasks_cfg.get("comparison", {})
        if comparison_cfg.get("enabled", False):
            out_cmp = behaviors_dir / "comparison"
            guess_rate_cmp = float(comparison_cfg.get("guess_rate", behavioral_cfg.get("guess_rate", 0.01)))
            try:
                comparison_inputs = behavioral_inputs
                cmp_train = comparison_cfg.get("train_pickle")
                cmp_test = comparison_cfg.get("test_pickle")
                if cmp_train and cmp_test:
                    cmp_mat = comparison_cfg.get("mat_file") or behavioral_cfg.get("mat_file")
                    comparison_inputs = load_behavioral_inputs(
                        Path(cmp_train),
                        Path(cmp_test),
                        Path(cmp_mat) if cmp_mat else Path(behavioral_cfg.get("mat_file")),
                        device_behavior,
                    )
                results_cmp = run_task_comparison(
                    model_for_behavior,
                    comparison_inputs,
                    out_cmp,
                    behavior_label,
                    guess_rate=guess_rate_cmp,
                )
                if wandb_run is not None:
                    wandb_run.log(
                        {
                            "behavioral/comparison/accuracy_test": results_cmp.get("accuracy_test"),
                            "behavioral/comparison/accuracy_train": results_cmp.get("accuracy_train"),
                            "behavioral/comparison/weber_fraction": results_cmp.get("weber_fraction"),
                            "behavioral/comparison/beta_number": results_cmp.get("beta_number"),
                        }
                    )
            except Exception as exc:
                print(f"[Behavioral] Task comparison fallito: {exc}")

        fixed_cfg = tasks_cfg.get("fixed_reference", {})
        if fixed_cfg.get("enabled", False):
            refs = fixed_cfg.get("references", [])
            train_template = fixed_cfg.get("train_template")
            test_template = fixed_cfg.get("test_template")
            mat_path = fixed_cfg.get("mat_file") or behavioral_cfg.get("mat_file")
            if not (train_template and test_template and mat_path):
                print("[Behavioral] Percorsi mancanti per fixed_reference")
            else:
                for ref in refs:
                    try:
                        train_path = Path(train_template.format(ref=ref))
                        test_path = Path(test_template.format(ref=ref))
                        fixed_inputs = load_fixed_reference_inputs(
                            train_path,
                            test_path,
                            Path(mat_path),
                            device_behavior or _get_model_device(model_for_behavior),
                        )
                        results_fixed = run_task_fixed_reference(
                            model_for_behavior,
                            fixed_inputs,
                            behaviors_dir / "fixed_reference",
                            behavior_label,
                            ref_num=int(ref),
                            guess_rate=float(fixed_cfg.get("guess_rate", behavioral_cfg.get("guess_rate", 0.01))),
                        )
                        if wandb_run is not None:
                            wandb_run.log(
                                {
                                    f"behavioral/fixed_reference/{ref}/accuracy_test": results_fixed.get("accuracy_test"),
                                    f"behavioral/fixed_reference/{ref}/weber_fraction": results_fixed.get("weber_fraction"),
                                }
                            )
                    except Exception as exc:
                        print(f"[Behavioral] Fixed reference {ref} fallito: {exc}")

    estimation_cfg = tasks_cfg.get("estimation", {})
    if estimation_cfg.get("enabled", False):
        ds_map = estimation_cfg.get("datasets")
        if isinstance(ds_map, DictConfig):
            ds_map = OmegaConf.to_container(ds_map, resolve=True)
        ds_entry = None
        if isinstance(ds_map, dict):
            ds_entry = ds_map.get(dist_name) or ds_map.get("default") or ds_map.get("all")
        if ds_entry is None:
            ds_entry = {
                "train_pickle": estimation_cfg.get("train_pickle"),
                "test_pickle": estimation_cfg.get("test_pickle"),
            }
        if isinstance(ds_entry, DictConfig):
            ds_entry = OmegaConf.to_container(ds_entry, resolve=True)
        train_est = ds_entry.get("train_pickle")
        test_est = ds_entry.get("test_pickle")
        if train_est and test_est:
            try:
                train_est_path = Path(train_est)
                if not train_est_path.is_absolute():
                    train_est_path = PROJECT_ROOT / train_est_path
                test_est_path = Path(test_est)
                if not test_est_path.is_absolute():
                    test_est_path = PROJECT_ROOT / test_est_path
                device_est = _get_model_device(model_for_behavior)
                train_ds = load_estimation_dataset(train_est_path, device_est)
                test_ds = load_estimation_dataset(test_est_path, device_est)
                est_dir = behaviors_dir / "estimation"
                est_summary = run_task_numerosity_estimation(
                    model_for_behavior,
                    train_ds,
                    test_ds,
                    est_dir,
                    behavior_label,
                    classifiers=estimation_cfg.get("classifiers"),
                    label_mode=estimation_cfg.get("label_mode", "int"),
                    scale_targets=bool(estimation_cfg.get("scale_targets", False)),
                    max_numerosity=estimation_cfg.get("max_numerosity"),
                    distribution=dist_name,
                    max_display_classes=int(estimation_cfg.get("max_display_classes", 32)),
                    wandb_run=wandb_run,
                )
                if wandb_run is not None and est_summary:
                    try:
                        import wandb
                        wandb_run.log(
                            {
                                "behavioral/estimation/results_csv": str(est_summary.get("results_path")),
                                "behavioral/estimation/per_class_csv": str(est_summary.get("per_class_path")),
                            }
                        )
                    except Exception:
                        pass
            except Exception as exc:
                print(f"[Behavioral] Numerosity estimation fallita ({exc})")
        else:
            print(f"[Behavioral] Dataset numerosity estimation mancante per {dist_name}")

    hist_dir = out_dir / "label_histograms"
    hist_dir.mkdir(parents=True, exist_ok=True)
    try:
        plot_label_histogram(
            np.asarray(analyzer.dataset_uniform.labels),
            title=f"Label Histogram (uniform)",
            save_path=hist_dir / "uniform.png",
        )
    except Exception as exc:
        print(f"[Labels] Impossibile generare histogramma uniform: {exc}")
    try:
        plot_label_histogram(
            np.asarray(analyzer.dataset_zipfian.labels),
            title=f"Label Histogram (zipfian)",
            save_path=hist_dir / "zipfian.png",
        )
    except Exception as exc:
        print(f"[Labels] Impossibile generare histogramma zipfian: {exc}")

    # Helper to save simple 2D/3D plots locally in addition to your W&B plots
    def _save_basic_plots(emb: np.ndarray, features: dict, method: str, target_dir: Path, title_prefix: str):
        target_dir.mkdir(parents=True, exist_ok=True)
        feats_to_plot = [
            ("Labels", features.get("Labels")),
            ("Cumulative Area", features.get("cumArea")),
            ("Convex Hull", features.get("CH")),
        ]
        valid = [(name, vals) for name, vals in feats_to_plot if vals is not None and len(vals) == emb.shape[0]]
        if not valid:
            return
        if emb.shape[1] == 2:
            fig, axes = plt.subplots(1, len(valid), figsize=(6 * len(valid), 5))
            if len(valid) == 1:
                axes = [axes]
            for ax, (fname, vals) in zip(axes, valid):
                sc = ax.scatter(emb[:, 0], emb[:, 1], c=vals, cmap="viridis", s=20, alpha=0.85)
                ax.set_title(fname)
                ax.set_xlabel(f"{method}-1")
                ax.set_ylabel(f"{method}-2")
                fig.colorbar(sc, ax=ax, shrink=0.8)
            fig.suptitle(f"{title_prefix} — {method} 2D", fontsize=14)
            fig.tight_layout(rect=[0, 0, 1, 0.94])
            fig.savefig(target_dir / f"{method}_2d_overview.png", dpi=220)
            plt.close(fig)
        if emb.shape[1] == 3:
            from mpl_toolkits.mplot3d import Axes3D  # noqa
            fig = plt.figure(figsize=(6 * len(valid), 5))
            for idx, (fname, vals) in enumerate(valid, start=1):
                ax = fig.add_subplot(1, len(valid), idx, projection='3d')
                sc = ax.scatter(emb[:, 0], emb[:, 1], emb[:, 2], c=vals, cmap="viridis", s=16, alpha=0.8)
                ax.set_title(fname)
                ax.set_xlabel(f"{method}-1")
                ax.set_ylabel(f"{method}-2")
                ax.set_zlabel(f"{method}-3")
                fig.colorbar(sc, ax=ax, shrink=0.7, pad=0.08)
            fig.suptitle(f"{title_prefix} — {method} 3D", fontsize=14)
            fig.tight_layout(rect=[0, 0, 1, 0.94])
            fig.savefig(target_dir / f"{method}_3d_overview.png", dpi=220)
            plt.close(fig)

    # (Top-level reductions and metrics now run per-layer only)

    # 3) PROBE — usa i TUOI probe con adapter
    # Costruisci features dict nel formato atteso dai probe (nomi tolleranti)
    # N.B.: labels le recuperano direttamente dal val_loader; passarle qui non fa male
    ds = analyzer.val_dataloader_uniform.dataset
    idxs = getattr(ds, "indices", np.arange(len(ds)))
    base = getattr(ds, "dataset", ds)
    cumArea_list = [base.cumArea_list[i] for i in idxs] if hasattr(base, "cumArea_list") else None
    CH_list      = [base.CH_list[i] for i in idxs] if hasattr(base, "CH_list") else None
    density_list = [base.density_list[i] for i in idxs] if hasattr(base, "density_list") else None
    mean_item_size_list = [base.mean_item_size_list[i] for i in idxs] if hasattr(base, "mean_item_size_list") else None

    features_for_probes = {}
    if cumArea_list is not None: features_for_probes["Cumulative Area"] = np.asarray(cumArea_list)
    if CH_list is not None:      features_for_probes["Convex Hull"]     = np.asarray(CH_list)
    if density_list is not None: features_for_probes["Density"]          = np.asarray(density_list)
    if mean_item_size_list is not None: features_for_probes["Mean Item Size"] = np.asarray(mean_item_size_list)
    # numerosity (labels) – necessario per includere il target "labels" nei probe
    if labels is not None and len(labels) > 0:
        features_for_probes["Labels"] = np.asarray(labels)
    # (Labels le ricava il probe dal val_loader; ok così)

    feature_dir = out_dir / "feature_analysis"
    feature_dir.mkdir(parents=True, exist_ok=True)
    corr_features: Dict[str, np.ndarray] = {}
    if labels.size:
        corr_features["labels"] = labels.astype(float)
    if cumArea.size:
        corr_features["cumArea"] = cumArea.astype(float)
    if CH.size:
        corr_features["CH"] = CH.astype(float)
    density_arr = features_for_probes.get("Density")
    if density_arr is not None and len(density_arr) == labels.size:
        corr_features["Density"] = density_arr.astype(float)
    mean_arr = features_for_probes.get("Mean Item Size")
    if mean_arr is not None and len(mean_arr) == labels.size:
        corr_features["mean_item_size"] = mean_arr.astype(float)

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
        corr_df = corr_df.rename(columns=rename_map)
        if "labels" in corr_df:
            mask = corr_df["Numerosity"] > 5
            corr_df = corr_df[mask]
        corr_matrix = corr_df.corr(method="kendall")
        corr_matrix.to_csv(feature_dir / f"feature_correlations_{dist_name}.csv")
        plt.figure(figsize=(6, 4))
        sns.heatmap(
            corr_matrix,
            annot=True,
            fmt=".2f",
            cmap="viridis",
            cbar=True,
            square=True,
        )
        plt.title(f"Feature Correlations — {arch_name} ({dist_name})")
        plt.xticks(rotation=45, ha="right")
        plt.yticks(rotation=0)
        plt.tight_layout()
        plt.savefig(feature_dir / f"feature_correlations_{dist_name}.png", dpi=300)
        plt.close()

    # Scegli il modello giusto per i probe (uniform/zipfian) — tipicamente uniform per coerenza
    model_for_probe = analyzer.model_uniform if dist_name == "uniform" else analyzer.model_zipfian
    pr_cfg = probing_cfg or {}
    layers = pr_cfg.get("layers", ["top"]) if pr_cfg.get("enabled", True) else []
    for layer in layers:
        upto = None if str(layer).lower() == "top" else int(layer)
        layer_tag = "top" if upto is None else f"layer{upto}"
        prm = ProbeReadyModel(
            raw_model=model_for_probe,
            val_loader=analyzer.val_dataloader_uniform,  # stesso batch coerente usato per Z
            features_dict=features_for_probes,
            out_dir=out_dir / "probes" / layer_tag,
            wandb_run=wandb_run
        )
        summary_rows = log_linear_probe(
            model=prm,
            epoch=0,
            n_bins=int(pr_cfg.get("n_bins", 5)),
            test_size=float(pr_cfg.get("test_size", 0.2)),
            steps=int(pr_cfg.get("steps", 1000)),
            lr=float(pr_cfg.get("lr", 1e-2)),
            rng_seed=int(seed),
            patience=int(pr_cfg.get("patience", 20)),
            min_delta=0.0,
            save_csv=True,
            upto_layer=upto,
            layer_tag=layer_tag,
        )
        if summary_rows:
            probe_dir = out_dir / "probes" / layer_tag
            probe_dir.mkdir(parents=True, exist_ok=True)
            df_probe = pd.DataFrame(summary_rows, columns=["metric", "accuracy"])
            df_probe.to_csv(probe_dir / "probe_summary.csv", index=False)
            plt.figure(figsize=(max(6, len(summary_rows) * 1.2), 4))
            sns.barplot(data=df_probe, x="metric", y="accuracy", color="steelblue", errorbar=None)
            plt.ylim(0, 1)
            plt.ylabel("Accuracy")
            plt.xlabel("")
            plt.xticks(rotation=45, ha="right")
            plt.title(f"Linear probe accuracy — {arch_name} ({dist_name}) [{layer_tag}]")
            plt.tight_layout()
            plt.savefig(probe_dir / "probe_summary.png", dpi=300)
            plt.close()

    # (Top-level RSA/RDM/monotonicity/partial/traversal now executed per-layer only)
    red_flags = reductions_cfg or {}
    mse_cfg = mse_cfg or {}
    cka_cfg = cka_cfg or {}

    # 6) Analisi per-layer (1..K) con salvataggi in cartelle separate
    model_sel = analyzer.model_uniform if dist_name == "uniform" else analyzer.model_zipfian
    layers = getattr(model_sel, "layers", [])
    if layers:
        device = _get_model_device(model_sel)
        layer_reps: list[np.ndarray] = []
        layer_states: list[torch.Tensor] = []
        with torch.no_grad():
            inputs_device = inputs_cpu.to(device).view(inputs_cpu.shape[0], -1)
            cur = inputs_device
            for li, rbm in enumerate(layers, start=1):
                cur = rbm.forward(cur)
                layer_states.append(cur.detach())
                layer_reps.append(cur.detach().cpu().numpy())
            del inputs_device
        for li, Zl in enumerate(layer_reps, start=1):
            ldir = out_dir / "layers" / f"layer{li}"
            ldir.mkdir(parents=True, exist_ok=True)
            # per-layer monotonicity: complete set of visuals + summary
            mono_l = ldir / "monotonicity"; mono_l.mkdir(parents=True, exist_ok=True)
            cm_l, cls_l = compute_class_centroids(Zl, labels)
            D_l = pairwise_centroid_distances(cm_l, metric="euclidean")
            stats_l1 = plot_distance_vs_deltaN(D_l, cls_l, mono_l / "deltaN_vs_distance.png")
            plot_violin_by_deltaN(D_l, cls_l, mono_l / "violin_by_deltaN.png")
            plot_centroid_heatmap(D_l, cls_l, mono_l / "centroid_heatmap.png")
            stats_l2 = plot_ordinal_trajectory_1d(cm_l, cls_l, mono_l / "ordinal_trajectory_1d.png")
            plot_ordinal_trajectory_2d(cm_l, cls_l, mono_l / "ordinal_trajectory_2d.png")
            save_deltaN_stats_csv(D_l, cls_l, mono_l / "deltaN_stats.csv")
            outliers_info = stats_l1.get("outliers", []) if isinstance(stats_l1, dict) else []
            if outliers_info:
                plot_outlier_pairs(
                    analyzer.dataset_uniform,
                    cm_l,
                    cls_l,
                    outliers_info,
                    mono_l / "outlier_pairs.png",
                )
                save_outlier_pairs_csv(outliers_info, mono_l / "outlier_pairs.csv")
            stats_summary = {k: v for k, v in stats_l1.items() if k != "outliers"}
            stats_summary.update(stats_l2)
            pd.DataFrame([stats_summary]).to_csv(mono_l / "monotonicity_summary.csv", index=False)
            if wandb_run is not None:
                try:
                    import wandb
                    wandb_payload = {
                        f"monotonicity/layer{li}/deltaN_vs_distance": wandb.Image(str(mono_l / "deltaN_vs_distance.png")),
                        f"monotonicity/layer{li}/violin_by_deltaN": wandb.Image(str(mono_l / "violin_by_deltaN.png")),
                        f"monotonicity/layer{li}/centroid_heatmap": wandb.Image(str(mono_l / "centroid_heatmap.png")),
                        f"monotonicity/layer{li}/ordinal_trajectory_1d": wandb.Image(str(mono_l / "ordinal_trajectory_1d.png")),
                        f"monotonicity/layer{li}/ordinal_trajectory_2d": wandb.Image(str(mono_l / "ordinal_trajectory_2d.png")),
                    }
                    if outliers_info:
                        wandb_payload[f"monotonicity/layer{li}/outlier_pairs"] = wandb.Image(str(mono_l / "outlier_pairs.png"))
                    wandb_run.log(wandb_payload)
                except Exception:
                    pass
            # reductions + plots
            red_dir_l = ldir / "reductions"; red_dir_l.mkdir(parents=True, exist_ok=True)
            reductions_l = compute_reductions(
                Zl,
                seed=seed,
                enable_tsne=bool(red_flags.get("tsne", {}).get("enabled", True)),
                enable_mds=bool(red_flags.get("mds", {}).get("enabled", True)),
                enable_umap=bool(red_flags.get("umap", {}).get("enabled", False)),
            )
            reduction_title_prefix = f"{arch_name} - {dist_name} - Layer {li}"
            for name, emb in reductions_l.items():
                method = name.split("_")[0].upper()
                if emb.shape[1] == 2:
                    _ = plot_2d_embedding_and_correlations(
                        emb_2d=emb,
                        features={"Labels": labels, "cumArea": cumArea, "CH": CH},
                        arch_name=arch_name,
                        dist_name=f"{dist_name}/layer{li}",
                        method_name=method,
                        wandb_run=wandb_run
                    )
                if emb.shape[1] == 3:
                    _ = plot_3d_embedding_and_correlations(
                        emb_3d=emb,
                        features={"Labels": labels, "cumArea": cumArea, "CH": CH},
                        arch_name=arch_name,
                        dist_name=f"{dist_name}/layer{li}",
                        method_name=method,
                        wandb_run=wandb_run
                    )
                _save_basic_plots(
                    emb,
                    {"Labels": labels, "cumArea": cumArea, "CH": CH},
                    method,
                    red_dir_l,
                    reduction_title_prefix,
                )
            if mse_cfg.get("enabled", True):
                mse_dir = ldir / "mse"; mse_dir.mkdir(parents=True, exist_ok=True)
                layer_state = layer_states[li - 1].clone()
                with torch.no_grad():
                    rec = layer_state
                    for rbm_back in reversed(layers[:li]):
                        rec = rbm_back.backward(rec)
                recon_flat = rec.detach().cpu().view(orig_flat.shape[0], -1).numpy()
                del rec
                mses = compute_sample_mse(orig_flat, recon_flat)
                df_mse, bin_info = prepare_mse_dataframe(
                    mses=mses,
                    numerosity=labels,
                    cum_area=cumArea,
                    convex_hull=CH,
                    n_bins=int(mse_cfg.get("n_bins", 5)),
                )
                df_mse.to_csv(mse_dir / "mse_samples.csv", index=False)
                plot_mse_heatmap(
                    df_mse,
                    row_col="cumarea_bin",
                    col_col="numerosity",
                    out_path=mse_dir / "heatmap_cumarea_vs_numerosity.png",
                    title=f"{arch_name} - {dist_name} - Layer {li} - MSE (cum-area vs numerosity)",
                    row_label="Cum-area bin",
                    col_label="Numerosity",
                    ascending=False,
                )
                plot_mse_heatmap(
                    df_mse,
                    row_col="convex_hull_bin",
                    col_col="numerosity",
                    out_path=mse_dir / "heatmap_convex_hull_vs_numerosity.png",
                    title=f"{arch_name} - {dist_name} - Layer {li} - MSE (convex hull vs numerosity)",
                    row_label="Convex-hull bin",
                    col_label="Numerosity",
                    ascending=False,
                )
                plot_mse_vs_numerosity(
                    df_mse,
                    feature_col="cumarea_bin",
                    feature_label="Cum-area bin",
                    out_path=mse_dir / "mse_vs_numerosity_by_cumarea.png",
                    title=f"{arch_name} - {dist_name} - Layer {li} - MSE vs numerosity",
                )
                coeff_df, summary_txt, reg_metrics = save_regression_results(df_mse, mse_dir)
                summary_stats = {
                    "mean_mse": float(df_mse["mse"].mean()),
                    "std_mse": float(df_mse["mse"].std()),
                    "n_samples": int(df_mse.shape[0]),
                }
                pd.DataFrame([summary_stats]).to_csv(mse_dir / "mse_summary.csv", index=False)
                with open(mse_dir / "bin_edges.json", "w", encoding="utf-8") as f:
                    json.dump({k: [float(vv) for vv in v] for k, v in bin_info.items()}, f, indent=2)
                if wandb_run is not None:
                    try:
                        import wandb
                        wandb_run.log({
                            f"mse/layer{li}/heatmap_cumarea": wandb.Image(str(mse_dir / "heatmap_cumarea_vs_numerosity.png")),
                            f"mse/layer{li}/heatmap_convex_hull": wandb.Image(str(mse_dir / "heatmap_convex_hull_vs_numerosity.png")),
                            f"mse/layer{li}/mse_vs_numerosity": wandb.Image(str(mse_dir / "mse_vs_numerosity_by_cumarea.png")),
                            f"mse/layer{li}/mean_mse": summary_stats["mean_mse"],
                            f"mse/layer{li}/std_mse": summary_stats["std_mse"],
                        })
                        coeff_table = wandb.Table(dataframe=coeff_df)
                        wandb_run.log({
                            f"mse/layer{li}/regression_coefficients": coeff_table,
                            f"mse/layer{li}/regression_summary": wandb.Html(f"<pre>{summary_txt}</pre>"),
                            f"mse/layer{li}/regression_r2": reg_metrics.get("r2", math.nan),
                            f"mse/layer{li}/regression_adj_r2": reg_metrics.get("adj_r2", math.nan),
                        })
                    except Exception:
                        pass
                del layer_state
            # RSA/RDM/monotonicity/partial
            if (rsa_cfg or {}).get("enabled", True):
                run_rsa_with_fdr(Zl, feats, arch_name, f"{dist_name}_layer{li}", ldir, alpha=float((rsa_cfg or {}).get("alpha", 0.01)))
            if (rdm_cfg or {}).get("enabled", True):
                for m in (rdm_cfg or {}).get("metrics", ["cosine", "euclidean"]):
                    pairwise_class_rdm(Zl, labels, arch_name, f"{dist_name}_layer{li}", ldir, metric=m)
            if (monotonicity_cfg or {}).get("enabled", True):
                mon = monotonicity_deltaN(Zl, labels)
                pd.DataFrame([mon]).to_csv(ldir / "monotonicity.csv", index=False)
            if (partial_rsa_cfg or {}).get("enabled", True):
                prs = partial_rsa_numerosity(Zl, labels, {"cumArea": cumArea, "CH": CH})
                pd.DataFrame([prs]).to_csv(ldir / "partial_rsa.csv", index=False)
            # Latent traversal from this layer's latent
            if tr_cfg.get("enabled", True):
                latent_dir_l = ldir / "latent"; latent_dir_l.mkdir(parents=True, exist_ok=True)
                pcs = tuple(tr_cfg.get("pcs", [0, 1]))
                steps_tr = int(tr_cfg.get("steps", 7))
                delta_tr = float(tr_cfg.get("delta", 2.0))
                latent_grid_on_pca(
                    model_sel, Zl, base_batch,
                    latent_dir_l / "grid_pc0_pc1.png",
                    anchor_idx=anchor_idx, pcs=pcs, steps=steps_tr, delta=delta_tr, seed=seed,
                    start_layer=li
                )

    # 6) Recon grid opzionale su W&B (se vuoi)
    # orig, rec = analyzer.reconstruct_input(input_type=dist_name)
    # if wandb_run is not None:
    #     log_reconstructions_to_wandb(orig, rec, step=0, num_images=8, name=f"recon_{arch_name}_{dist_name}")

    maybe_generate_monotonicity_comparisons(
        arch_name=arch_name,
        out_root=out_root,
        dist_name=dist_name,
        monotonicity_cfg=monotonicity_cfg,
        seed=seed,
        wandb_run=wandb_run,
    )

    if wandb_run is not None:
        try:
            import wandb
            wandb_run.finish()
        except Exception:
            pass

    print(f"✅ Completed analysis for {arch_name} ({dist_name}) → {out_dir}")

# =========================
# HYDRA MAIN
# =========================
@hydra.main(config_path="../configs", config_name="analysis", version_base="1.3")
def main(cfg: DictConfig):
    # Risolvi path relativi perché Hydra cambia cwd
    def ABS(p):
        if p is None:
            return None
        candidate = Path(p)
        if candidate.is_absolute():
            return str(candidate)
        return str((PROJECT_ROOT / candidate).resolve())

    output_root = ABS(cfg.get("output_root", "results/analysis"))

    def resolve_behavioral(cfg_dict):
        if not cfg_dict:
            return {}
        resolved = dict(cfg_dict)
        for key in ("train_pickle", "test_pickle", "mat_file"):
            if resolved.get(key):
                resolved[key] = ABS(resolved[key])
        tasks_cfg = resolved.get("tasks")
        if isinstance(tasks_cfg, dict):
            new_tasks = {}
            for task_name, task_cfg in tasks_cfg.items():
                t_cfg = dict(task_cfg)
                for path_key in ("train_template", "test_template", "mat_file", "train_pickle", "test_pickle"):
                    if t_cfg.get(path_key):
                        try:
                            t_cfg[path_key] = ABS(t_cfg[path_key])
                        except Exception:
                            pass
                datasets_cfg = t_cfg.get("datasets")
                if isinstance(datasets_cfg, dict):
                    new_ds_cfg = {}
                    for ds_name, ds_entry in datasets_cfg.items():
                        if isinstance(ds_entry, dict):
                            entry_copy = dict(ds_entry)
                            for path_key in ("train_pickle", "test_pickle"):
                                if entry_copy.get(path_key):
                                    try:
                                        entry_copy[path_key] = ABS(entry_copy[path_key])
                                    except Exception:
                                        pass
                            new_ds_cfg[ds_name] = entry_copy
                        else:
                            new_ds_cfg[ds_name] = ds_entry
                    t_cfg["datasets"] = new_ds_cfg
                new_tasks[task_name] = t_cfg
            resolved["tasks"] = new_tasks
        return resolved

    default_behavioral_cfg = resolve_behavioral(cfg.get("behavioral", {}))

    for model_params in cfg.get("models", []):
        arch_name   = model_params.get("arch", "unknown_arch")
        dist_name   = model_params.get("distribution", "uniform")
        dataset_path= ABS(model_params.get("dataset_path", cfg.get("dataset_path")))
        dataset_name= model_params.get("dataset_name", cfg.get("dataset_name"))

        model_uniform = ABS(model_params.get("model_uniform", model_params.get("model_path")))
        model_zipfian = ABS(model_params.get("model_zipfian", model_params.get("model_path")))

        use_wandb     = bool(cfg.get("use_wandb", False))
        wandb_project = cfg.get("wandb_project", None)
        wandb_run_name= f"{arch_name}-{dist_name}-offline"

        model_behavioral_cfg = dict(default_behavioral_cfg)
        if "behavioral" in model_params:
            overrides = resolve_behavioral(model_params["behavioral"])
            model_behavioral_cfg.update({k: v for k, v in overrides.items() if v is not None})

        analyze_one_model(
            dataset_path=dataset_path,
            dataset_name=dataset_name,
            arch_name=arch_name,
            dist_name=dist_name,
            model_uniform=model_uniform,
            model_zipfian=model_zipfian,
            out_root=output_root,
            val_size=float(model_params.get("val_size", cfg.get("val_size", 0.05))),
            reductions_cfg=cfg.get("reductions", {}),
            probing_cfg=cfg.get("probing", {}),
            rsa_cfg=cfg.get("rsa", {}),
            rdm_cfg=cfg.get("rdm", {}),
            monotonicity_cfg=cfg.get("monotonicity", {}),
            partial_rsa_cfg=cfg.get("partial_rsa", {}),
            traversal_cfg=cfg.get("traversal", {}),
            cka_cfg=cfg.get("cka", {}),
            mse_cfg=cfg.get("mse", {}),
            behavioral_cfg=model_behavioral_cfg,
            use_wandb=use_wandb,
            wandb_project=wandb_project,
            wandb_run_name=wandb_run_name,
            anchor_idx=int(cfg.get("anchor", 0)),
            seed=int(cfg.get("seed", 42)),
        )

if __name__ == "__main__":
    main()
