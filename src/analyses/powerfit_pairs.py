from __future__ import annotations

import math
from pathlib import Path
from typing import Dict, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import pairwise_distances

plt.switch_backend("Agg")


def build_pairwise_xy(
    Z: np.ndarray,
    labels: np.ndarray,
    metric: str = "euclidean",
) -> Tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    """Compute all pairwise (ΔN, distance) from class centroids."""

    if Z.ndim != 2:
        raise ValueError("Z must be a 2D array of embeddings (N, D).")
    labels = np.asarray(labels)
    if labels.shape[0] != Z.shape[0]:
        raise ValueError("labels and Z must have the same first dimension.")

    classes = np.unique(labels)
    classes = np.sort(classes.astype(int))
    # Compute centroids
    centroids = []
    for c in classes:
        idx = labels == c
        if not np.any(idx):
            continue
        centroids.append(Z[idx].mean(axis=0))
    if not centroids:
        return np.array([]), np.array([]), pd.DataFrame(columns=["i", "j", "deltaN", "distance"])

    centroids = np.vstack(centroids)
    # Recompute classes in case some were skipped (shouldn't happen with mean over empty sets)
    valid_classes = classes[: centroids.shape[0]]

    D = pairwise_distances(centroids, metric=metric)
    xs, ys, rows = [], [], []
    for idx_i in range(len(valid_classes)):
        for idx_j in range(idx_i + 1, len(valid_classes)):
            dist = float(D[idx_i, idx_j])
            if not np.isfinite(dist) or dist <= 0:
                continue
            i_val = int(valid_classes[idx_i])
            j_val = int(valid_classes[idx_j])
            delta = abs(i_val - j_val)
            if delta == 0:
                continue
            xs.append(delta)
            ys.append(dist)
            rows.append({"i": i_val, "j": j_val, "deltaN": delta, "distance": dist})

    pairs_df = pd.DataFrame(rows)
    if not pairs_df.empty:
        pairs_df.sort_values(["deltaN", "i", "j"], inplace=True)
        pairs_df.reset_index(drop=True, inplace=True)
    return np.asarray(xs, dtype=float), np.asarray(ys, dtype=float), pairs_df


def fit_power_loglog_pairs(x: np.ndarray, y: np.ndarray) -> Dict[str, np.ndarray]:
    """Fit y = a * x**b using log-log linear regression."""

    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    mask = (x > 0) & (y > 0) & np.isfinite(x) & np.isfinite(y)
    x = x[mask]
    y = y[mask]
    if x.size == 0:
        raise ValueError("No valid positive pairs available for power-law fit.")

    log_x = np.log(x)
    log_y = np.log(y)

    A = np.column_stack([np.ones_like(log_x), log_x])
    alpha, b = np.linalg.lstsq(A, log_y, rcond=None)[0]
    yhat_log = A @ np.array([alpha, b])
    ss_res = np.sum((log_y - yhat_log) ** 2)
    ss_tot = np.sum((log_y - log_y.mean()) ** 2)
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 1.0

    a = math.exp(alpha)
    yhat = a * np.power(x, b)
    residuals = y - yhat

    return {
        "a": a,
        "b": b,
        "alpha": alpha,
        "r2": r2,
        "x": x,
        "y": y,
        "yhat": yhat,
        "residuals": residuals,
        "n_points": x.size,
    }


def plot_pairs_fit(x: np.ndarray, y: np.ndarray, fit: Dict[str, np.ndarray], out_png: Path, title: str) -> None:
    if x.size == 0:
        return
    out_png.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(7, 5))
    plt.scatter(x, y, s=18, alpha=0.25, color="tab:blue", label="pairs")
    xs_line = np.linspace(x.min(), x.max(), 200)
    ys_line = fit["a"] * np.power(xs_line, fit["b"])
    plt.plot(xs_line, ys_line, color="tab:red", linewidth=2.5, label=r"fit $y=ax^b$")
    plt.xlabel("ΔN")
    plt.ylabel("Centroid distance")
    plt.title(title)
    plt.suptitle(f"b = {fit['b']:.3f} | R² = {fit['r2']:.3f}", fontsize=10, y=0.94)
    plt.legend()
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig(out_png, dpi=300)
    plt.close()


def plot_pairs_fit_loglog(x: np.ndarray, y: np.ndarray, fit: Dict[str, np.ndarray], out_png: Path, title: str) -> None:
    if x.size == 0:
        return
    out_png.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(7, 5))
    plt.scatter(x, y, s=18, alpha=0.25, color="tab:green", label="pairs")
    xs_line = np.logspace(np.log10(x.min()), np.log10(x.max()), 200)
    ys_line = fit["a"] * np.power(xs_line, fit["b"])
    plt.plot(xs_line, ys_line, color="tab:red", linewidth=2.0, label=r"fit $y=ax^b$")
    plt.xscale("log")
    plt.yscale("log")
    plt.xlabel("ΔN (log scale)")
    plt.ylabel("Centroid distance (log scale)")
    plt.title(title)
    plt.suptitle(f"b = {fit['b']:.3f} | R² = {fit['r2']:.3f}", fontsize=10, y=0.94)
    plt.legend()
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig(out_png, dpi=300)
    plt.close()


def save_pairs_fit(fit: Dict[str, np.ndarray], out_csv: Path) -> None:
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    params_df = pd.DataFrame(
        [
            {
                "a": fit["a"],
                "b": fit["b"],
                "r2": fit["r2"],
                "n_points": fit["n_points"],
            }
        ]
    )
    params_df.to_csv(out_csv, index=False)

    residuals = np.asarray(fit["residuals"], dtype=float)
    if residuals.size == 0:
        return
    rmse = math.sqrt(np.mean(residuals ** 2)) if residuals.size else math.nan
    summary_df = pd.DataFrame(
        [
            {
                "residual_median": float(np.median(residuals)),
                "residual_q25": float(np.percentile(residuals, 25)),
                "residual_q75": float(np.percentile(residuals, 75)),
                "rmse": rmse,
            }
        ]
    )
    summary_df.to_csv(out_csv.parent / "residuals_summary.csv", index=False)

