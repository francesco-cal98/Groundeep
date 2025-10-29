from __future__ import annotations

from pathlib import Path
from typing import Dict, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import statsmodels.api as sm

plt.switch_backend("Agg")


def compute_sample_mse(originals: np.ndarray, reconstructions: np.ndarray) -> np.ndarray:
    """Return per-sample mean squared error between original and reconstructed inputs."""
    originals = np.asarray(originals, dtype=np.float64)
    reconstructions = np.asarray(reconstructions, dtype=np.float64)
    if originals.shape != reconstructions.shape:
        raise ValueError("Originals and reconstructions must share the same shape.")
    diff = originals - reconstructions
    return np.mean(np.square(diff), axis=1)


def _bin_numeric(values: np.ndarray, n_bins: int) -> Tuple[np.ndarray, np.ndarray]:
    values = np.asarray(values, dtype=np.float64)
    if values.size == 0:
        return np.array([], dtype=int), np.array([], dtype=np.float64)
    if n_bins <= 1 or np.all(values == values[0]):
        return np.zeros(values.shape[0], dtype=int), np.array([values.min(), values.max()])
    vmin, vmax = float(values.min()), float(values.max())
    if not np.isfinite(vmin) or not np.isfinite(vmax):
        raise ValueError("Non-finite values encountered while binning numeric feature.")
    edges = np.linspace(vmin, vmax, n_bins + 1)
    # Avoid zero-width bins due to identical values
    edges = np.unique(edges)
    if edges.size <= 2:
        return np.zeros(values.shape[0], dtype=int), edges
    bins = np.digitize(values, edges[1:-1], right=False)
    return bins.astype(int), edges


def prepare_mse_dataframe(
    mses: np.ndarray,
    numerosity: np.ndarray,
    cum_area: np.ndarray,
    convex_hull: np.ndarray,
    n_bins: int = 5,
) -> Tuple[pd.DataFrame, Dict[str, np.ndarray]]:
    mses = np.asarray(mses, dtype=np.float64)
    numerosity = np.asarray(numerosity).astype(int)
    cum_area = np.asarray(cum_area, dtype=np.float64)
    convex_hull = np.asarray(convex_hull, dtype=np.float64)

    df = pd.DataFrame(
        {
            "numerosity": numerosity,
            "cum_area": cum_area,
            "convex_hull": convex_hull,
            "mse": mses,
        }
    )

    cum_bins, cum_edges = _bin_numeric(cum_area, max(2, n_bins))
    hull_bins, hull_edges = _bin_numeric(convex_hull, max(2, n_bins))

    df["cumarea_bin"] = cum_bins
    df["convex_hull_bin"] = hull_bins

    info = {
        "cum_area_edges": cum_edges,
        "convex_hull_edges": hull_edges,
    }

    return df, info


def plot_mse_heatmap(
    df: pd.DataFrame,
    row_col: str,
    col_col: str,
    out_path: Path,
    title: str,
    row_label: str,
    col_label: str,
    ascending: bool = False,
) -> None:
    if df.empty:
        return
    pivot = df.pivot_table(index=row_col, columns=col_col, values="mse", aggfunc="mean")
    if pivot.empty:
        return
    pivot = pivot.sort_index(ascending=ascending)
    pivot = pivot.reindex(sorted(pivot.columns), axis=1)

    plt.figure(figsize=(9, 6))
    sns.heatmap(pivot, annot=True, fmt=".3f", cmap="viridis", cbar=True)
    plt.title(title)
    plt.xlabel(col_label)
    plt.ylabel(row_label)
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=300)
    plt.close()


def plot_mse_vs_numerosity(
    df: pd.DataFrame,
    feature_col: str,
    feature_label: str,
    out_path: Path,
    title: str,
) -> None:
    if df.empty:
        return
    plt.figure(figsize=(8, 5))
    global_series = df.groupby("numerosity")["mse"].mean()
    global_series = global_series.sort_index()
    plt.plot(
        global_series.index,
        global_series.values,
        label="All",
        color="black",
        linestyle="--",
        linewidth=2,
    )

    for bin_id, group in df.groupby(feature_col):
        series = group.groupby("numerosity")["mse"].mean().sort_index()
        plt.plot(series.index, series.values, marker="o", label=f"{feature_label} {bin_id}")

    plt.title(title)
    plt.xlabel("Numerosity")
    plt.ylabel("Mean MSE")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=300)
    plt.close()


def save_regression_results(df: pd.DataFrame, out_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)
    X = df[["numerosity", "cum_area", "convex_hull"]].astype(float)
    y = df["mse"].astype(float)

    X_const = sm.add_constant(X)
    model = sm.OLS(y, X_const, hasconst=True).fit()

    coeff_df = pd.DataFrame(
        {
            "Variable": model.params.index,
            "Coefficient": model.params.values,
            "P-value": model.pvalues.values,
            "CI_lower": model.conf_int().iloc[:, 0].values,
            "CI_upper": model.conf_int().iloc[:, 1].values,
        }
    )
    coeff_df.to_csv(out_dir / "regression_coefficients.csv", index=False)

    summary_text = model.summary().as_text()
    (out_dir / "regression_summary.txt").write_text(summary_text, encoding="utf-8")

    metrics = {"r2": float(model.rsquared), "adj_r2": float(model.rsquared_adj)}
    return coeff_df, summary_text, metrics

