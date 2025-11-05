from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterable, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import warnings

plt.switch_backend("Agg")


def _bin_numeric_equal_count(values: Iterable[float], n_bins: int) -> Tuple[np.ndarray, np.ndarray, Dict[int, str]]:
    """
    Bin numeric values using approximately equal-count quantile bins.

    Returns
    -------
    bin_indices : np.ndarray[int]
        Bin assignment for each value (0-indexed).
    edges : np.ndarray[float]
        Inclusive bin edges (len = n_bins'+1).
    labels : Dict[int, str]
        Mapping from bin index to pretty string interval label.
    """
    values = np.asarray(values, dtype=np.float64)
    if values.size == 0:
        return np.zeros(0, dtype=int), np.array([], dtype=np.float64), {}
    finite_mask = np.isfinite(values)
    finite_vals = values[finite_mask]
    if finite_vals.size == 0:
        bins = np.zeros(values.shape[0], dtype=int)
        edges = np.array([0.0, 0.0], dtype=np.float64)
        return bins, edges, {0: "[0, 0]"}
    unique_vals = np.unique(finite_vals)
    if n_bins <= 1 or unique_vals.size <= 1:
        bins = np.zeros(values.shape[0], dtype=int)
        edges = np.array([float(values.min()), float(values.max())], dtype=np.float64)
        return bins, edges, {0: f"[{edges[0]:.3g}, {edges[-1]:.3g}]"}
    q = min(int(n_bins), int(unique_vals.size))
    series = pd.Series(finite_vals)
    try:
        codes_finite, edges = pd.qcut(
            series,
            q=q,
            labels=False,
            retbins=True,
            precision=6,
            duplicates="drop",
        )
    except ValueError:
        bins = np.zeros(values.shape[0], dtype=int)
        vmin, vmax = float(np.nanmin(values)), float(np.nanmax(values))
        edges = np.array([vmin, vmax], dtype=np.float64)
        return bins, edges, {0: f"[{vmin:.3g}, {vmax:.3g}]"}

    codes_finite = np.asarray(codes_finite, dtype=int)
    codes = np.zeros(values.shape[0], dtype=int)
    codes[finite_mask] = codes_finite
    edges = np.asarray(edges, dtype=np.float64)
    labels = {}
    for idx in range(len(edges) - 1):
        left = edges[idx]
        right = edges[idx + 1]
        inclusive = "]" if idx == len(edges) - 2 else ")"
        labels[idx] = f"[{left:.3g}, {right:.3g}{inclusive}"
    return codes, edges, labels


def prepare_metric_dataframe(
    metric_values: Iterable[float],
    numerosity: Iterable[int],
    cum_area: Iterable[float],
    convex_hull: Iterable[float],
    density: Iterable[float] | None = None,
    *,
    metric_name: str,
    n_bins: int = 5,
) -> Tuple[pd.DataFrame, Dict[str, object]]:
    metric_values = np.asarray(metric_values, dtype=np.float64)
    numerosity = np.asarray(numerosity, dtype=int)
    cum_area = np.asarray(cum_area, dtype=np.float64)
    convex_hull = np.asarray(convex_hull, dtype=np.float64)

    data_dict = {
        "numerosity": numerosity,
        "cum_area": cum_area,
        "convex_hull": convex_hull,
        metric_name: metric_values,
    }
    if density is not None:
        data_dict["density"] = np.asarray(density, dtype=np.float64)

    df = pd.DataFrame(data_dict)

    cum_bins, cum_edges, cum_labels = _bin_numeric_equal_count(cum_area, max(2, n_bins))
    hull_bins, hull_edges, hull_labels = _bin_numeric_equal_count(convex_hull, max(2, n_bins))

    df["cumarea_bin"] = cum_bins
    df["convex_hull_bin"] = hull_bins
    df["cumarea_bin_label"] = [cum_labels.get(idx, str(idx)) for idx in cum_bins]
    df["convex_hull_bin_label"] = [hull_labels.get(idx, str(idx)) for idx in hull_bins]

    dens_bins = dens_edges = dens_labels = None
    if density is not None:
        dens_bins, dens_edges, dens_labels = _bin_numeric_equal_count(data_dict["density"], max(2, n_bins))
        df["density_bin"] = dens_bins
        df["density_bin_label"] = [dens_labels.get(idx, str(idx)) for idx in dens_bins]

    info = {
        "cum_area_edges": cum_edges.tolist(),
        "convex_hull_edges": hull_edges.tolist(),
        "cum_area_labels": cum_labels,
        "convex_hull_labels": hull_labels,
    }
    if dens_edges is not None:
        info["density_edges"] = dens_edges.tolist()
        info["density_labels"] = dens_labels

    counts_cum = (
        df.groupby(["numerosity", "cumarea_bin"]).size().unstack(fill_value=0)
        if not df.empty
        else pd.DataFrame()
    )
    counts_hull = (
        df.groupby(["numerosity", "convex_hull_bin"]).size().unstack(fill_value=0)
        if not df.empty
        else pd.DataFrame()
    )
    counts_density = (
        df.groupby(["numerosity", "density_bin"]).size().unstack(fill_value=0)
        if density is not None and "density_bin" in df.columns
        else pd.DataFrame()
    )
    info["combo_counts"] = {
        "cumarea": counts_cum.astype(int).to_dict() if not counts_cum.empty else {},
        "convex_hull": counts_hull.astype(int).to_dict() if not counts_hull.empty else {},
        "density": counts_density.astype(int).to_dict() if not counts_density.empty else {},
    }

    return df, info


def plot_metric_heatmap(
    df: pd.DataFrame,
    *,
    metric_col: str,
    row_col: str,
    col_col: str,
    out_path: Path,
    title: str,
    row_label: str,
    col_label: str,
    cmap: str = "rocket_r",
) -> None:
    if df.empty:
        return
    pivot = df.pivot_table(index=row_col, columns=col_col, values=metric_col, aggfunc="mean")
    if pivot.empty:
        return
    counts = df.groupby([row_col, col_col]).size().unstack(fill_value=0)
    pivot = pivot.reindex(index=counts.index, columns=counts.columns)
    pivot = pivot.sort_index()
    pivot = pivot.reindex(sorted(pivot.columns), axis=1)

    if pivot.isna().any().any() or (counts == 0).any().any():
        missing_pairs = []
        if (counts == 0).any().any():
            missing_idx = np.argwhere((counts == 0).values)
            for r, c in missing_idx:
                missing_pairs.append((counts.index[r], counts.columns[c]))
            if missing_pairs:
                preview = ", ".join(f"({r},{c})" for r, c in missing_pairs[:8])
                suffix = "…" if len(missing_pairs) > 8 else ""
                warnings.warn(
                    f"[recon] Missing {len(missing_pairs)} combinations for ({row_col}, {col_col}); "
                    f"applying interpolation. Sample: {preview}{suffix}"
                )
        interp = pivot.copy()
        interp = interp.apply(lambda col: col.interpolate(limit_direction="both"))
        interp = interp.T.apply(lambda row: row.interpolate(limit_direction="both")).T
        interp = interp.fillna(method="ffill").fillna(method="bfill")
        interp = interp.fillna(method="ffill", axis=1).fillna(method="bfill", axis=1)
        pivot = pivot.where(~pivot.isna(), interp)

    fig, ax = plt.subplots(figsize=(10, 6))
    sns.heatmap(
        pivot,
        annot=True,
        fmt=".3f",
        cmap=cmap,
        cbar=True,
        linewidths=0.5,
        linecolor="white",
        square=False,
        mask=pivot.isna(),
        ax=ax,
        cbar_kws={"shrink": 0.85},
    )
    ax.set_title(title)
    ax.set_xlabel(col_label)
    ax.set_ylabel(row_label)
    ax.tick_params(axis="x", rotation=45)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=300, facecolor="white")
    plt.close(fig)


def plot_metric_vs_numerosity(
    df: pd.DataFrame,
    *,
    metric_col: str,
    feature_col: str,
    feature_label_col: str,
    feature_label: str,
    out_path: Path,
    title: str,
) -> None:
    if df.empty:
        return
    if feature_col not in df.columns or feature_label_col not in df.columns:
        return
    order = sorted(df["numerosity"].unique())
    grouped_all = df.groupby("numerosity")[metric_col].mean().reindex(order)
    grouped_all = grouped_all.interpolate(limit_direction="both")
    grouped_all = grouped_all.fillna(method="ffill").fillna(method="bfill")

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(order, grouped_all.values, label="All", color="black", linestyle="--", linewidth=2)

    bins = df[[feature_col, feature_label_col]].drop_duplicates().sort_values(feature_col)
    palette = sns.color_palette("viridis", n_colors=len(bins)) if len(bins) > 0 else ["tab:blue"]

    for (bin_id, label), color in zip(zip(bins[feature_col], bins[feature_label_col]), palette):
        subset = df[df[feature_col] == bin_id]
        series = subset.groupby("numerosity")[metric_col].mean().reindex(order)
        if series.isna().any():
            missing_nums = [int(n) for n, flag in zip(order, series.isna()) if flag]
            if missing_nums:
                warnings.warn(
                    f"[recon] Missing numerosity values for {feature_label} bin '{label}': "
                    f"{missing_nums}. Interpolating to keep lines continuous."
                )
        series = series.interpolate(limit_direction="both")
        series = series.fillna(method="ffill").fillna(method="bfill")
        ax.plot(series.index, series.values, marker="o", label=label, color=color)

    ax.set_title(title)
    ax.set_xlabel("Numerosity")
    ax.set_ylabel(f"Mean {metric_col.upper()}")
    ax.grid(True, alpha=0.3)
    ax.legend(title=feature_label, frameon=False)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=300, facecolor="white")
    plt.close(fig)


def save_metric_regression_results(
    df: pd.DataFrame,
    out_dir: Path,
    *,
    metric_col: str,
    filename_stub: str | None = None,
) -> Tuple[pd.DataFrame, str, Dict[str, float]]:
    import statsmodels.api as sm

    out_dir.mkdir(parents=True, exist_ok=True)
    X = df[["numerosity", "cum_area", "convex_hull"]].astype(float)
    y = df[metric_col].astype(float)
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

    base = filename_stub or "regression"
    coeff_df.to_csv(out_dir / f"{base}_coefficients.csv", index=False)
    summary_text = model.summary().as_text()
    (out_dir / f"{base}_summary.txt").write_text(summary_text, encoding="utf-8")

    metrics = {"r2": float(model.rsquared), "adj_r2": float(model.rsquared_adj)}
    return coeff_df, summary_text, metrics
