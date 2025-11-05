from __future__ import annotations

from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pandas as pd

from .recon_metric_utils import (
    plot_metric_heatmap,
    plot_metric_vs_numerosity,
    prepare_metric_dataframe,
    save_metric_regression_results,
)


def compute_sample_mse(originals: np.ndarray, reconstructions: np.ndarray) -> np.ndarray:
    """Return per-sample mean squared error between original and reconstructed inputs."""
    originals = np.asarray(originals, dtype=np.float64)
    reconstructions = np.asarray(reconstructions, dtype=np.float64)
    if originals.shape != reconstructions.shape:
        raise ValueError("Originals and reconstructions must share the same shape.")
    diff = originals - reconstructions
    return np.mean(np.square(diff), axis=1)


def prepare_mse_dataframe(
    mses: np.ndarray,
    numerosity: np.ndarray,
    cum_area: np.ndarray,
    convex_hull: np.ndarray,
    density: np.ndarray | None = None,
    n_bins: int = 5,
) -> Tuple[pd.DataFrame, Dict[str, object]]:
    return prepare_metric_dataframe(
        metric_values=mses,
        numerosity=numerosity,
        cum_area=cum_area,
        convex_hull=convex_hull,
        density=density,
        metric_name="mse",
        n_bins=n_bins,
    )


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
    plot_metric_heatmap(
        df,
        metric_col="mse",
        row_col=row_col,
        col_col=col_col,
        out_path=out_path,
        title=title,
        row_label=row_label,
        col_label=col_label,
        cmap="magma",
    )


def plot_mse_vs_numerosity(
    df: pd.DataFrame,
    feature_col: str,
    feature_label: str,
    out_path: Path,
    title: str,
) -> None:
    plot_metric_vs_numerosity(
        df,
        metric_col="mse",
        feature_col=feature_col,
        feature_label_col=f"{feature_col}_label" if f"{feature_col}_label" in df.columns else feature_col,
        feature_label=feature_label,
        out_path=out_path,
        title=title,
    )


def save_regression_results(df: pd.DataFrame, out_dir: Path):
    return save_metric_regression_results(df, out_dir, metric_col="mse")
