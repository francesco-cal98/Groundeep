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


def _prepare_images(array: np.ndarray) -> np.ndarray:
    array = np.asarray(array, dtype=np.float64)
    if array.ndim == 4:
        if array.shape[1] == 1:
            return array[:, 0]
        return array.mean(axis=1)
    if array.ndim == 3:
        return array
    raise ValueError(f"Unsupported image array shape {array.shape} for AFP computation.")


def compute_sample_afp(originals: np.ndarray, reconstructions: np.ndarray) -> np.ndarray:
    originals = _prepare_images(originals)
    reconstructions = _prepare_images(reconstructions)
    if originals.shape != reconstructions.shape:
        raise ValueError("Original and reconstructed images must have matching shapes.")

    n_samples = originals.shape[0]
    afp_values = np.empty(n_samples, dtype=np.float64)
    for idx in range(n_samples):
        fft_orig = np.abs(np.fft.fft2(originals[idx]))
        fft_recon = np.abs(np.fft.fft2(reconstructions[idx]))
        sum_orig = fft_orig.sum()
        sum_recon = fft_recon.sum()
        if sum_orig > 0:
            fft_orig /= sum_orig
        if sum_recon > 0:
            fft_recon /= sum_recon
        afp_values[idx] = np.sum(np.abs(fft_orig - fft_recon))
    return afp_values


def prepare_afp_dataframe(
    afp: np.ndarray,
    numerosity: np.ndarray,
    cum_area: np.ndarray,
    convex_hull: np.ndarray,
    density: np.ndarray | None = None,
    n_bins: int = 5,
) -> Tuple[pd.DataFrame, Dict[str, object]]:
    return prepare_metric_dataframe(
        metric_values=afp,
        numerosity=numerosity,
        cum_area=cum_area,
        convex_hull=convex_hull,
        density=density,
        metric_name="afp",
        n_bins=n_bins,
    )


def plot_afp_heatmap(
    df,
    *,
    row_col: str,
    col_col: str,
    out_path: Path,
    title: str,
    row_label: str,
    col_label: str,
) -> None:
    plot_metric_heatmap(
        df,
        metric_col="afp",
        row_col=row_col,
        col_col=col_col,
        out_path=out_path,
        title=title,
        row_label=row_label,
        col_label=col_label,
        cmap="plasma",
    )


def plot_afp_vs_numerosity(
    df,
    *,
    feature_col: str,
    feature_label: str,
    out_path: Path,
    title: str,
) -> None:
    plot_metric_vs_numerosity(
        df,
        metric_col="afp",
        feature_col=feature_col,
        feature_label_col=f"{feature_col}_label" if f"{feature_col}_label" in df.columns else feature_col,
        feature_label=feature_label,
        out_path=out_path,
        title=title,
    )


def save_afp_regression_results(df, out_dir: Path):
    return save_metric_regression_results(df, out_dir, metric_col="afp", filename_stub="afp_regression")
