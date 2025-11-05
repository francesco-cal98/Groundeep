from __future__ import annotations

from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pandas as pd
from skimage.metrics import structural_similarity as ssim

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
        return np.moveaxis(array, 1, -1)
    if array.ndim == 3:
        return array
    raise ValueError(f"Unsupported image array shape {array.shape} for SSIM computation.")


def compute_sample_ssim(originals: np.ndarray, reconstructions: np.ndarray) -> np.ndarray:
    originals = _prepare_images(originals)
    reconstructions = _prepare_images(reconstructions)
    if originals.shape != reconstructions.shape:
        raise ValueError("Original and reconstructed images must have matching shapes.")

    n_samples = originals.shape[0]
    scores = np.empty(n_samples, dtype=np.float64)
    for idx in range(n_samples):
        img0 = originals[idx]
        img1 = reconstructions[idx]
        data_min = min(float(np.min(img0)), float(np.min(img1)))
        data_max = max(float(np.max(img0)), float(np.max(img1)))
        data_range = data_max - data_min
        if data_range <= 0:
            data_range = 1.0
        if img0.ndim == 2:
            scores[idx] = ssim(img0, img1, data_range=data_range)
        else:
            scores[idx] = ssim(img0, img1, channel_axis=-1, data_range=data_range)
    return scores


def prepare_ssim_dataframe(
    scores: np.ndarray,
    numerosity: np.ndarray,
    cum_area: np.ndarray,
    convex_hull: np.ndarray,
    density: np.ndarray | None = None,
    n_bins: int = 5,
) -> Tuple[pd.DataFrame, Dict[str, object]]:
    return prepare_metric_dataframe(
        metric_values=scores,
        numerosity=numerosity,
        cum_area=cum_area,
        convex_hull=convex_hull,
        density=density,
        metric_name="ssim",
        n_bins=n_bins,
    )


def plot_ssim_heatmap(
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
        metric_col="ssim",
        row_col=row_col,
        col_col=col_col,
        out_path=out_path,
        title=title,
        row_label=row_label,
        col_label=col_label,
        cmap="crest",
    )


def plot_ssim_vs_numerosity(
    df,
    *,
    feature_col: str,
    feature_label: str,
    out_path: Path,
    title: str,
) -> None:
    plot_metric_vs_numerosity(
        df,
        metric_col="ssim",
        feature_col=feature_col,
        feature_label_col=f"{feature_col}_label" if f"{feature_col}_label" in df.columns else feature_col,
        feature_label=feature_label,
        out_path=out_path,
        title=title,
    )


def save_ssim_regression_results(df, out_dir: Path):
    return save_metric_regression_results(df, out_dir, metric_col="ssim", filename_stub="ssim_regression")
