"""
Slide-ready CKA analysis module and CLI.

README
------
Running this module produces:
1. `cka_matrix_linear.png` / `cka_matrix_rbf.png` — cross-model CKA heatmaps with contours and match markers.
2. `cka_diagonal_vs_depth.png` — diagonal CKA profiles with bootstrap 95% CIs for both kernels.
3. `cka_best_match_mapping.png` — best-match scatter from Uniform layers to Zipfian layers.
4. `cka_ridge_bandwidth.png` — ridge tightness per layer with early/mid/late averages.
5. `cka_intra_model_progress.png` — intra-model CKA between consecutive layers.
6. `cka_null_test.png` — permutation null distribution for a mid-depth layer pair.
7. `cka_mini_intro.png` — schematic explainer figure for CKA.
8. CSVs mirroring the key numeric outputs (`cka_matrix_*.csv`, `cka_diagonal_bootstrap.csv`,
   `cka_best_match.csv`, `cka_ridge_bandwidth.csv`).

Drop the resulting assets into slides to illustrate how CKA compares Uniform vs Zipfian models.
"""

from __future__ import annotations

import argparse
import json
import math
import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import List, Sequence, Tuple

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

matplotlib.use("Agg")
plt.rcParams.update(
    {
        "font.size": 10,
        "axes.titlesize": 10,
        "axes.labelsize": 10,
        "legend.fontsize": 9,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
    }
)

__all__ = [
    "compute_cka_matrix",
    "bootstrap_diagonal",
    "best_match_indices",
    "ridge_bandwidth",
    "generate_report",
]


# ---------------------------------------------------------------------------
# Core CKA utilities
# ---------------------------------------------------------------------------


def _as_float_matrix(X: np.ndarray) -> np.ndarray:
    X = np.asarray(X, dtype=np.float64)
    if X.ndim != 2:
        raise ValueError("Activation arrays must be 2D (samples x features).")
    return X


def _check_sample_alignment(actsA: Sequence[np.ndarray], actsB: Sequence[np.ndarray]) -> int:
    if not actsA or not actsB:
        raise ValueError("Activation lists must be non-empty.")
    n = actsA[0].shape[0]
    for idx, arr in enumerate(actsA):
        if arr.shape[0] != n:
            raise ValueError(f"Uniform layer {idx} has {arr.shape[0]} samples; expected {n}.")
    for idx, arr in enumerate(actsB):
        if arr.shape[0] != n:
            raise ValueError(f"Zipf layer {idx} has {arr.shape[0]} samples; expected {n}.")
    return n


def _center_gram(K: np.ndarray) -> np.ndarray:
    mean_row = np.mean(K, axis=0, keepdims=True)
    mean_col = np.mean(K, axis=1, keepdims=True)
    mean_all = np.mean(K)
    return K - mean_row - mean_col + mean_all


def _pairwise_sq_dists(X: np.ndarray) -> np.ndarray:
    norms = np.sum(X * X, axis=1, keepdims=True)
    dists = norms + norms.T - 2.0 * (X @ X.T)
    np.maximum(dists, 0.0, out=dists)
    return dists


def _default_gamma(X: np.ndarray) -> float:
    dists = _pairwise_sq_dists(X)
    upper = dists[np.triu_indices_from(dists, k=1)]
    if upper.size == 0:
        return 1.0
    median = np.median(upper)
    if median <= 0.0 or not np.isfinite(median):
        return 1.0
    return 1.0 / max(median, 1e-12)


def _gram_matrix(X: np.ndarray, kernel: str, gamma: float | None) -> np.ndarray:
    if kernel == "linear":
        return X @ X.T
    if kernel == "rbf":
        gamma_eff = _default_gamma(X) if gamma is None else gamma
        return np.exp(-_pairwise_sq_dists(X) * gamma_eff)
    raise ValueError(f"Unsupported kernel '{kernel}'. Expected 'linear' or 'rbf'.")


def _cka(X: np.ndarray, Y: np.ndarray, kernel: str, gamma: float | None) -> float:
    if X.shape[0] != Y.shape[0]:
        raise ValueError("CKA inputs must share the sample dimension.")
    K = _gram_matrix(X, kernel, gamma)
    L = _gram_matrix(Y, kernel, gamma)
    Kc = _center_gram(K)
    Lc = _center_gram(L)
    numerator = np.sum(Kc * Lc)
    denom = math.sqrt(max(np.sum(Kc * Kc), 1e-30) * max(np.sum(Lc * Lc), 1e-30))
    return float(numerator / denom) if denom > 0 else 0.0


def compute_cka_matrix(
    actsA: Sequence[np.ndarray],
    actsB: Sequence[np.ndarray],
    kernel: str = "linear",
    gamma: float | None = None,
) -> np.ndarray:
    """
    Compute the cross-layer CKA matrix between two models' activations.

    Parameters
    ----------
    actsA, actsB : sequences of arrays shaped [N, D_l]
        Layer activations (aligned on samples).
    kernel : {"linear", "rbf"}
        CKA kernel to use.
    gamma : float or None
        Optional RBF gamma; defaults to 1 / median_pairwise_sqdist when None.

    Returns
    -------
    np.ndarray of shape [len(actsA), len(actsB)]
    """
    actsA = [_as_float_matrix(a) for a in actsA]
    actsB = [_as_float_matrix(b) for b in actsB]
    _check_sample_alignment(actsA, actsB)

    out = np.empty((len(actsA), len(actsB)), dtype=np.float64)
    for i, Xa in enumerate(actsA):
        for j, Yb in enumerate(actsB):
            out[i, j] = _cka(Xa, Yb, kernel=kernel, gamma=gamma)
    return out


def bootstrap_diagonal(
    actsA: Sequence[np.ndarray],
    actsB: Sequence[np.ndarray],
    kernel: str,
    B: int = 200,
    gamma: float | None = None,
    random_state: int | None = 0,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Bootstrap the diagonal (layer-wise correspondence) with sample resampling.

    Returns
    -------
    mean : np.ndarray
    ci_lo : np.ndarray
    ci_hi : np.ndarray
    """
    actsA = [_as_float_matrix(a) for a in actsA]
    actsB = [_as_float_matrix(b) for b in actsB]
    n_samples = _check_sample_alignment(actsA, actsB)

    diag_len = min(len(actsA), len(actsB))
    if diag_len == 0:
        raise ValueError("No overlapping layers to compute diagonal CKA.")

    rng = np.random.default_rng(random_state)
    samples = np.empty((B, diag_len), dtype=np.float64)

    for b in range(B):
        idx = rng.integers(0, n_samples, size=n_samples)
        for l in range(diag_len):
            Xa = actsA[l][idx]
            Yb = actsB[l][idx]
            samples[b, l] = _cka(Xa, Yb, kernel=kernel, gamma=gamma)

    mean = np.mean(samples, axis=0)
    ci_lo = np.percentile(samples, 2.5, axis=0)
    ci_hi = np.percentile(samples, 97.5, axis=0)
    return mean, ci_lo, ci_hi


def best_match_indices(M: np.ndarray) -> np.ndarray:
    """Argmax column index for each row (best Zipf match per Uniform layer)."""
    if M.ndim != 2:
        raise ValueError("CKA matrix must be 2D.")
    return np.argmax(M, axis=1)


def ridge_bandwidth(M: np.ndarray, tau: float = 0.1) -> np.ndarray:
    """Softmax-weighted std of Zipf indices for each Uniform layer."""
    if M.ndim != 2:
        raise ValueError("CKA matrix must be 2D.")
    if tau <= 0:
        raise ValueError("Temperature tau must be positive.")

    indices = np.arange(M.shape[1], dtype=np.float64)
    out = np.empty(M.shape[0], dtype=np.float64)

    for i, row in enumerate(M):
        logits = row / tau
        logits -= logits.max()
        weights = np.exp(logits)
        denom = weights.sum()
        if denom <= 0:
            out[i] = np.nan
            continue
        weights /= denom
        mean = np.sum(weights * indices)
        variance = np.sum(weights * (indices - mean) ** 2)
        out[i] = math.sqrt(max(variance, 0.0))
    return out


# ---------------------------------------------------------------------------
# Plotting helpers
# ---------------------------------------------------------------------------


def _prepare_axis(figsize: Tuple[float, float] = (6.0, 4.0)) -> Tuple[plt.Figure, plt.Axes]:
    fig, ax = plt.subplots(figsize=figsize)
    return fig, ax


def _save_fig(fig: plt.Figure, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=300, transparent=False, facecolor="white")
    plt.close(fig)


def _plot_heatmap(
    M: np.ndarray,
    namesA: Sequence[str],
    namesB: Sequence[str],
    title: str,
    out_path: Path,
) -> None:
    row_best = best_match_indices(M)
    col_best = np.argmax(M, axis=0)

    fig, ax = _prepare_axis(figsize=(6 + 0.2 * len(namesB), 4.5))
    cax = ax.imshow(M, origin="lower", cmap="viridis", vmin=0.0, vmax=1.0)

    x = np.arange(M.shape[1])
    y = np.arange(M.shape[0])
    X, Y = np.meshgrid(x, y)
    levels = np.linspace(0.0, 1.0, 11)
    ax.contour(X, Y, M, levels=levels, colors="white", linewidths=0.6, alpha=0.7)

    ax.scatter(row_best, y, facecolors="none", edgecolors="white", linewidths=1.2, s=50, label="Row best")
    ax.scatter(x, col_best, marker="x", color="black", s=45, label="Col best")

    ax.set_xticks(x)
    ax.set_xticklabels(namesB, rotation=45, ha="right")
    ax.set_yticks(y)
    ax.set_yticklabels(namesA)
    ax.set_xlabel("Zipfian layers")
    ax.set_ylabel("Uniform layers")
    ax.set_title(title)
    ax.legend(loc="upper right", frameon=False)

    cbar = fig.colorbar(cax, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("CKA")
    _save_fig(fig, out_path)


def _plot_diagonal(
    diag_data: List["DiagonalBootstrap"],
    layer_labels: Sequence[str],
    out_path: Path,
) -> None:
    if not diag_data:
        return
    x = np.arange(len(layer_labels))
    fig, ax = _prepare_axis(figsize=(6.0, 4.0))

    for data in diag_data:
        ax.plot(x, data.mean, marker="o", label=f"{data.kernel} mean", linewidth=1.6)
        ax.fill_between(x, data.ci_lo, data.ci_hi, alpha=0.2, label=f"{data.kernel} 95% CI")

    ax.set_xticks(x)
    ax.set_xticklabels(layer_labels, rotation=45, ha="right")
    ax.set_ylabel("CKA")
    ax.set_title("Diagonal CKA vs depth")
    ax.set_ylim(0.0, 1.05)
    ax.legend(frameon=False)
    _save_fig(fig, out_path)


def _plot_best_match_scatter(
    matrix: np.ndarray,
    out_path: Path,
) -> None:
    row_best = best_match_indices(matrix)
    if row_best.size == 0:
        fig, ax = _prepare_axis()
        ax.set_axis_off()
        ax.text(0.5, 0.5, "No layers available for best-match mapping.", ha="center", va="center")
        _save_fig(fig, out_path)
        return
    x = np.arange(len(row_best))
    fig, ax = _prepare_axis()
    ax.scatter(x, row_best, color="tab:blue", s=40)
    ax.plot([0, x[-1]], [0, x[-1]], color="gray", linestyle="--", linewidth=1.0)
    ax.set_xlabel("Uniform layer index")
    ax.set_ylabel("Best Zipf match index")
    ax.set_title("Best-match mapping (Uniform→Zipf)")
    _save_fig(fig, out_path)


def _plot_ridge_bandwidth(
    bandwidth: np.ndarray,
    out_path: Path,
) -> None:
    x = np.arange(len(bandwidth))
    segments = np.array_split(x, 3) if len(bandwidth) >= 3 else [x]
    legend_entries = []
    for label, seg in zip(["Early", "Mid", "Late"], segments):
        if seg.size == 0:
            continue
        mean_val = float(np.nanmean(bandwidth[seg]))
        legend_entries.append(f"{label} mean={mean_val:.3f}")

    fig, ax = _prepare_axis()
    ax.plot(x, bandwidth, marker="o", color="tab:purple")
    ax.set_xlabel("Uniform layer index")
    ax.set_ylabel("Softmax std (Zipf index)")
    ax.set_title("Ridge bandwidth (τ=0.1)")
    if legend_entries:
        ax.legend(legend_entries, frameon=False, loc="upper right")
    _save_fig(fig, out_path)


def _plot_intra_model_progress(
    uniform_vals: np.ndarray,
    zipf_vals: np.ndarray,
    kernel: str,
    out_path: Path,
) -> None:
    x_u = np.arange(len(uniform_vals))
    x_z = np.arange(len(zipf_vals))
    fig, ax = _prepare_axis()
    if len(uniform_vals):
        ax.plot(x_u, uniform_vals, marker="o", label="Uniform")
    if len(zipf_vals):
        ax.plot(x_z, zipf_vals, marker="s", label="Zipfian")
    ax.set_xlabel("Layer transition (Lᵗ → Lᵗ⁺¹)")
    ax.set_ylabel(f"CKA ({kernel})")
    ax.set_title("Intra-model layer progress")
    ax.set_ylim(0.0, 1.05)
    ax.legend(frameon=False)
    _save_fig(fig, out_path)


def _plot_null_test(
    null_values: np.ndarray,
    observed: float,
    kernel: str,
    out_path: Path,
) -> None:
    fig, ax = _prepare_axis()
    ax.hist(null_values, bins=30, color="tab:orange", alpha=0.7, edgecolor="white")
    ax.axvline(observed, color="tab:red", linewidth=1.8, label="Observed CKA")
    p_value = (np.sum(null_values >= observed) + 1) / (null_values.size + 1)
    ax.set_xlabel(f"CKA ({kernel})")
    ax.set_ylabel("Count")
    ax.set_title("Permutation null (mid-depth layer pair)")
    ax.legend(frameon=False, title=f"Approx. p-value ≈ {p_value:.3f}")
    _save_fig(fig, out_path)


def _plot_mini_intro(out_path: Path) -> None:
    rng = np.random.default_rng(0)
    cloud_uniform = rng.normal(loc=(-1.0, 0.0), scale=0.6, size=(60, 2))
    cloud_zipf = rng.normal(loc=(1.0, 0.5), scale=0.6, size=(60, 2))
    all_points = np.vstack([cloud_uniform, cloud_zipf])

    fig, axes = plt.subplots(1, 3, figsize=(11.0, 3.8))

    ax0, ax1, ax2 = axes

    ax0.scatter(cloud_uniform[:, 0], cloud_uniform[:, 1], color="tab:blue", label="Uniform", s=25, alpha=0.8)
    ax0.scatter(cloud_zipf[:, 0], cloud_zipf[:, 1], color="tab:orange", label="Zipfian", s=25, alpha=0.8)
    ax0.set_title("Layer activations")
    ax0.set_xticks([])
    ax0.set_yticks([])
    ax0.legend(frameon=False, loc="upper left")

    gram_sample = all_points[:40]
    K = _center_gram(gram_sample @ gram_sample.T)
    L = _center_gram((gram_sample + 0.2) @ (gram_sample + 0.2).T)

    ax1.imshow(K, cmap="viridis")
    ax1.set_title("Centered Gram matrices")
    ax1.set_xticks([])
    ax1.set_yticks([])
    ax1.annotate("K = XXᵀ", xy=(0.05, 0.9), xycoords="axes fraction", color="white", fontsize=10)
    ax1.annotate("L = YYᵀ", xy=(0.55, 0.1), xycoords="axes fraction", color="white", fontsize=10)

    formula = (
        r"$\mathrm{CKA}(X, Y) = \dfrac{\mathrm{HSIC}(K, L)}{\|K\|_F \, \|L\|_F}$"
        "\nwhere HSIC uses centered K, L"
    )
    ax2.axis("off")
    ax2.text(0.5, 0.6, formula, ha="center", va="center", fontsize=11)
    ax2.text(0.5, 0.3, "Invariant to rotations/scaling", ha="center", va="center", fontsize=10, color="gray")

    fig.text(
        0.5,
        0.02,
        "CKA compares representational geometry, invariant to rotations/scaling.",
        ha="center",
        fontsize=10,
    )

    fig.tight_layout(rect=(0.02, 0.05, 0.98, 1.0))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=300, transparent=False, facecolor="white")
    plt.close(fig)


# ---------------------------------------------------------------------------
# CLI helper structures
# ---------------------------------------------------------------------------


@dataclass
class DiagonalBootstrap:
    kernel: str
    mean: np.ndarray
    ci_lo: np.ndarray
    ci_hi: np.ndarray


def _load_activation_list(path: Path) -> List[np.ndarray]:
    if not path.exists():
        raise FileNotFoundError(f"Activation file not found: {path}")
    if path.suffix == ".npz":
        data = np.load(path, allow_pickle=True)
        if "acts" not in data:
            raise KeyError(f"Expected key 'acts' in {path}.npz")
        acts_obj = data["acts"]
    else:
        with open(path, "rb") as f:
            acts_obj = pickle.load(f)
    acts_list = list(acts_obj)
    if not acts_list:
        raise ValueError(f"No activations found in {path}.")
    return [_as_float_matrix(a) for a in acts_list]


def _load_layer_names(path: Path | None, n_layers: int, prefix: str) -> List[str]:
    if path is None:
        return [f"{prefix}L{idx + 1}" for idx in range(n_layers)]
    with open(path, "r", encoding="utf-8") as f:
        names = json.load(f)
    if len(names) != n_layers:
        raise ValueError(f"Expected {n_layers} layer names in {path}, found {len(names)}.")
    return [str(name) for name in names]


def _save_matrix_csv(matrix: np.ndarray, rows: Sequence[str], cols: Sequence[str], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    header = ",".join(["layer"] + [str(col) for col in cols])
    with open(path, "w", encoding="utf-8") as f:
        f.write(header + "\n")
        for name, row in zip(rows, matrix):
            values = ",".join(f"{float(val):.6f}" for val in row)
            f.write(f"{name},{values}\n")


def _save_best_match_csv(
    indices: np.ndarray,
    matrix: np.ndarray,
    out_path: Path,
) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("layer_uniform,best_zipf_index,cka_value\n")
        for idx, best in enumerate(indices):
            value = matrix[idx, best]
            f.write(f"{idx},{best},{value:.6f}\n")


def _save_bandwidth_csv(bandwidth: np.ndarray, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("layer_uniform,bandwidth\n")
        for idx, val in enumerate(bandwidth):
            f.write(f"{idx},{float(val):.6f}\n")


def _save_diagonal_csv(
    diag_entries: List[DiagonalBootstrap],
    layer_labels: Sequence[str],
    out_path: Path,
) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("layer,kernel,mean,ci_lo,ci_hi\n")
        for data in diag_entries:
            for layer, mean_val, lo, hi in zip(layer_labels, data.mean, data.ci_lo, data.ci_hi):
                f.write(
                    f"{layer},{data.kernel},{float(mean_val):.6f},{float(lo):.6f},{float(hi):.6f}\n"
                )


def _intra_model_consecutive(acts: Sequence[np.ndarray], kernel: str, gamma: float | None) -> np.ndarray:
    if len(acts) < 2:
        return np.zeros(0, dtype=np.float64)
    vals = np.empty(len(acts) - 1, dtype=np.float64)
    for i in range(len(acts) - 1):
        vals[i] = _cka(acts[i], acts[i + 1], kernel=kernel, gamma=gamma)
    return vals


def _null_test(
    acts_uniform: Sequence[np.ndarray],
    acts_zipf: Sequence[np.ndarray],
    kernel: str,
    gamma: float | None,
    permutations: int = 1000,
    random_state: int | None = 0,
) -> Tuple[np.ndarray, float]:
    diag_len = min(len(acts_uniform), len(acts_zipf))
    if diag_len == 0:
        raise ValueError("Need at least one layer pair for null test.")
    layer_idx = diag_len // 2
    X = acts_uniform[layer_idx]
    Y = acts_zipf[layer_idx]
    rng = np.random.default_rng(random_state)
    observed = _cka(X, Y, kernel=kernel, gamma=gamma)
    null_values = np.empty(permutations, dtype=np.float64)
    for i in range(permutations):
        perm = rng.permutation(X.shape[0])
        null_values[i] = _cka(X, Y[perm], kernel=kernel, gamma=gamma)
    return null_values, observed


def _ensure_outdir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def generate_report(
    acts_uniform: Sequence[np.ndarray],
    acts_zipf: Sequence[np.ndarray],
    *,
    outdir: Path | str,
    layer_names_uniform: Sequence[str] | None = None,
    layer_names_zipf: Sequence[str] | None = None,
    kernels: Sequence[str] | None = None,
    gamma: float | None = None,
    bootstrap: int = 200,
    ridge_tau: float = 0.1,
    null_permutations: int = 1000,
    seed: int | None = 0,
) -> None:
    """
    Generate the full CKA report directly from activations.

    Parameters mirror the CLI options; see module docstring for outputs.
    """
    outdir = Path(outdir)
    _ensure_outdir(outdir)

    if kernels is None:
        kernels = ("linear", "rbf")
    kernels = list(dict.fromkeys(kernels))  # preserve order, drop duplicates

    acts_uniform = [_as_float_matrix(a) for a in acts_uniform]
    acts_zipf = [_as_float_matrix(a) for a in acts_zipf]
    _check_sample_alignment(acts_uniform, acts_zipf)

    if layer_names_uniform is None:
        layer_names_uniform = [f"UL{idx + 1}" for idx in range(len(acts_uniform))]
    if layer_names_zipf is None:
        layer_names_zipf = [f"ZL{idx + 1}" for idx in range(len(acts_zipf))]

    matrices = {}
    diagonal_entries: List[DiagonalBootstrap] = []
    diag_labels: List[str] = []

    for kernel in kernels:
        matrix = compute_cka_matrix(acts_uniform, acts_zipf, kernel=kernel, gamma=gamma)
        matrices[kernel] = matrix
        heatmap_name = f"cka_matrix_{kernel}.png"
        _plot_heatmap(
            matrix,
            layer_names_uniform,
            layer_names_zipf,
            title=f"{kernel.upper()} CKA — Uniform vs Zipfian",
            out_path=outdir / heatmap_name,
        )
        _save_matrix_csv(
            matrix,
            layer_names_uniform,
            layer_names_zipf,
            outdir / f"cka_matrix_{kernel}.csv",
        )

        mean, ci_lo, ci_hi = bootstrap_diagonal(
            acts_uniform,
            acts_zipf,
            kernel=kernel,
            B=bootstrap,
            gamma=gamma,
            random_state=seed,
        )
        diag_labels = layer_names_uniform[: len(mean)]
        diagonal_entries.append(DiagonalBootstrap(kernel=kernel, mean=mean, ci_lo=ci_lo, ci_hi=ci_hi))

    if diagonal_entries:
        _plot_diagonal(diagonal_entries, diag_labels, outdir / "cka_diagonal_vs_depth.png")
        _save_diagonal_csv(diagonal_entries, diag_labels, outdir / "cka_diagonal_bootstrap.csv")

    if not matrices:
        return

    primary_kernel = "linear" if "linear" in matrices else next(iter(matrices))
    primary_matrix = matrices[primary_kernel]

    _plot_best_match_scatter(primary_matrix, outdir / "cka_best_match_mapping.png")
    best_idx = best_match_indices(primary_matrix)
    _save_best_match_csv(best_idx, primary_matrix, outdir / "cka_best_match.csv")

    bandwidth = ridge_bandwidth(primary_matrix, tau=ridge_tau)
    _plot_ridge_bandwidth(bandwidth, outdir / "cka_ridge_bandwidth.png")
    _save_bandwidth_csv(bandwidth, outdir / "cka_ridge_bandwidth.csv")

    intra_uniform = _intra_model_consecutive(acts_uniform, kernel=primary_kernel, gamma=gamma)
    intra_zipf = _intra_model_consecutive(acts_zipf, kernel=primary_kernel, gamma=gamma)
    _plot_intra_model_progress(
        intra_uniform,
        intra_zipf,
        kernel=primary_kernel,
        out_path=outdir / "cka_intra_model_progress.png",
    )

    null_values, observed = _null_test(
        acts_uniform,
        acts_zipf,
        kernel=primary_kernel,
        gamma=gamma,
        permutations=null_permutations,
        random_state=seed,
    )
    _plot_null_test(null_values, observed, kernel=primary_kernel, out_path=outdir / "cka_null_test.png")

    _plot_mini_intro(outdir / "cka_mini_intro.png")


def run_cli(args: argparse.Namespace) -> None:
    acts_uniform = _load_activation_list(Path(args.uniform))
    acts_zipf = _load_activation_list(Path(args.zipf))

    names_uniform = _load_layer_names(
        Path(args.layer_names_uniform) if args.layer_names_uniform else None,
        len(acts_uniform),
        prefix="U",
    )
    names_zipf = _load_layer_names(
        Path(args.layer_names_zipf) if args.layer_names_zipf else None,
        len(acts_zipf),
        prefix="Z",
    )

    generate_report(
        acts_uniform,
        acts_zipf,
        outdir=args.outdir,
        layer_names_uniform=names_uniform,
        layer_names_zipf=names_zipf,
        kernels=args.kernel,
        gamma=args.gamma,
        bootstrap=args.bootstrap,
        ridge_tau=args.ridge_tau,
        null_permutations=args.null_permutations,
        seed=args.seed,
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Generate slide-ready CKA report.")
    parser.add_argument("--uniform", type=str, required=True, help="Path to Uniform activations (.npz or .pkl).")
    parser.add_argument("--zipf", type=str, required=True, help="Path to Zipf activations (.npz or .pkl).")
    parser.add_argument(
        "--kernel",
        nargs="+",
        default=None,
        choices=["linear", "rbf"],
        help="One or more kernels to evaluate (default: both).",
    )
    parser.add_argument("--outdir", type=str, required=True, help="Output directory for figures and CSVs.")
    parser.add_argument("--layer-names-uniform", type=str, default=None, help="Optional JSON list of Uniform names.")
    parser.add_argument("--layer-names-zipf", type=str, default=None, help="Optional JSON list of Zipf names.")
    parser.add_argument("--gamma", type=float, default=None, help="Optional RBF gamma override.")
    parser.add_argument("--bootstrap", type=int, default=200, help="Bootstrap iterations for diagonal CI.")
    parser.add_argument("--ridge-tau", type=float, default=0.1, help="Softmax temperature for ridge bandwidth.")
    parser.add_argument("--null-permutations", type=int, default=1000, help="Permutations for null test histogram.")
    parser.add_argument("--seed", type=int, default=0, help="Random seed for reproducibility.")
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    run_cli(args)


if __name__ == "__main__":
    main()
