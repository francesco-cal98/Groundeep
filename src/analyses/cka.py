from __future__ import annotations

import math
from pathlib import Path
from typing import Callable, Sequence

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

plt.switch_backend("Agg")


def center_rows(X: np.ndarray) -> np.ndarray:
    """Center each column of X by subtracting the mean (float64)."""
    X = np.asarray(X, dtype=np.float64)
    if X.ndim != 2:
        raise ValueError("X must be a 2D array.")
    return X - X.mean(axis=0, keepdims=True)


def _frobenius_norm_sq(M: np.ndarray) -> float:
    return float(np.sum(M * M))


def linear_cka(X: np.ndarray, Y: np.ndarray) -> float:
    """Linear CKA between two representation matrices X and Y."""
    Xc = center_rows(X)
    Yc = center_rows(Y)

    K = Xc.T @ Yc
    numerator = _frobenius_norm_sq(K)
    denom_x = _frobenius_norm_sq(Xc.T @ Xc)
    denom_y = _frobenius_norm_sq(Yc.T @ Yc)
    denom = math.sqrt(max(denom_x, 1e-30) * max(denom_y, 1e-30))
    return float(numerator / denom) if denom > 0 else 0.0


def _pairwise_sq_dists(X: np.ndarray) -> np.ndarray:
    norms = np.sum(X * X, axis=1, keepdims=True)
    sq = norms + norms.T - 2.0 * (X @ X.T)
    np.maximum(sq, 0.0, out=sq)
    return sq


def rbf_gram(X: np.ndarray, sigma: float | None = None) -> np.ndarray:
    X = np.asarray(X, dtype=np.float64)
    sq = _pairwise_sq_dists(X)
    if sigma is None:
        triu = sq[np.triu_indices_from(sq, k=1)]
        if triu.size == 0:
            sigma = 1.0
        else:
            med_sq = np.median(triu)
            sigma = math.sqrt(max(med_sq, 1e-12))
    gamma = 1.0 / (2.0 * sigma * sigma)
    return np.exp(-sq * gamma)


def rbf_cka(
    X: np.ndarray,
    Y: np.ndarray,
    *,
    sigma_x: float | None = None,
    sigma_y: float | None = None,
) -> float:
    X = np.asarray(X, dtype=np.float64)
    Y = np.asarray(Y, dtype=np.float64)
    if X.shape[0] != Y.shape[0]:
        raise ValueError("X and Y must have the same number of samples.")
    K = rbf_gram(X, sigma_x)
    L = rbf_gram(Y, sigma_y)
    n = K.shape[0]
    H = np.eye(n) - np.full((n, n), 1.0 / n)
    HKH = H @ K @ H
    HLH = H @ L @ H
    numerator = np.sum(HKH * HLH)
    denom = math.sqrt(max(_frobenius_norm_sq(HKH), 1e-30) * max(_frobenius_norm_sq(HLH), 1e-30))
    return float(numerator / denom) if denom > 0 else 0.0


def compute_layerwise_cka(
    get_repr_fn: Callable[[int, str], np.ndarray],
    layers_A: Sequence[int],
    layers_B: Sequence[int],
    *,
    tag_A: str = "A",
    tag_B: str = "B",
    indices: np.ndarray | None = None,
    Nmax: int | None = None,
    kind: str = "linear",
    rng: np.random.Generator | None = None,
) -> tuple[np.ndarray, list[str], list[str]]:
    if not layers_A or not layers_B:
        raise ValueError("Layer lists must be non-empty.")
    rng = rng or np.random.default_rng(0)

    sample = get_repr_fn(layers_A[0], tag_A)
    N = sample.shape[0]
    if indices is None:
        if Nmax is not None and N > Nmax:
            indices = np.sort(rng.choice(N, size=Nmax, replace=False))
    if indices is not None:
        sample = sample[indices]

    namesA = [f"{tag_A}_L{layer}" for layer in layers_A]
    namesB = [f"{tag_B}_L{layer}" for layer in layers_B]
    M = np.zeros((len(layers_A), len(layers_B)), dtype=np.float64)

    for i, layer_a in enumerate(layers_A):
        Xa = get_repr_fn(layer_a, tag_A)
        Xa = Xa if indices is None else Xa[indices]
        for j, layer_b in enumerate(layers_B):
            Yb = get_repr_fn(layer_b, tag_B)
            Yb = Yb if indices is None else Yb[indices]
            if kind == "linear":
                val = linear_cka(Xa, Yb)
            elif kind == "rbf":
                val = rbf_cka(Xa, Yb)
            else:
                raise ValueError(f"Unknown kind '{kind}' (expected 'linear' or 'rbf').")
            M[i, j] = val
    return M, namesA, namesB


def plot_cka_heatmap(
    M: np.ndarray,
    namesA: Sequence[str],
    namesB: Sequence[str],
    title: str,
    out_png: Path,
):
    out_png.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(6 + 0.5 * len(namesB), 5))
    ax = sns.heatmap(
        M,
        vmin=0.0,
        vmax=1.0,
        cmap="viridis",
        annot=True,
        annot_kws={"fontsize": 8},
        xticklabels=namesB,
        yticklabels=namesA,
        square=True,
    )
    ax.set_title(title)
    plt.tight_layout()
    plt.savefig(out_png, dpi=300)
    plt.close()


def permutation_test_cka(
    X: np.ndarray,
    Y: np.ndarray,
    *,
    n_perm: int = 500,
    kind: str = "linear",
    sigma_x: float | None = None,
    sigma_y: float | None = None,
    rng: np.random.Generator | None = None,
) -> float:
    """Permutation test (row-wise shuffle of Y). Returns p-value."""
    rng = rng or np.random.default_rng(0)
    X = np.asarray(X, dtype=np.float64)
    Y = np.asarray(Y, dtype=np.float64)
    if X.shape[0] != Y.shape[0]:
        raise ValueError("X and Y must have the same number of samples.")

    if kind == "linear":
        observed = linear_cka(X, Y)

        def _stat(Y_perm: np.ndarray) -> float:
            return linear_cka(X, Y_perm)

    elif kind == "rbf":
        observed = rbf_cka(X, Y, sigma_x=sigma_x, sigma_y=sigma_y)

        def _stat(Y_perm: np.ndarray) -> float:
            return rbf_cka(X, Y_perm, sigma_x=sigma_x, sigma_y=sigma_y)

    else:
        raise ValueError("kind must be 'linear' or 'rbf'.")

    count = 0
    Y_perm = Y.copy()
    for _ in range(n_perm):
        rng.shuffle(Y_perm, axis=0)
        if _stat(Y_perm) >= observed:
            count += 1
    p_value = (count + 1) / (n_perm + 1)
    return float(p_value)

