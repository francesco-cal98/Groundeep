from __future__ import annotations

import math
from dataclasses import dataclass, asdict
import json
from pathlib import Path
from typing import Dict, Iterable, Tuple

import matplotlib.pyplot as plt
import numpy as np
from numpy.linalg import eigh, norm
from scipy.stats import spearmanr, norm as norm_dist
from sklearn.decomposition import PCA
from sklearn.manifold import Isomap

__all__ = [
    "run_pca_geometry",
]

EPS = 1e-12


def _zscore_cols(X: np.ndarray) -> np.ndarray:
    X = X.astype(np.float64, copy=False)
    mu = X.mean(axis=0, keepdims=True)
    sd = X.std(axis=0, keepdims=True) + 1e-8
    return (X - mu) / sd


def _scatter_matrices(X: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    classes = np.unique(y)
    n, d = X.shape
    mu = X.mean(axis=0)
    Sw = np.zeros((d, d), dtype=np.float64)
    Sb = np.zeros((d, d), dtype=np.float64)
    for c in classes:
        idx = np.where(y == c)[0]
        Xc = X[idx]
        pc = len(idx) / n
        muc = Xc.mean(axis=0)
        Xc0 = Xc - muc
        if len(idx) > 1:
            Sw += pc * (Xc0.T @ Xc0) / (len(idx) - 1)
        dmu = (muc - mu)[:, None]
        Sb += pc * (dmu @ dmu.T)
    return Sw, Sb


def _eig_sorted(A: np.ndarray):
    w, V = eigh(A)
    idx = np.argsort(w)[::-1]
    return w[idx], V[:, idx]


def _cos2(a: np.ndarray, b: np.ndarray) -> float:
    a_norm = a / (norm(a) + EPS)
    b_norm = b / (norm(b) + EPS)
    return float((a_norm @ b_norm) ** 2)


def _class_centroids(X: np.ndarray, y: np.ndarray):
    classes = np.unique(y)
    centroids = np.stack([X[y == c].mean(axis=0) for c in classes])
    return centroids, classes


def _pca_on(X: np.ndarray, n: int = 3):
    pca = PCA(n_components=min(n, X.shape[1]), random_state=0)
    scores = pca.fit_transform(X)
    return pca, scores


def _balanced_indices(y: np.ndarray, per_class: int) -> np.ndarray:
    rng = np.random.default_rng(0)
    idx_all = []
    for c in np.unique(y):
        Ic = np.where(y == c)[0]
        if len(Ic) <= per_class:
            idx_all.append(Ic)
        else:
            idx_all.append(rng.choice(Ic, per_class, replace=False))
    return np.concatenate(idx_all)


def _curvature_3pt(x: np.ndarray) -> np.ndarray:
    curv = []
    for i in range(1, len(x) - 1):
        curv.append(norm(x[i + 1] - 2 * x[i] + x[i - 1]))
    return np.array(curv)


@dataclass
class VarianceStats:
    trace_within: float
    anisotropy_within: float
    participation_within: float
    trace_between: float
    anisotropy_between: float
    participation_between: float


@dataclass
class AngleStats:
    rho_pc1: float
    rho_pc2: float
    angle_pc1_deg: float
    angle_pc2_deg: float


@dataclass
class BalancedPCAStats:
    rho_pc1: float
    rho_pc2: float


@dataclass
class CurvatureStats:
    mean: float
    low_mean: float
    high_mean: float


@dataclass
class PCAGeometryReport:
    variance: VarianceStats
    angles: AngleStats
    balanced: BalancedPCAStats
    curvature: CurvatureStats


def _variance_report(X: np.ndarray, y: np.ndarray) -> Tuple[VarianceStats, Tuple[np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray]]:
    Sw, Sb = _scatter_matrices(X, y)
    wW, VW = _eig_sorted(Sw)
    wB, VB = _eig_sorted(Sb)
    traceW = float(wW.sum())
    traceB = float(wB.sum())
    anisW = float(wW[0] / (traceW + EPS))
    anisB = float(wB[0] / (traceB + EPS))
    prW = float((traceW ** 2) / (np.dot(wW, wW) + EPS))
    prB = float((traceB ** 2) / (np.dot(wB, wB) + EPS))
    stats = VarianceStats(
        trace_within=traceW,
        anisotropy_within=anisW,
        participation_within=prW,
        trace_between=traceB,
        anisotropy_between=anisB,
        participation_between=prB,
    )
    return stats, (wW, VW), (wB, VB)


def _angle_report(X: np.ndarray, y: np.ndarray) -> AngleStats:
    P, Z = _pca_on(X, n=3)
    rho1 = spearmanr(y, Z[:, 0]).statistic
    rho2 = spearmanr(y, Z[:, 1]).statistic

    centroids, _ = _class_centroids(X, y)
    Pc, _ = _pca_on(centroids, n=2)
    number_axis = Pc.components_[0]

    ang_p1 = math.degrees(math.acos(math.sqrt(_cos2(number_axis, P.components_[0]))))
    ang_p2 = math.degrees(math.acos(math.sqrt(_cos2(number_axis, P.components_[1]))))
    return AngleStats(
        rho_pc1=rho1,
        rho_pc2=rho2,
        angle_pc1_deg=ang_p1,
        angle_pc2_deg=ang_p2,
    )


def _balanced_report(X: np.ndarray, y: np.ndarray, per_class: int) -> BalancedPCAStats:
    idx = _balanced_indices(y, per_class)
    P, Z = _pca_on(X[idx], n=2)
    rho1 = spearmanr(y[idx], Z[:, 0]).statistic
    rho2 = spearmanr(y[idx], Z[:, 1]).statistic
    return BalancedPCAStats(rho_pc1=rho1, rho_pc2=rho2)


def _curvature_report(name: str, X: np.ndarray, y: np.ndarray, outdir: Path, run_isomap: bool) -> CurvatureStats:
    centroids, classes = _class_centroids(X, y)
    P3, Z3 = _pca_on(centroids, n=3)
    order = np.argsort(classes)
    traj = Z3[order]
    curv = _curvature_3pt(traj)
    c_mean = float(curv.mean())
    c_low = float(curv[:10].mean())
    c_high = float(curv[-11:].mean())

    plt.figure(figsize=(5, 4))
    plt.plot(range(2, len(traj)), curv, marker="o", lw=1.5)
    plt.axhline(c_mean, ls="--", lw=1)
    plt.title(f"{name}: PCA-3D curvature along centroids")
    plt.xlabel("Class index")
    plt.ylabel(r"$||x_{i+1} - 2x_i + x_{i-1}||$")
    plt.tight_layout()
    plt.savefig(outdir / f"curvature_centroids_{name}.png", dpi=200)
    plt.close()

    if run_isomap:
        try:
            iso = Isomap(n_components=2, n_neighbors=8)
            z_iso = iso.fit_transform(centroids[order])
            plt.figure(figsize=(4.5, 4))
            plt.plot(z_iso[:, 0], z_iso[:, 1], "-o", ms=4)
            for i, c in enumerate(classes[order]):
                plt.text(z_iso[i, 0], z_iso[i, 1], str(int(c)), fontsize=6)
            plt.title(f"{name}: Centroid trajectory (Isomap-2D)")
            plt.tight_layout()
            plt.savefig(outdir / f"isomap_centroids_{name}.png", dpi=200)
            plt.close()
        except Exception as exc:  # pragma: no cover - optional dependency
            print(f"[pca_geometry] Isomap skipped ({exc})")

    return CurvatureStats(mean=c_mean, low_mean=c_low, high_mean=c_high)


def _plot_angle_figures(X: np.ndarray, y: np.ndarray, outdir: Path, tag: str) -> AngleStats:
    y_int = y.astype(int)
    Xs = _zscore_cols(X)

    # centroids / number axis
    centroids, classes = _class_centroids(Xs, y_int)
    pca_cent = PCA(n_components=2, svd_solver="full").fit(centroids)
    number_axis = pca_cent.components_[0]
    number_axis = number_axis / (norm(number_axis) + EPS)

    # sample PCA
    pca_samples = PCA(n_components=2, svd_solver="full").fit(Xs)
    Z = pca_samples.transform(Xs)
    pc1 = pca_samples.components_[0] / (norm(pca_samples.components_[0]) + EPS)
    pc2 = pca_samples.components_[1] / (norm(pca_samples.components_[1]) + EPS)

    def _angle(u, v):
        c = float(np.clip(np.abs(np.dot(u, v)), 0.0, 1.0))
        return math.degrees(math.acos(c))

    ang_pc1 = _angle(number_axis, pc1)
    ang_pc2 = _angle(number_axis, pc2)

    # scatter + arrows
    outdir.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(8, 6))
    order = np.argsort(y_int)
    sc = plt.scatter(
        Z[order, 0],
        Z[order, 1],
        c=y_int[order],
        cmap="viridis",
        s=6,
        alpha=0.25,
        edgecolor="none",
    )
    plt.colorbar(sc, label="Class")

    centroids_proj = pca_samples.transform(centroids)
    plt.plot(
        centroids_proj[:, 0],
        centroids_proj[:, 1],
        "-o",
        color="black",
        lw=2,
        ms=3,
        alpha=0.9,
        label="Centroid path",
    )

    origin = Z.mean(axis=0)
    proj_num = np.stack([pc1, pc2], axis=0) @ number_axis
    proj_num = proj_num / (norm(proj_num) + EPS)
    scale = np.percentile(np.linalg.norm(Z, axis=1), 80) * 0.6

    def _arrow(vec2, color, label):
        plt.arrow(
            origin[0],
            origin[1],
            vec2[0] * scale,
            vec2[1] * scale,
            head_width=0.07 * scale,
            head_length=0.12 * scale,
            fc=color,
            ec=color,
            lw=2,
            length_includes_head=True,
            label=label,
        )

    _arrow(np.array([1.0, 0.0]), "tab:blue", "PC1")
    _arrow(np.array([0.0, 1.0]), "tab:orange", "PC2")
    _arrow(proj_num, "tab:red", "Number axis")

    plt.legend(loc="lower right", frameon=True)
    plt.title(
        f"{tag} — PCA vs Number axis\n∠(num, PC1)={ang_pc1:.1f}°, ∠(num, PC2)={ang_pc2:.1f}°"
    )
    plt.xlabel("PC1 (samples)")
    plt.ylabel("PC2 (samples)")
    plt.tight_layout()
    plt.savefig(outdir / f"angles_scatter_{tag}.png", dpi=220)
    plt.close()

    # bar plot
    plt.figure(figsize=(4.5, 4))
    vals = [ang_pc1, ang_pc2]
    bars = plt.bar(["∠(num, PC1)", "∠(num, PC2)"], vals, color=["tab:blue", "tab:orange"])
    for bar, val in zip(bars, vals):
        plt.text(bar.get_x() + bar.get_width() / 2, val + 1.5, f"{val:.1f}°", ha="center", va="bottom")
    plt.ylim(0, 95)
    plt.ylabel("Degrees")
    plt.title(f"{tag} — Alignment angles")
    plt.tight_layout()
    plt.savefig(outdir / f"angles_bar_{tag}.png", dpi=220)
    plt.close()

    return AngleStats(
        rho_pc1=spearmanr(y_int, Z[:, 0]).statistic,
        rho_pc2=spearmanr(y_int, Z[:, 1]).statistic,
        angle_pc1_deg=ang_pc1,
        angle_pc2_deg=ang_pc2,
    )


def run_pca_geometry(
    X: np.ndarray,
    y: np.ndarray,
    outdir: Path,
    *,
    tag: str,
    per_class: int = 200,
    run_isomap: bool = False,
) -> PCAGeometryReport:
    outdir.mkdir(parents=True, exist_ok=True)

    y_int = np.asarray(y).astype(int)
    Xs = _zscore_cols(np.asarray(X))

    variance_stats, _, _ = _variance_report(Xs, y_int)
    angle_stats = _plot_angle_figures(X, y_int, outdir, tag)
    counts = np.bincount(y_int)
    counts = counts[counts > 0]
    per_class_eff = int(min(per_class, counts.min())) if counts.size else per_class
    balanced_stats = _balanced_report(Xs, y_int, per_class_eff)
    curvature_stats = _curvature_report(tag, Xs, y_int, outdir, run_isomap)

    report = PCAGeometryReport(
        variance=variance_stats,
        angles=angle_stats,
        balanced=balanced_stats,
        curvature=curvature_stats,
    )

    summary_path = outdir / f"pca_geometry_{tag}.json"
    with summary_path.open("w", encoding="utf-8") as f:
        json_report = {
            "variance": asdict(report.variance),
            "angles": asdict(report.angles),
            "balanced": asdict(report.balanced),
            "curvature": asdict(report.curvature),
        }
        json.dump(json_report, f, indent=2)

    return report
