from __future__ import annotations

import math
from pathlib import Path
from typing import Dict, Tuple

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import spearmanr
from sklearn.cross_decomposition import PLSRegression
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


def _between_fraction(values: np.ndarray, labels: np.ndarray) -> float:
    total_var = float(np.var(values, ddof=1))
    if total_var <= 0:
        return 0.0
    overall_mean = float(np.mean(values))
    between = 0.0
    for lab in np.unique(labels):
        mask = labels == lab
        if not mask.any():
            continue
        group_mean = float(np.mean(values[mask]))
        between += mask.sum() * (group_mean - overall_mean) ** 2
    return float(between / (total_var * labels.shape[0]))


def _flip_components(scores: np.ndarray, comps: np.ndarray, labels: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    rhos = []
    for idx in range(scores.shape[1]):
        rho, _ = spearmanr(scores[:, idx], labels)
        if np.isnan(rho):
            rho = 0.0
        if rho < 0:
            scores[:, idx] *= -1
            comps[idx] *= -1
            rho = -rho
        rhos.append(float(rho))
    return scores, comps, np.asarray(rhos)


def _projection_angle(weight_vec: np.ndarray, pcs: np.ndarray) -> float | None:
    if pcs.shape[0] < 2:
        return None
    proj = np.zeros_like(weight_vec)
    for i in range(2):
        proj += np.dot(weight_vec, pcs[i]) * pcs[i]
    norm = np.linalg.norm(proj)
    if norm < 1e-12:
        return None
    proj /= norm
    cos_theta = np.clip(np.dot(proj, pcs[0]), -1.0, 1.0)
    return float(np.degrees(np.arccos(cos_theta)))


def generate_pca_decomposition_report(
    X: np.ndarray,
    y: np.ndarray,
    *,
    regime_name: str,
    layer_tag: str,
    out_dir: Path,
    random_state: int = 42,
) -> Dict[str, object]:
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if X.ndim != 2 or y.ndim != 1 or X.shape[0] != y.shape[0] or X.shape[0] < 4:
        return {}

    scaler = StandardScaler()
    X_std = scaler.fit_transform(X)

    pca_samples = PCA(n_components=min(3, X_std.shape[1]), random_state=random_state)
    sample_scores = pca_samples.fit_transform(X_std)
    sample_scores, pca_samples.components_, rho_samples = _flip_components(sample_scores, pca_samples.components_, y)
    evr_samples = pca_samples.explained_variance_ratio_[:3]
    between_fraction = np.array([
        _between_fraction(sample_scores[:, idx], y) for idx in range(sample_scores.shape[1])
    ])

    pls = PLSRegression(n_components=1)
    pls.fit(X_std, y)
    w = pls.x_weights_.ravel()
    if np.linalg.norm(w) > 0:
        w /= np.linalg.norm(w)
    w_pc = pca_samples.components_ @ w
    angle_deg = _projection_angle(w, pca_samples.components_)
    pls_scores = X_std @ w
    rho_pls, _ = spearmanr(pls_scores, y)
    if not math.isnan(rho_pls) and rho_pls < 0:
        w *= -1
        w_pc *= -1
        angle_deg = _projection_angle(w, pca_samples.components_)

    unique_labels = np.unique(y)
    centroids = np.vstack([X_std[y == lab].mean(axis=0) for lab in unique_labels])
    pca_centroids = PCA(n_components=min(3, centroids.shape[1]), random_state=random_state)
    centroid_scores = pca_centroids.fit_transform(centroids)
    centroid_scores, pca_centroids.components_, rho_centroids = _flip_components(
        centroid_scores, pca_centroids.components_, unique_labels
    )
    evr_centroids = pca_centroids.explained_variance_ratio_[:3]

    class_centered = X_std.copy()
    for lab, centroid in zip(unique_labels, centroids):
        class_centered[y == lab] -= centroid
    pca_centered = PCA(n_components=min(3, X_std.shape[1]), random_state=random_state)
    centered_scores = pca_centered.fit_transform(class_centered)
    centered_scores, pca_centered.components_, rho_centered = _flip_components(
        centered_scores, pca_centered.components_, y
    )
    evr_centered = pca_centered.explained_variance_ratio_[:3]

    fig, axes = plt.subplots(2, 2)
    axA, axB, axC, axD = axes.flat

    scA = axA.scatter(sample_scores[:, 0], sample_scores[:, 1], c=y, cmap="viridis", s=8, alpha=0.6)
    axA.axvline(0, color="gray", lw=0.8, alpha=0.5)
    axA.axhline(0, color="gray", lw=0.8, alpha=0.5)
    axA.arrow(0, 0, 1, 0, color="black", width=0.002, length_includes_head=True)
    axA.arrow(0, 0, 0, 1, color="black", width=0.002, length_includes_head=True)
    if np.linalg.norm(w_pc[:2]) > 1e-12:
        vec = w_pc[:2] / np.linalg.norm(w_pc[:2])
        axA.arrow(0, 0, vec[0], vec[1], color="crimson", width=0.003, length_includes_head=True)
    axA.set_title(
        (f"{regime_name} — {layer_tag} samples PCA\n"
         f"ρ₁={rho_samples[0]:.2f}, ρ₂={rho_samples[1]:.2f} | "
         f"EVR₁={evr_samples[0]:.2f}, EVR₂={evr_samples[1]:.2f}\n"
         f"Between₁={between_fraction[0]:.2f}, Between₂={between_fraction[1]:.2f} | "
         f"angle(PLS1,PC1)={angle_deg:.1f}°"),
        fontsize=14,
    )
    axA.set_xlabel("PC1")
    axA.set_ylabel("PC2")
    cbarA = fig.colorbar(scA, ax=axA, orientation="vertical", pad=0.02)
    cbarA.set_label("Numerosity (N)")
    axA.set_aspect("equal", adjustable="box")

    scB = axB.scatter(centroid_scores[:, 0], centroid_scores[:, 1], c=unique_labels, cmap="viridis", s=40, alpha=0.8)
    axB.plot(centroid_scores[:, 0], centroid_scores[:, 1], color="black", lw=0.8, alpha=0.5)
    axB.set_xlabel("PC1")
    axB.set_ylabel("PC2")
    axB.set_title(
        f"{regime_name} — {layer_tag} centroids PCA\nρ₁={rho_centroids[0]:.3f}, EVR₁={evr_centroids[0]:.2f}",
        fontsize=14,
    )
    cbarB = fig.colorbar(scB, ax=axB, orientation="vertical", pad=0.02)
    cbarB.set_label("Numerosity (N)")
    axB.set_aspect("equal", adjustable="box")

    scC = axC.scatter(centered_scores[:, 0], centered_scores[:, 1], c=y, cmap="viridis", s=8, alpha=0.6)
    axC.axvline(0, color="gray", lw=0.8, alpha=0.5)
    axC.axhline(0, color="gray", lw=0.8, alpha=0.5)
    axC.set_xlabel("PC1")
    axC.set_ylabel("PC2")
    axC.set_title(
        f"{regime_name} — {layer_tag} class-centered PCA\nρ₁={rho_centered[0]:.2e}, EVR₁={evr_centered[0]:.2f}",
        fontsize=14,
    )
    cbarC = fig.colorbar(scC, ax=axC, orientation="vertical", pad=0.02)
    cbarC.set_label("Numerosity (N)")
    axC.set_aspect("equal", adjustable="box")

    bars = axD.bar([0, 1], between_fraction[:2], color=["royalblue", "seagreen"], alpha=0.85)
    axD.set_xticks([0, 1])
    axD.set_xticklabels(["PC1", "PC2"])
    axD.set_ylim(0, 1)
    axD.set_ylabel("Between-class variance fraction")
    axD.set_title("Samples PCA between-class share", fontsize=14)
    for bar, val in zip(bars, between_fraction[:2]):
        axD.text(bar.get_x() + bar.get_width() / 2, val + 0.02, f"{val:.2f}", ha="center", va="bottom", fontsize=11)
    axD.text(
        0.5,
        0.1,
        f"angle(PLS1, PC1) = {angle_deg:.1f}°",
        transform=axD.transAxes,
        ha="center",
        fontsize=12,
        bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray"),
    )

    fig.tight_layout(rect=[0, 0, 1, 0.98])
    fig.suptitle(f"PCA decomposition — {regime_name} ({layer_tag})", fontsize=18, y=0.995)
    fig.savefig(out_dir / f"pca_decomposition_{regime_name}_{layer_tag}.png", dpi=300, facecolor="white")
    plt.close(fig)

    lines = [
        "| Set | PC | EVR | Spearman ρ(N) | Between fraction |",
        "|-----|----|-----|---------------|------------------|",
    ]
    for idx in range(len(evr_samples)):
        lines.append(
            f"| Samples | {idx + 1} | {evr_samples[idx]:.4f} | {rho_samples[idx]:.4f} | {between_fraction[idx]:.4f} |"
        )
    lines.append(f"| Centroids | 1 | {evr_centroids[0]:.4f} | {rho_centroids[0]:.4f} | NA |")
    lines.append(f"| Class-centered | 1 | {evr_centered[0]:.4f} | {rho_centered[0]:.4e} | {between_fraction[0]:.4f} |")
    lines.append(f"\nPLS1 vs PC1 angle: {angle_deg:.2f} degrees")
    (out_dir / f"pca_decomposition_{regime_name}_{layer_tag}.md").write_text("\n".join(lines), encoding="utf-8")

    return {
        "samples": dict(evr=evr_samples.tolist(), rho=rho_samples.tolist(), between=between_fraction.tolist()),
        "centroids": dict(evr=evr_centroids.tolist(), rho=rho_centroids.tolist()),
        "class_centered": dict(evr=evr_centered.tolist(), rho=rho_centered.tolist()),
        "angle_deg": angle_deg,
    }
