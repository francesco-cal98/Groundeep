import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Tuple, Dict

from sklearn.metrics import pairwise_distances
from sklearn.decomposition import PCA
from scipy.stats import spearmanr

plt.switch_backend("Agg")


def compute_class_centroids(Z: np.ndarray, labels: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    classes = np.unique(labels)
    means = []
    for c in classes:
        idx = (labels == c)
        if idx.sum() == 0:
            continue
        means.append(Z[idx].mean(axis=0))
    class_means = np.vstack(means) if means else np.empty((0, Z.shape[1]))
    return class_means, classes


def pairwise_centroid_distances(class_means: np.ndarray, metric: str = "euclidean") -> np.ndarray:
    if class_means.size == 0:
        return np.empty((0, 0))
    return pairwise_distances(class_means, metric=metric)


def _pairs_deltaN_and_distances(D: np.ndarray, classes: np.ndarray):
    xs, ys = [], []
    C = len(classes)
    for i in range(C):
        for j in range(i + 1, C):
            xs.append(abs(int(classes[i]) - int(classes[j])))
            ys.append(D[i, j])
    return np.array(xs, dtype=float), np.array(ys, dtype=float)


def plot_distance_vs_deltaN(D: np.ndarray, classes: np.ndarray, out_path: Path) -> Dict[str, float]:
    if D.size == 0 or len(classes) < 2:
        return {"spearman_r": np.nan, "p": np.nan, "pairs": 0}

    x, y = _pairs_deltaN_and_distances(D, classes)

    # Aggregate by ΔN with simple bootstrap CI (percentile)
    uniq = np.unique(x).astype(int)
    medians, lo, hi = [], [], []
    rng = np.random.default_rng(12345)
    for d in uniq:
        vals = y[x == d]
        if len(vals) == 0:
            medians.append(np.nan); lo.append(np.nan); hi.append(np.nan)
            continue
        medians.append(np.median(vals))
        # bootstrap
        if len(vals) > 1:
            boots = [np.median(rng.choice(vals, size=len(vals), replace=True)) for _ in range(200)]
            lo.append(np.percentile(boots, 2.5))
            hi.append(np.percentile(boots, 97.5))
        else:
            lo.append(vals[0]); hi.append(vals[0])

    rho, p = spearmanr(x, y)

    plt.figure(figsize=(7, 5))
    plt.scatter(x, y, color="gray", alpha=0.25, s=16, label="pairs")
    plt.plot(uniq, medians, marker="o", color="tab:blue", label="median")
    for xd, l, h in zip(uniq, lo, hi):
        if np.isnan(l) or np.isnan(h):
            continue
        plt.vlines(xd, l, h, colors="tab:blue", alpha=0.8)
    plt.xlabel("ΔN (|i-j|)")
    plt.ylabel("Centroid distance")
    plt.title(f"ΔN vs distance — Spearman ρ={rho:.3f}, p={p:.2e}")
    plt.legend()
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=300)
    plt.close()
    return {"spearman_r": float(rho), "p": float(p), "pairs": int(len(x))}


def plot_violin_by_deltaN(D: np.ndarray, classes: np.ndarray, out_path: Path):
    if D.size == 0 or len(classes) < 2:
        return
    x, y = _pairs_deltaN_and_distances(D, classes)
    df = pd.DataFrame({"deltaN": x.astype(int), "distance": y})
    plt.figure(figsize=(8, 5))
    sns.violinplot(data=df, x="deltaN", y="distance", inner="box", cut=0)
    plt.title("Distribution of distances by ΔN")
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=300)
    plt.close()


def plot_centroid_heatmap(D: np.ndarray, classes: np.ndarray, out_path: Path):
    if D.size == 0:
        return
    plt.figure(figsize=(7, 6))
    sns.heatmap(D, cmap="viridis", square=True, xticklabels=classes.astype(int), yticklabels=classes.astype(int))
    plt.xlabel("Numerosity"); plt.ylabel("Numerosity")
    plt.title("Centroid distance (class×class)")
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=300)
    plt.close()


def plot_ordinal_trajectory_1d(class_means: np.ndarray, classes: np.ndarray, out_path: Path) -> Dict[str, float]:
    if class_means.size == 0:
        return {"spearman_r_1d": np.nan, "p": np.nan}
    pca = PCA(n_components=1, random_state=42).fit(class_means)
    s = pca.transform(class_means).reshape(-1)
    order = np.argsort(classes)
    s_ord = s[order]
    c_ord = classes[order]
    rho, p = spearmanr(c_ord, s_ord)
    plt.figure(figsize=(7, 4))
    plt.plot(c_ord, s_ord, marker="o")
    plt.xlabel("Class (numerosity)"); plt.ylabel("PCA-1 score")
    plt.title(f"Ordinal 1D trajectory — Spearman ρ={rho:.3f}, p={p:.2e}")
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=300)
    plt.close()
    return {"spearman_r_1d": float(rho), "p": float(p)}


def plot_ordinal_trajectory_2d(class_means: np.ndarray, classes: np.ndarray, out_path: Path):
    if class_means.size == 0:
        return
    pca = PCA(n_components=2, random_state=42).fit(class_means)
    s2 = pca.transform(class_means)
    order = np.argsort(classes)
    c_ord = classes[order]
    s2_ord = s2[order]
    plt.figure(figsize=(7, 6))
    plt.scatter(s2[:, 0], s2[:, 1], c=classes, cmap="viridis", s=40)
    for (x, y), c in zip(s2, classes):
        plt.text(x, y, str(int(c)), fontsize=8, ha="center", va="center")
    # connect ordered classes
    plt.plot(s2_ord[:, 0], s2_ord[:, 1], color="gray", alpha=0.7)
    plt.xlabel("PCA-1"); plt.ylabel("PCA-2")
    plt.title("Ordinal 2D trajectory (class means)")
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=300)
    plt.close()


def save_deltaN_stats_csv(D: np.ndarray, classes: np.ndarray, out_csv: Path):
    if D.size == 0:
        pd.DataFrame(columns=["deltaN", "median", "mean", "std", "q25", "q75", "n_pairs"]).to_csv(out_csv, index=False)
        return
    x, y = _pairs_deltaN_and_distances(D, classes)
    uniq = np.unique(x).astype(int)
    rows = []
    for d in uniq:
        vals = y[x == d]
        if len(vals) == 0:
            continue
        rows.append({
            "deltaN": int(d),
            "median": float(np.median(vals)),
            "mean": float(np.mean(vals)),
            "std": float(np.std(vals)),
            "q25": float(np.quantile(vals, 0.25)),
            "q75": float(np.quantile(vals, 0.75)),
            "n_pairs": int(len(vals)),
        })
    df = pd.DataFrame(rows).sort_values("deltaN")
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_csv, index=False)

