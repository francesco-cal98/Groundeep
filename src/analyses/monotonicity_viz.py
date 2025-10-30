import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Tuple, Dict, List

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
    xs, ys, ci, cj = [], [], [], []
    C = len(classes)
    for i in range(C):
        for j in range(i + 1, C):
            xs.append(abs(int(classes[i]) - int(classes[j])))
            ys.append(D[i, j])
            ci.append(int(classes[i]))
            cj.append(int(classes[j]))
    return (
        np.array(xs, dtype=float),
        np.array(ys, dtype=float),
        np.array(ci, dtype=int),
        np.array(cj, dtype=int),
    )


def plot_distance_vs_deltaN(D: np.ndarray, classes: np.ndarray, out_path: Path) -> Dict[str, float | List[Dict[str, float]]]:
    if D.size == 0 or len(classes) < 2:
        return {"spearman_r": np.nan, "p": np.nan, "pairs": 0, "outliers": []}

    x, y, c_i, c_j = _pairs_deltaN_and_distances(D, classes)

    # Aggregate by ΔN with simple bootstrap CI (percentile)
    uniq = np.unique(x).astype(int)
    medians, lo, hi = [], [], []
    rng = np.random.default_rng(12345)
    median_map = {}
    for d in uniq:
        vals = y[x == d]
        if len(vals) == 0:
            medians.append(np.nan); lo.append(np.nan); hi.append(np.nan)
            continue
        m = np.median(vals)
        medians.append(m)
        median_map[int(d)] = m
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

    # Identify outliers (top residuals above median)
    residuals = y - np.array([median_map[int(d)] for d in x])
    outlier_idx = np.argsort(residuals)[::-1]
    outliers: List[Dict[str, float]] = []
    for idx in outlier_idx[: min(10, len(outlier_idx))]:
        if residuals[idx] <= 0:
            break
        outliers.append(
            {
                "deltaN": float(x[idx]),
                "distance": float(y[idx]),
                "residual": float(residuals[idx]),
                "class_i": float(c_i[idx]),
                "class_j": float(c_j[idx]),
            }
        )

    return {
        "spearman_r": float(rho),
        "p": float(p),
        "pairs": int(len(x)),
        "outliers": outliers,
    }


def plot_violin_by_deltaN(D: np.ndarray, classes: np.ndarray, out_path: Path):
    if D.size == 0 or len(classes) < 2:
        return
    x, y, _, _ = _pairs_deltaN_and_distances(D, classes)
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
    x, y, _, _ = _pairs_deltaN_and_distances(D, classes)
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


def plot_outlier_pairs(
    dataset,
    class_means: np.ndarray,
    classes: np.ndarray,
    outliers: List[Dict[str, float]],
    out_path: Path,
    max_examples: int = 5,
    include_examples: bool = True,
):
    if not outliers or class_means.size == 0:
        return

    out_path.parent.mkdir(parents=True, exist_ok=True)

    n_examples = min(max_examples, len(outliers))
    pca = PCA(n_components=2, random_state=42).fit(class_means)
    coords = pca.transform(class_means)
    class_to_coord = {int(cls): coords[idx] for idx, cls in enumerate(classes)}

    x_min, x_max = coords[:, 0].min(), coords[:, 0].max()
    y_min, y_max = coords[:, 1].min(), coords[:, 1].max()
    margin_x = (x_max - x_min) * 0.1 if x_max > x_min else 1.0
    margin_y = (y_max - y_min) * 0.1 if y_max > y_min else 1.0

    fig, axes = plt.subplots(1, n_examples, figsize=(4.6 * n_examples, 4.6), sharex=True, sharey=True)
    if n_examples == 1:
        axes = [axes]

    data_np = None
    labels_np = None
    image_side = None
    if include_examples and dataset is not None and hasattr(dataset, "data") and hasattr(dataset, "labels"):
        data_tensor = dataset.data.detach().cpu()
        data_np = data_tensor.numpy()
        labels_np = np.asarray(dataset.labels).astype(int)
        if data_np.shape[1] > 0:
            image_side = int(math.sqrt(data_np.shape[1]))

    rng = np.random.default_rng(1234)

    for ax, out in zip(axes, outliers[:n_examples]):
        ax.scatter(coords[:, 0], coords[:, 1], c="lightgray", s=25, label="centroids")
        cls_i = int(out["class_i"])
        cls_j = int(out["class_j"])
        coord_i = class_to_coord.get(cls_i)
        coord_j = class_to_coord.get(cls_j)
        if coord_i is not None and coord_j is not None:
            ax.scatter([coord_i[0]], [coord_i[1]], color="tab:blue", s=70, label="class i")
            ax.scatter([coord_j[0]], [coord_j[1]], color="tab:red", s=70, label="class j")
            ax.plot([coord_i[0], coord_j[0]], [coord_i[1], coord_j[1]], color="tab:purple", linewidth=2, alpha=0.8)
            ax.text(coord_i[0], coord_i[1], f"{cls_i}", ha="center", va="center", color="white", fontsize=9)
            ax.text(coord_j[0], coord_j[1], f"{cls_j}", ha="center", va="center", color="white", fontsize=9)
        ax.set_title(
            f"ΔN={int(out['deltaN'])}\nDist={out['distance']:.3f}\nΔ={out['residual']:.3f}",
            fontsize=11,
        )
        ax.set_xlim(x_min - margin_x, x_max + margin_x)
        ax.set_ylim(y_min - margin_y, y_max + margin_y)
        ax.set_xticks([])
        ax.set_yticks([])

        if include_examples and data_np is not None and image_side is not None:
            mask_i = labels_np == cls_i
            mask_j = labels_np == cls_j
            if mask_i.any() and mask_j.any():
                ex_i = data_np[mask_i][rng.integers(mask_i.sum())].reshape(image_side, image_side)
                ex_j = data_np[mask_j][rng.integers(mask_j.sum())].reshape(image_side, image_side)
                ex_i_disp = ex_i / ex_i.max() if ex_i.max() > 0 else ex_i
                ex_j_disp = ex_j / ex_j.max() if ex_j.max() > 0 else ex_j

                inset_i = ax.inset_axes([0.02, 0.02, 0.42, 0.42])
                inset_i.imshow(ex_i_disp, cmap="gray")
                inset_i.set_title(f"Example class {cls_i}", fontsize=7)
                inset_i.axis("off")

                inset_j = ax.inset_axes([0.52, 0.02, 0.42, 0.42])
                inset_j.imshow(ex_j_disp, cmap="gray")
                inset_j.set_title(f"Example class {cls_j}", fontsize=7)
                inset_j.axis("off")

    plt.tight_layout()
    fig.savefig(out_path, dpi=300)
    plt.close(fig)


def save_outlier_pairs_csv(outliers: List[Dict[str, float]], out_csv: Path):
    if not outliers:
        return
    df = pd.DataFrame(outliers)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_csv, index=False)
