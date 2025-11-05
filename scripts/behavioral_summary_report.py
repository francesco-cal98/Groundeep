#!/usr/bin/env python
"""Aggregate behavioural metrics across architectures and generate summary table/plots.

Usage:
    python scripts/behavioral_summary_report.py \
        --analysis-root results/analysis \
        --net-root-uniform /home/student/Desktop/Groundeep/networks/uniform/dataset_10_10 \
        --net-root-zipfian /home/student/Desktop/Groundeep/networks/zipfian/dataset_10_10 \
        --outdir results/analysis/behavioral_summary

The script expects that behavioural outputs produced by `src/main_scripts/analyze.py`
exist for each architecture under `results/analysis/<distribution>/<arch>/behavioral/`.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.stats import norm, zscore

ANALYSIS_TASKS = {
    "pairwise": "comparison",
    "fixed_reference": "fixed_reference",
    "estimation": "estimation",
}

PAIRWISE_MAT = Path("circle_dataset_100x100/NumStim_7to28_100x100_TE.mat")
FIXED_MAT = Path("circle_dataset_100x100/NumStim_1to32_100x100_TE.mat")
GUESS_RATE = 0.01

sns.set_theme(style="whitegrid", context="talk")


def _load_mat_arrays(mat_path: Path) -> Dict[str, np.ndarray]:
    from scipy.io import loadmat

    mat = loadmat(mat_path)
    return {
        "N_list": np.squeeze(mat["N_list"]),
        "TSA_list": np.squeeze(mat["TSA_list"]),
        "FA_list": np.squeeze(mat["FA_list"]),
    }


def _flatten_idxs(idxs: np.ndarray) -> np.ndarray:
    arr = np.asarray(idxs)
    return arr.reshape(-1, arr.shape[-1]).astype(int)


def _compute_pairwise_features(idx_flat: np.ndarray, arrays: Dict[str, np.ndarray]) -> pd.DataFrame:
    idx_zero = idx_flat - 1
    N = arrays["N_list"]
    TSA = arrays["TSA_list"]
    FA = arrays["FA_list"]

    num_left = N[idx_zero[:, 0]]
    num_right = N[idx_zero[:, 1]]
    isa_left = TSA[idx_zero[:, 0]] / num_left
    isa_right = TSA[idx_zero[:, 1]] / num_right
    fa_left = FA[idx_zero[:, 0]]
    fa_right = FA[idx_zero[:, 1]]

    tsa_left = isa_left * num_left
    tsa_right = isa_right * num_right
    size_left = isa_left * tsa_left
    size_right = isa_right * tsa_right
    sparsity_left = fa_left / num_left
    sparsity_right = fa_right / num_right
    space_left = sparsity_left * fa_left
    space_right = sparsity_right * fa_right

    return pd.DataFrame(
        {
            "log_num": np.log2(num_right / num_left),
            "log_size": np.log2(size_right / size_left),
            "log_space": np.log2(space_right / space_left),
        }
    )


def _compute_fixed_features(idx_flat: np.ndarray, ref_num: int, arrays: Dict[str, np.ndarray]) -> pd.DataFrame:
    idxs = idx_flat.astype(int).ravel()
    N = arrays["N_list"]
    TSA = arrays["TSA_list"]
    FA = arrays["FA_list"]

    num = N[idxs]
    isa = TSA[idxs] / num
    fa = FA[idxs]

    tsa = isa * num
    size = isa * tsa
    spar = fa / num
    space = spar * fa

    # Map unique values to z-scores of log2 as in beta_extraction_ref_z
    def z_map(values: np.ndarray) -> Dict[float, float]:
        unique = np.unique(values)
        zvals = zscore(np.log2(unique))
        return {float(v): float(z) for v, z in zip(unique, zvals)}

    num_map = z_map(num)
    size_map = z_map(size)
    space_map = z_map(space)

    num_z = np.array([num_map[float(v)] for v in num])
    size_z = np.array([size_map[float(v)] for v in size])
    space_z = np.array([space_map[float(v)] for v in space])

    log_ratio = np.log2(num / ref_num)

    return pd.DataFrame(
        {
            "log_ratio": log_ratio,
            "num_z": num_z,
            "size_z": size_z,
            "space_z": space_z,
        }
    )


def _predict_probabilities_pair(df_feat: pd.DataFrame, intercept: float, betas: Iterable[float]) -> np.ndarray:
    beta_num, beta_size, beta_space = betas
    linear = (
        intercept
        + beta_num * df_feat["log_num"].to_numpy()
        + beta_size * df_feat["log_size"].to_numpy()
        + beta_space * df_feat["log_space"].to_numpy()
    )
    prob = (1 - GUESS_RATE) * (norm.cdf(linear) - 0.5) + 0.5
    return prob


def _predict_probabilities_fixed(df_feat: pd.DataFrame, intercept: float, betas: Iterable[float]) -> pd.DataFrame:
    beta_num, beta_size, beta_space = betas
    linear = (
        intercept
        + beta_num * df_feat["num_z"].to_numpy()
        + beta_size * df_feat["size_z"].to_numpy()
        + beta_space * df_feat["space_z"].to_numpy()
    )
    prob = (1 - GUESS_RATE) * (norm.cdf(linear) - 0.5) + 0.5
    df = df_feat.copy()
    df["prob"] = prob
    return df


@dataclass
class ArchitectureBehavioral:
    distribution: str
    architecture: str
    pairwise: Dict[str, float] = field(default_factory=dict)
    fixed_reference: pd.DataFrame | None = None
    estimation: Dict[str, float] = field(default_factory=dict)
    pairwise_curve: pd.DataFrame | None = None
    fixed_curve: pd.DataFrame | None = None
    pairwise_betas: pd.Series | None = None
    fixed_betas: pd.DataFrame | None = None
    estimation_confusion: np.ndarray | None = None
    estimation_labels: Iterable[int] | None = None


def collect_architectures(analysis_root: Path) -> Dict[str, List[Path]]:
    arch_paths: Dict[str, List[Path]] = {"uniform": [], "zipfian": []}
    for dist in arch_paths:
        dist_root = analysis_root / dist
        if not dist_root.exists():
            continue
        for arch_dir in dist_root.iterdir():
            behavioral = arch_dir / "behavioral"
            if behavioral.is_dir():
                arch_paths[dist].append(arch_dir)
    return arch_paths


def load_pairwise_summary(path: Path) -> Tuple[pd.Series, float, Tuple[float, float, float]]:
    df = pd.read_excel(path)
    row = df.iloc[0]
    return row, float(row["accuracy_test"]), (float(row["beta_number"]), float(row["beta_size"]), float(row["beta_spacing"]))


def load_fixed_reference_summaries(directory: Path) -> Tuple[pd.DataFrame, pd.Series]:
    records = []
    for file in sorted(directory.glob("fixed_reference_*.xlsx")):
        df = pd.read_excel(file)
        records.append(df.iloc[0])
    if not records:
        return pd.DataFrame(), pd.Series(dtype=float)
    df_all = pd.DataFrame(records)
    return df_all, df_all[["beta_number", "beta_size", "beta_spacing", "accuracy_test", "weber_fraction"]]


def load_estimation_summary(path: Path) -> pd.Series:
    df = pd.read_csv(path)
    return df.iloc[0]


def prepare_curves(
    arch_metrics: ArchitectureBehavioral,
    df_pair_feat: pd.DataFrame,
    fixed_features: Dict[str, pd.DataFrame],
    pairwise_row: pd.Series,
    fixed_df: pd.DataFrame,
) -> None:
    # Pairwise curve
    if pairwise_row is not None and not pairwise_row.empty:
        betas = (float(pairwise_row["beta_number"]), float(pairwise_row["beta_size"]), float(pairwise_row["beta_spacing"]))
        intercept = float(pairwise_row["intercept"])
        probs = _predict_probabilities_pair(df_pair_feat, intercept, betas)
        df_curve = (
            pd.DataFrame({"log_num": df_pair_feat["log_num"], "prob": probs})
            .groupby("log_num")
            .agg([np.mean, np.std])
        )
        df_curve.columns = ["mean", "std"]
        df_curve.reset_index(inplace=True)
        arch_metrics.pairwise_curve = df_curve
        arch_metrics.pairwise_betas = pairwise_row[["beta_number", "beta_size", "beta_spacing"]]

    # Fixed reference curve and betas
    if not fixed_df.empty:
        frames = []
        betas_rows = []
        for _, row in fixed_df.iterrows():
            ref = int(row["reference"])
            feat = fixed_features.get(str(ref))
            if feat is None:
                continue
            betas = (float(row["beta_number"]), float(row["beta_size"]), float(row["beta_spacing"]))
            intercept = float(row["intercept"])
            df_pred = _predict_probabilities_fixed(feat, intercept, betas)
            df_pred["reference"] = ref
            frames.append(df_pred)
            betas_rows.append(row[["reference", "beta_number", "beta_size", "beta_spacing"]])
        if frames:
            df_all = pd.concat(frames, ignore_index=True)
            df_curve = df_all.groupby("log_ratio").agg({"prob": [np.mean, np.std]}).reset_index()
            df_curve.columns = ["log_num", "mean", "std"]
            arch_metrics.fixed_curve = df_curve
        if betas_rows:
            arch_metrics.fixed_betas = pd.DataFrame(betas_rows)


def aggregate_behavioral(analysis_root: Path) -> Tuple[List[ArchitectureBehavioral], Dict[str, pd.DataFrame]]:
    arch_dirs = collect_architectures(analysis_root)

    # Preload dataset-derived features
    pairwise_arrays = _load_mat_arrays(PAIRWISE_MAT)
    with open("behavioral_datasets/binary_de_wind_test.pkl", "rb") as f:
        pair_data = pickle.load(f)
    pair_features = _compute_pairwise_features(_flatten_idxs(pair_data["idxs"]), pairwise_arrays)

    fixed_arrays = _load_mat_arrays(FIXED_MAT)
    fixed_features: Dict[str, pd.DataFrame] = {}
    for ref_file in Path("behavioral_datasets").glob("fixed_ref_REF_*_test.pkl"):
        ref = ref_file.stem.split("_")[-2]
        with open(ref_file, "rb") as f:
            dataset = pickle.load(f)
        feat = _compute_fixed_features(dataset["idxs"], int(ref), fixed_arrays)
        fixed_features[ref] = feat

    all_arch_metrics: List[ArchitectureBehavioral] = []

    for dist, dirs in arch_dirs.items():
        for arch_dir in dirs:
            arch = arch_dir.name
            behavioral = arch_dir / "behavioral"
            metrics = ArchitectureBehavioral(distribution=dist, architecture=arch)

            # Pairwise
            pairwise_file = behavioral / "comparison" / f"task_comparison_{arch}_{dist}.xlsx"
            pairwise_row = None
            if pairwise_file.exists():
                row, acc, betas = load_pairwise_summary(pairwise_file)
                metrics.pairwise = {
                    "accuracy_test": acc,
                    "weber_fraction": float(row["weber_fraction"]),
                }
                pairwise_row = row

            # Fixed reference
            fixed_dir = behavioral / "fixed_reference"
            fixed_df = pd.DataFrame()
            if fixed_dir.exists():
                fixed_df, _ = load_fixed_reference_summaries(fixed_dir)
                if not fixed_df.empty:
                    metrics.fixed_reference = fixed_df

            # Estimation
            est_file = behavioral / "estimation" / f"numerosity_estimation_{arch}_{dist}.csv"
            if est_file.exists():
                est_row = load_estimation_summary(est_file)
                metrics.estimation = {
                    "accuracy_test": float(est_row["accuracy_test"]),
                    "wmape": float(est_row["wmape"]),
                }
                classifier = str(est_row.get("classifier", "")).strip()
                if classifier:
                    conf_csv = behavioral / "estimation" / f"confusion_{arch}_{dist}_{classifier}.csv"
                    if conf_csv.exists():
                        conf_df = pd.read_csv(conf_csv, index_col=0)
                        metrics.estimation_confusion = conf_df.to_numpy(dtype=float)
                        metrics.estimation_labels = conf_df.columns.astype(int).tolist()

            # Compute curves/betas
            prepare_curves(metrics, pair_features, fixed_features, pairwise_row, fixed_df)

            all_arch_metrics.append(metrics)

    return all_arch_metrics, {"pairwise": pair_features, "fixed": fixed_features}


@dataclass
class SummaryTableRow:
    task: str
    distribution: str
    acc_mean: float
    acc_std: float
    wf_mean: float | None
    wf_std: float | None


def build_summary_table(metrics: List[ArchitectureBehavioral]) -> pd.DataFrame:
    records: List[SummaryTableRow] = []

    for dist in ["uniform", "zipfian"]:
        dist_metrics = [m for m in metrics if m.distribution == dist]

        # Pairwise
        pair_vals = [m.pairwise for m in dist_metrics if m.pairwise]
        if pair_vals:
            accs = [v["accuracy_test"] for v in pair_vals]
            wfs = [v["weber_fraction"] for v in pair_vals]
            records.append(
                SummaryTableRow(
                    task="Pairwise",
                    distribution=dist,
                    acc_mean=np.mean(accs),
                    acc_std=np.std(accs, ddof=0),
                    wf_mean=np.mean(wfs),
                    wf_std=np.std(wfs, ddof=0),
                )
            )

        # Fixed reference - average over references per architecture first
        fixed_vals = []
        if dist_metrics:
            for m in dist_metrics:
                if m.fixed_reference is None or m.fixed_reference.empty:
                    continue
                acc_mean = m.fixed_reference["accuracy_test"].mean()
                wf_mean = m.fixed_reference["weber_fraction"].mean()
                fixed_vals.append((acc_mean, wf_mean))
        if fixed_vals:
            accs = [v[0] for v in fixed_vals]
            wfs = [v[1] for v in fixed_vals]
            records.append(
                SummaryTableRow(
                    task="Fixed Ref.",
                    distribution=dist,
                    acc_mean=np.mean(accs),
                    acc_std=np.std(accs, ddof=0),
                    wf_mean=np.mean(wfs),
                    wf_std=np.std(wfs, ddof=0),
                )
            )

        # Estimation
        est_vals = [m.estimation for m in dist_metrics if m.estimation]
        if est_vals:
            accs = [v["accuracy_test"] for v in est_vals]
            records.append(
                SummaryTableRow(
                    task="Estimation",
                    distribution=dist,
                    acc_mean=np.mean(accs),
                    acc_std=np.std(accs, ddof=0),
                    wf_mean=None,
                    wf_std=None,
                )
            )

    df = pd.DataFrame([r.__dict__ for r in records])
    return df


def format_summary_table(df: pd.DataFrame) -> pd.DataFrame:
    formatted_rows = []
    for _, row in df.iterrows():
        acc = f"{row.acc_mean*100:.2f} ({row.acc_std*100:.2f})"
        if pd.notna(row.wf_mean):
            wf = f"{row.wf_mean:.2f} ({row.wf_std:.2f})"
        else:
            wf = "/"
        formatted_rows.append(
            {
                "Task": row.task,
                "Distr.": row.distribution,
                "Acc (±SD)%": acc,
                "Wf (±SD)": wf,
            }
        )
    return pd.DataFrame(formatted_rows)


def save_table_figure(df: pd.DataFrame, out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(5.8, len(df) * 0.55 + 0.8))
    ax.axis("off")
    table = ax.table(
        cellText=df.values,
        colLabels=df.columns,
        cellLoc="center",
        loc="center",
    )
    table.scale(1, 1.2)
    for (row, col), cell in table.get_celld().items():
        if row == 0:
            cell.set_facecolor("#f0f0f0")
            cell.set_fontsize(10)
        else:
            cell.set_fontsize(9)
    fig.tight_layout()
    fig.savefig(out_path, dpi=300)
    plt.close(fig)


def plot_curves(curves: Dict[str, List[pd.DataFrame]], out_path: Path, title: str) -> None:
    fig, ax = plt.subplots(figsize=(6.5, 3.6))
    colors = {"uniform": "#58a4b0", "zipfian": "#8b5fbf"}
    for dist, dist_curves in curves.items():
        if not dist_curves:
            continue
        df_concat = pd.concat(dist_curves, ignore_index=True)
        grouped = df_concat.groupby("log_num").agg({"mean": np.mean, "std": np.std}).reset_index()
        grouped.columns = ["log_num", "mean", "std"]
        ax.plot(
            grouped["log_num"],
            grouped["mean"],
            label=dist,
            color=colors.get(dist, None),
            linewidth=1.8,
            marker="o",
            markersize=4.5,
        )
        if grouped["std"].notna().any():
            ax.fill_between(
                grouped["log_num"],
                grouped["mean"] - grouped["std"],
                grouped["mean"] + grouped["std"],
                alpha=0.15,
                color=colors.get(dist, None),
            )
    ax.set_xlabel("log2 Numerosity Ratio", fontsize=12)
    ax.set_ylabel("Predicted P(choose higher)", fontsize=12)
    ax.set_title(title, fontsize=13)
    ax.tick_params(axis="both", labelsize=10)
    ax.legend(fontsize=10)
    fig.tight_layout()
    fig.savefig(out_path, dpi=300)
    plt.close(fig)


def plot_beta_boxplots(beta_data: Dict[str, pd.DataFrame], out_path: Path, title: str) -> None:
    records = []
    for dist, df in beta_data.items():
        if df is None or df.empty:
            continue
        if "reference" in df.columns:
            df_use = df.copy()
        else:
            df_use = df.reset_index(drop=True)
        df_melt = df_use.melt(
            id_vars=[col for col in df_use.columns if col in ("reference",)],
            value_vars=["beta_number", "beta_size", "beta_spacing"],
            var_name="beta",
            value_name="value",
        )
        df_melt["distribution"] = dist
        records.append(df_melt)
    if not records:
        return
    df_plot = pd.concat(records, ignore_index=True)
    fig, ax = plt.subplots(figsize=(6, 3.2))
    sns.boxplot(
        data=df_plot,
        x="beta",
        y="value",
        hue="distribution",
        palette="Set2",
        ax=ax,
        linewidth=1.0,
        fliersize=3,
    )
    ax.set_xlabel("", fontsize=11)
    ax.set_ylabel("β Coefficients", fontsize=11)
    ax.set_title(title, fontsize=13)
    ax.tick_params(axis="both", labelsize=10)
    ax.legend(fontsize=10)
    fig.tight_layout()
    fig.savefig(out_path, dpi=300)
    plt.close(fig)


def plot_estimation_accuracy_and_cov(metrics: List[ArchitectureBehavioral], out_dir: Path) -> None:
    records = []
    for m in metrics:
        if not m.estimation:
            continue
        per_class_path = (
            Path("results/analysis")
            / m.distribution
            / m.architecture
            / "behavioral"
            / "estimation"
            / f"numerosity_estimation_per_class_{m.architecture}_{m.distribution}.csv"
        )
        if not per_class_path.exists():
            continue
        df = pd.read_csv(per_class_path)
        df["architecture"] = m.architecture
        df["distribution"] = m.distribution
        records.append(df)
    if not records:
        return
    df_all = pd.concat(records, ignore_index=True)

    agg_by_dist = {}
    for dist, df_dist in df_all.groupby("distribution"):
        grouped = df_dist.groupby("numerosity").agg(
            {
                "accuracy_mean": [np.mean, np.std],
                "cov_mean": [np.mean, np.std],
            }
        ).reset_index()
        grouped.columns = [
            "numerosity",
            "acc_mean",
            "acc_std",
            "cov_mean",
            "cov_std",
        ]
        agg_by_dist[dist] = grouped

    colors = {"uniform": "#58a4b0", "zipfian": "#8b5fbf"}

    # Accuracy plot
    fig, ax = plt.subplots(figsize=(6.5, 3.6))
    for dist, grouped in agg_by_dist.items():
        ax.plot(
            grouped["numerosity"],
            grouped["acc_mean"],
            label=dist,
            color=colors.get(dist, None),
            linewidth=1.8,
            marker="o",
            markersize=4.5,
        )
        ax.fill_between(
            grouped["numerosity"],
            np.maximum(grouped["acc_mean"] - grouped["acc_std"], 0),
            np.minimum(grouped["acc_mean"] + grouped["acc_std"], 1),
            alpha=0.15,
            color=colors.get(dist, None),
        )
    ax.set_xlabel("Numerosity", fontsize=12)
    ax.set_ylabel("Accuracy", fontsize=12)
    ax.set_title("Numerosity Estimation Accuracy", fontsize=13)
    ax.tick_params(axis="both", labelsize=10)
    ax.legend(fontsize=10)
    ax.set_ylim(0.0, 1.0)
    fig.tight_layout()
    fig.savefig(out_dir / "estimation_accuracy.png", dpi=300)
    plt.close(fig)

    # Coefficient of variation plot
    fig, ax = plt.subplots(figsize=(6.5, 3.6))
    for dist, grouped in agg_by_dist.items():
        ax.plot(
            grouped["numerosity"],
            grouped["cov_mean"],
            label=dist,
            color=colors.get(dist, None),
            linewidth=1.8,
            marker="o",
            markersize=4.5,
        )
        ax.fill_between(
            grouped["numerosity"],
            np.maximum(grouped["cov_mean"] - grouped["cov_std"], 0),
            grouped["cov_mean"] + grouped["cov_std"],
            alpha=0.15,
            color=colors.get(dist, None),
        )
    ax.set_xlabel("Numerosity", fontsize=12)
    ax.set_ylabel("Coefficient of Variation", fontsize=12)
    ax.set_title("Numerosity Estimation CoV", fontsize=13)
    ax.tick_params(axis="both", labelsize=10)
    ax.legend(fontsize=10)
    ax.set_ylim(0.0, 1.0)
    fig.tight_layout()
    fig.savefig(out_dir / "estimation_cov.png", dpi=300)
    plt.close(fig)


def plot_estimation_confusion(metrics: List[ArchitectureBehavioral], out_path: Path) -> None:
    dist_mats: Dict[str, List[np.ndarray]] = {"uniform": [], "zipfian": []}
    label_ref: Dict[str, Iterable[int]] = {}

    for m in metrics:
        if m.estimation_confusion is None:
            continue
        mats = dist_mats.setdefault(m.distribution, [])
        mats.append(m.estimation_confusion)
        if m.estimation_labels is not None:
            label_ref[m.distribution] = m.estimation_labels

    if not any(dist_mats.values()):
        return

    n_plots = sum(1 for mats in dist_mats.values() if mats)
    fig, axes = plt.subplots(1, n_plots, figsize=(5.5 * n_plots, 4))
    if n_plots == 1:
        axes = [axes]

    idx_ax = 0
    for dist, mats in dist_mats.items():
        if not mats:
            continue
        mean_mat = np.mean(mats, axis=0)
        labels = label_ref.get(dist)
        if labels is None:
            labels = np.arange(1, mean_mat.shape[0] + 1)
        ax = axes[idx_ax]
        sns.heatmap(
            mean_mat * 100.0,
            xticklabels=labels,
            yticklabels=labels,
            cmap="viridis",
            vmin=0,
            vmax=100,
            ax=ax,
            cbar_kws={"label": "%"},
        )
        ax.set_xlabel("True", fontsize=11)
        ax.set_ylabel("Predicted", fontsize=11)
        ax.set_title(f"Estimation confusion ({dist})", fontsize=12)
        ax.tick_params(axis="both", labelsize=9)
        idx_ax += 1

    fig.tight_layout()
    fig.savefig(out_path, dpi=300)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="Aggregate behavioural analyses and produce report plots.")
    parser.add_argument("--analysis-root", type=Path, default=Path("results/analysis"))
    parser.add_argument("--outdir", type=Path, default=Path("results/analysis/behavioral_summary"))
    args = parser.parse_args()

    all_metrics, feature_cache = aggregate_behavioral(args.analysis_root)
    args.outdir.mkdir(parents=True, exist_ok=True)

    summary_df = build_summary_table(all_metrics)
    formatted = format_summary_table(summary_df)
    formatted.to_csv(args.outdir / "behavioral_summary_table.csv", index=False)
    save_table_figure(formatted, args.outdir / "behavioral_summary_table.png")

    # Curves and boxplots
    pairwise_curves: Dict[str, List[pd.DataFrame]] = {"uniform": [], "zipfian": []}
    fixed_curves: Dict[str, List[pd.DataFrame]] = {"uniform": [], "zipfian": []}
    pairwise_betas: Dict[str, pd.DataFrame] = {"uniform": pd.DataFrame(), "zipfian": pd.DataFrame()}
    fixed_betas: Dict[str, pd.DataFrame] = {"uniform": pd.DataFrame(), "zipfian": pd.DataFrame()}

    for m in all_metrics:
        if m.pairwise_curve is not None:
            pairwise_curves[m.distribution].append(m.pairwise_curve)
        if m.fixed_curve is not None:
            fixed_curves[m.distribution].append(m.fixed_curve)
        if m.pairwise_betas is not None:
            pairwise_betas[m.distribution] = pd.concat([
                pairwise_betas[m.distribution],
                pd.DataFrame([m.pairwise_betas]),
            ], ignore_index=True)
        if m.fixed_betas is not None:
            df_betas = m.fixed_betas[["beta_number", "beta_size", "beta_spacing"]]
            df_betas["architecture"] = m.architecture
            fixed_betas[m.distribution] = pd.concat([
                fixed_betas[m.distribution],
                df_betas,
            ], ignore_index=True)

    if any(pairwise_curves.values()):
        plot_curves(pairwise_curves, args.outdir / "pairwise_curves.png", "Pairwise Comparison")
    if any(fixed_curves.values()):
        plot_curves(fixed_curves, args.outdir / "fixed_reference_curves.png", "Fixed Reference")

    if any(df is not None and not df.empty for df in pairwise_betas.values()):
        plot_beta_boxplots(pairwise_betas, args.outdir / "pairwise_betas.png", "Pairwise β coefficients")
    if any(df is not None and not df.empty for df in fixed_betas.values()):
        plot_beta_boxplots(fixed_betas, args.outdir / "fixed_reference_betas.png", "Fixed Reference β coefficients")

    plot_estimation_accuracy_and_cov(all_metrics, args.outdir)
    plot_estimation_confusion(all_metrics, args.outdir / "estimation_confusion.png")

    print(f"Saved behavioural summary to {args.outdir}")


if __name__ == "__main__":
    import pickle  # Needed for aggregate_behavioral helper

    main()
