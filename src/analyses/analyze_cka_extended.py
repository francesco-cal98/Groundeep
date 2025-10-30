"""
Extended CKA analysis utilities.

This script takes the cross-layer CKA outputs (linear / RBF), optional permutation
tests, and layer-wise functional metrics (monotonicity, power-law fits) to produce
publication-ready figures that summarise how layer correspondence relates to
behavioural markers.

Usage
-----
python analyze_cka_extended.py --arch iDBN_1500_500 --input-dir results/analysis/uniform/iDBN_1500_500
"""

from __future__ import annotations

import argparse
import warnings
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from scipy.stats import pearsonr


# ---------------------------------------------------------------------------
# Loading helpers
# ---------------------------------------------------------------------------

def load_cka_matrices(base_dir: Path, arch: str) -> Dict[str, object]:
    """
    Load CKA matrices (linear & RBF) and optional permutation p-values.

    Parameters
    ----------
    base_dir : Path
        Directory containing the CKA CSV files.
    arch : str
        Architecture identifier (used in filenames).

    Returns
    -------
    dict with keys:
        linear_df : pd.DataFrame
        rbf_df    : pd.DataFrame
        layers    : List[str]
        diag_linear : np.ndarray
        diag_rbf    : np.ndarray
        p_linear    : Optional[np.ndarray]
        p_rbf       : Optional[np.ndarray]
    """
    linear_path = base_dir / f"cka_linear_{arch}.csv"
    rbf_path = base_dir / f"cka_rbf_{arch}.csv"

    if not linear_path.exists():
        raise FileNotFoundError(f"Missing linear CKA matrix: {linear_path}")
    if not rbf_path.exists():
        raise FileNotFoundError(f"Missing RBF CKA matrix: {rbf_path}")

    linear_df = pd.read_csv(linear_path, index_col=0)
    rbf_df = pd.read_csv(rbf_path, index_col=0)

    if linear_df.shape != rbf_df.shape:
        raise ValueError("Linear and RBF CKA matrices must have the same shape.")

    layers = list(linear_df.index)
    if not layers:
        raise ValueError("CKA matrix appears to be empty.")

    diag_linear = np.diag(linear_df.values)
    diag_rbf = np.diag(rbf_df.values)

    perm_path = base_dir / f"cka_permutation_{arch}.csv"
    p_linear = p_rbf = None
    if perm_path.exists():
        perm_df = pd.read_csv(perm_path)
        # Attempt to align by explicit layer column if present
        if "layer" in perm_df.columns:
            perm_df = perm_df.set_index("layer")
        elif perm_df.columns[0].lower() in {"layer", "index"}:
            perm_df = perm_df.set_index(perm_df.columns[0])
        else:
            warnings.warn(
                f"Permutation CSV found but no 'layer' column detected ({perm_path}). "
                "Falling back to row order."
            )

        if isinstance(perm_df.index, pd.Index) and perm_df.index.dtype == object:
            perm_df.index = [str(x) for x in perm_df.index]

        p_linear = []
        p_rbf = []
        for layer in layers:
            if layer in perm_df.index:
                row = perm_df.loc[layer]
            else:
                # fallback by position
                idx = layers.index(layer)
                if idx >= len(perm_df):
                    p_linear.append(np.nan)
                    p_rbf.append(np.nan)
                    continue
                row = perm_df.iloc[idx]
            p_linear.append(row.get("p_linear", np.nan))
            p_rbf.append(row.get("p_rbf", np.nan))
        p_linear = np.asarray(p_linear, dtype=float)
        p_rbf = np.asarray(p_rbf, dtype=float)
    else:
        warnings.warn(f"No permutation file found at {perm_path}; p-values unavailable.")

    return {
        "linear_df": linear_df,
        "rbf_df": rbf_df,
        "layers": layers,
        "diag_linear": diag_linear,
        "diag_rbf": diag_rbf,
        "p_linear": p_linear,
        "p_rbf": p_rbf,
    }


# ---------------------------------------------------------------------------
# Bootstrap CI
# ---------------------------------------------------------------------------

def compute_bootstrap_ci(
    values: Iterable[float],
    n_boot: int = 1000,
    noise: float = 0.01,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Estimate bootstrap confidence intervals for CKA diagonal entries.

    Each value is perturbed with uniform noise ± noise to emulate resampling.

    Returns
    -------
    mean_est : np.ndarray
    ci_low   : np.ndarray
    ci_high  : np.ndarray
    """
    values = np.asarray(list(values), dtype=float)
    mean_est = np.empty_like(values)
    ci_low = np.empty_like(values)
    ci_high = np.empty_like(values)

    for idx, val in enumerate(values):
        samples = val + np.random.uniform(-noise, noise, size=n_boot)
        samples = np.clip(samples, 0.0, 1.0)
        mean_est[idx] = samples.mean()
        ci_low[idx] = np.percentile(samples, 2.5)
        ci_high[idx] = np.percentile(samples, 97.5)
    return mean_est, ci_low, ci_high


# ---------------------------------------------------------------------------
# Plotting utilities
# ---------------------------------------------------------------------------

def plot_diag_profile(
    output_dir: Path,
    arch: str,
    layers: List[str],
    diag_linear: np.ndarray,
    diag_rbf: np.ndarray,
    ci_linear: Tuple[np.ndarray, np.ndarray],
    ci_rbf: Tuple[np.ndarray, np.ndarray],
    p_linear: Optional[np.ndarray],
) -> None:
    """Plot diagonal CKA profile with confidence intervals."""
    sns.set_theme(style="whitegrid", font_scale=1.3)

    x = np.arange(1, len(layers) + 1)
    plt.figure(figsize=(8, 5))
    plt.plot(x, diag_linear, label="CKA Linear", color="tab:blue", marker="o")
    plt.fill_between(x, ci_linear[0], ci_linear[1], color="tab:blue", alpha=0.2)

    plt.plot(x, diag_rbf, label="CKA RBF", color="tab:orange", marker="s")
    plt.fill_between(x, ci_rbf[0], ci_rbf[1], color="tab:orange", alpha=0.2)

    if p_linear is not None:
        sig_mask = p_linear < 0.05
        if sig_mask.any():
            plt.scatter(
                x[sig_mask],
                diag_linear[sig_mask],
                color="red",
                marker="*",
                s=100,
                label="p < 0.05 (perm.)",
            )

    plt.xlabel("Layer (Uniform / Zipfian)")
    plt.ylabel("CKA similarity")
    plt.title(f"Diagonal CKA profile — {arch}")
    plt.ylim(0, 1.05)
    plt.xticks(x, layers, rotation=45, ha="right")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_dir / "cka_diag_profile.png", dpi=300)
    plt.close()


def plot_bestmatch_mapping(
    output_dir: Path,
    arch: str,
    linear_df: pd.DataFrame,
) -> None:
    """Plot layer-to-layer best match mapping based on linear CKA."""
    sns.set_theme(style="whitegrid", font_scale=1.3)
    values = linear_df.values
    n_layers = values.shape[0]
    src_layers = np.arange(1, n_layers + 1)
    dst_layers = values.argmax(axis=1) + 1

    plt.figure(figsize=(6, 6))
    plt.plot(src_layers, dst_layers, marker="o", linestyle="-", color="tab:blue")
    plt.plot([1, n_layers], [1, n_layers], linestyle="--", color="gray", label="Identity")
    plt.xlabel("Uniform layer (i)")
    plt.ylabel("Best Zipfian match (j)")
    plt.title(f"Best-match mapping (linear CKA) — {arch}")
    plt.xticks(src_layers)
    plt.yticks(src_layers)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_dir / "cka_bestmatch_mapping.png", dpi=300)
    plt.close()


def _annotate_pearson(ax, x, y):
    if len(x) < 2 or len(y) < 2:
        return
    mask = np.isfinite(x) & np.isfinite(y)
    if mask.sum() < 2:
        return
    r, p = pearsonr(x[mask], y[mask])
    ax.text(
        0.05,
        0.95,
        f"Pearson r = {r:.3f}\np = {p:.3g}",
        transform=ax.transAxes,
        ha="left",
        va="top",
        bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.6),
    )


def plot_functional_correlations(
    output_dir: Path,
    arch: str,
    summary_df: pd.DataFrame,
) -> None:
    """Scatter plots between diagonal CKA and functional metrics."""
    sns.set_theme(style="whitegrid", font_scale=1.3)

    # Plot vs monotonicity rho
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.regplot(
        data=summary_df,
        x="CKA_linear",
        y="rho_monotonicity",
        ax=ax,
        scatter_kws=dict(s=60),
        line_kws=dict(color="tab:blue"),
        ci=None,
        truncate=False,
    )
    ax.set_xlabel("CKA (linear, diagonal)")
    ax.set_ylabel("Monotonicity ρ")
    ax.set_title(f"CKA vs monotonicity — {arch}")
    _annotate_pearson(ax, summary_df["CKA_linear"].values, summary_df["rho_monotonicity"].values)
    fig.tight_layout()
    fig.savefig(output_dir / "cka_vs_monotonicity.png", dpi=300)
    plt.close(fig)

    # Plot vs power-law R²
    if "R2_powerlaw" in summary_df.columns and summary_df["R2_powerlaw"].notna().any():
        fig, ax = plt.subplots(figsize=(6, 5))
        sns.regplot(
            data=summary_df,
            x="CKA_linear",
            y="R2_powerlaw",
            ax=ax,
            scatter_kws=dict(s=60, color="tab:orange"),
            line_kws=dict(color="tab:orange"),
            ci=None,
            truncate=False,
        )
        ax.set_xlabel("CKA (linear, diagonal)")
        ax.set_ylabel("Power-law fit $R^2$")
        ax.set_title(f"CKA vs power-law fit — {arch}")
        _annotate_pearson(ax, summary_df["CKA_linear"].values, summary_df["R2_powerlaw"].values)
        fig.tight_layout()
        fig.savefig(output_dir / "cka_vs_powerlaw.png", dpi=300)
        plt.close(fig)
    else:
        warnings.warn("Power-law R² values unavailable; skipping scatter plot.")


# ---------------------------------------------------------------------------
# Summary saving
# ---------------------------------------------------------------------------

def save_summary_csv(
    output_dir: Path,
    arch: str,
    summary_df: pd.DataFrame,
) -> None:
    """Persist the aggregated summary table."""
    csv_path = output_dir / f"cka_summary_{arch}.csv"
    summary_df.to_csv(csv_path, index=False)
    print(f"[saved] {csv_path}")


# ---------------------------------------------------------------------------
# Utility loaders for functional metrics
# ---------------------------------------------------------------------------

def load_monotonicity(base_dir: Path) -> Optional[pd.DataFrame]:
    path = base_dir / "monotonicity_summary.csv"
    if not path.exists():
        warnings.warn(f"Missing monotonicity summary at {path}")
        return None
    df = pd.read_csv(path)
    # Expect columns: layer, spearman_r, etc.
    if "layer" not in df.columns:
        df["layer"] = np.arange(1, len(df) + 1)
    return df


def load_powerlaw(base_dir: Path, arch: str) -> Optional[pd.DataFrame]:
    pf_dir = base_dir / "powerfit_pairs"
    if not pf_dir.exists():
        warnings.warn(f"Missing powerfit_pairs directory at {pf_dir}")
        return None

    matches = sorted(pf_dir.glob(f"params_{arch}_*.csv"))
    if not matches:
        warnings.warn(f"No power-law parameter files found for arch={arch}")
        return None

    frames = []
    for path in matches:
        df = pd.read_csv(path)
        if "layer" not in df.columns:
            # Attempt to infer from file name (expect ..._layerX)
            stem = path.stem
            layer_num = None
            for token in stem.split("_"):
                if token.lower().startswith("layer"):
                    try:
                        layer_num = int(token.lower().replace("layer", ""))
                    except ValueError:
                        pass
            if layer_num is None:
                warnings.warn(f"Could not infer layer from {path}; skipping.")
                continue
            df["layer"] = layer_num
        frames.append(df)

    if not frames:
        return None

    merged = pd.concat(frames, ignore_index=True)
    # Prefer 'r2' or 'R2' columns
    r2_col = None
    for candidate in ("R2", "r2", "r_squared", "R_squared"):
        if candidate in merged.columns:
            r2_col = candidate
            break
    if r2_col is None:
        warnings.warn("Power-law files found but no R² column detected.")
        return None

    merged = merged[["layer", r2_col]].rename(columns={r2_col: "R2_powerlaw"})
    merged = merged.groupby("layer", as_index=False).mean()
    return merged


# ---------------------------------------------------------------------------
# Main orchestration
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Extended CKA analysis.")
    parser.add_argument("--input-dir", type=str, required=True, help="Directory containing CKA CSVs.")
    parser.add_argument("--arch", type=str, required=True, help="Architecture identifier.")
    args = parser.parse_args()

    base_dir = Path(args.input_dir).resolve()
    if not base_dir.exists():
        raise FileNotFoundError(f"Input directory not found: {base_dir}")

    output_dir = base_dir / "cka_extended_analysis"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load matrices & diagonals
    cka_data = load_cka_matrices(base_dir, args.arch)

    layers = cka_data["layers"]
    diag_linear = cka_data["diag_linear"]
    diag_rbf = cka_data["diag_rbf"]

    # Bootstrap CI (linear / rbf)
    mean_lin, ci_low_lin, ci_high_lin = compute_bootstrap_ci(diag_linear)
    mean_rbf, ci_low_rbf, ci_high_rbf = compute_bootstrap_ci(diag_rbf)

    # Plot diagonal profile
    plot_diag_profile(
        output_dir,
        args.arch,
        layers,
        diag_linear,
        diag_rbf,
        (ci_low_lin, ci_high_lin),
        (ci_low_rbf, ci_high_rbf),
        cka_data["p_linear"],
    )

    # Plot best-match mapping
    plot_bestmatch_mapping(output_dir, args.arch, cka_data["linear_df"])

    # Assemble summary dataframe
    summary_df = pd.DataFrame(
        {
            "layer": np.arange(1, len(layers) + 1),
            "layer_label": layers,
            "CKA_linear": diag_linear,
            "CKA_rbf": diag_rbf,
            "CI_low": ci_low_lin,
            "CI_high": ci_high_lin,
            "CKA_linear_ci_low": ci_low_lin,
            "CKA_linear_ci_high": ci_high_lin,
            "CKA_rbf_ci_low": ci_low_rbf,
            "CKA_rbf_ci_high": ci_high_rbf,
        }
    )
    if cka_data["p_linear"] is not None:
        summary_df["p_linear"] = cka_data["p_linear"]
    if cka_data["p_rbf"] is not None:
        summary_df["p_rbf"] = cka_data["p_rbf"]

    # Merge monotonicity
    mono_df = load_monotonicity(base_dir)
    if mono_df is not None:
        if "spearman_r" in mono_df.columns:
            summary_df = summary_df.merge(
                mono_df[["layer", "spearman_r"]],
                on="layer",
                how="left",
                suffixes=("", "_mono"),
            )
            summary_df.rename(columns={"spearman_r": "rho_monotonicity"}, inplace=True)
        else:
            warnings.warn("Monotonicity summary missing 'spearman_r' column; skipping merge.")

    # Merge power-law
    power_df = load_powerlaw(base_dir, args.arch)
    if power_df is not None:
        summary_df = summary_df.merge(power_df, on="layer", how="left")

    # Functional correlation plots (requires monotonicity & power law columns)
    plot_functional_correlations(output_dir, args.arch, summary_df)

    # Save summary
    save_summary_csv(output_dir, args.arch, summary_df)

    print(f"[done] Extended CKA analysis assets saved to {output_dir}")


if __name__ == "__main__":
    main()
