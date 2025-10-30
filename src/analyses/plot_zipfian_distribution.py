from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import Dict

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
import yaml

from src.datasets.uniform_dataset import create_dataloaders_zipfian


def shifted_zipf_pmf(k_vals: np.ndarray, a: float = 112.27, s: float = 714.33) -> np.ndarray:
    """Return the shifted Zipf PMF used in the original training pipeline."""
    numer = (k_vals + s) ** (-a)
    denom = np.sum((np.arange(1, k_vals.max() + 1) + s) ** (-a))
    return numer / denom


def load_training_config(path: Path) -> Dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def collect_label_counts(loader) -> np.ndarray:
    counts = {}
    for batch in loader:
        _, labels = batch
        if torch.is_tensor(labels):
            labels_np = labels.detach().cpu().numpy()
        else:
            labels_np = np.asarray(labels)
        labels_flat = labels_np.reshape(-1).astype(int)
        for lbl in labels_flat:
            counts[lbl] = counts.get(lbl, 0) + 1
    max_label = max(counts.keys())
    freq = np.zeros(max_label + 1, dtype=int)
    for lbl, cnt in counts.items():
        freq[lbl] = cnt
    return freq


def plot_distribution(classes: np.ndarray, empirical: np.ndarray, theoretical: np.ndarray, out_png: Path | None) -> None:
    sns.set_theme(style="whitegrid", font_scale=1.2)
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.bar(classes, empirical, color="steelblue", alpha=0.8, label="Empirical (dataloader)")
    ax.plot(classes, theoretical, color="darkorange", marker="o", linewidth=2, label="Target Zipf PMF")
    ax.set_xlabel("Numerosity")
    ax.set_ylabel("Probability")
    ax.set_title("Zipfian distribution (training dataloader)")
    ax.set_xticks(classes)
    ax.set_ylim(0.0, max(theoretical.max(), empirical.max()) * 1.1)
    ax.legend()
    fig.tight_layout()
    if out_png is not None:
        fig.savefig(out_png, dpi=320)
        plt.close(fig)
    else:
        plt.show()


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot the Zipfian numerosity distribution from the training dataloader.")
    parser.add_argument(
        "--config",
        type=str,
        default="src/configs/training_config.yaml",
        help="Path to the training configuration YAML (default: %(default)s).",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Optional path to save the plot (png). If omitted, the figure is shown interactively.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help="Optional batch size override. When omitted the config value is used.",
    )
    args = parser.parse_args()

    cfg_path = Path(args.config)
    if not cfg_path.exists():
        raise FileNotFoundError(f"Training config not found at {cfg_path}")
    cfg = load_training_config(cfg_path)

    data_path = cfg["dataset_path"]
    data_name = cfg["dataset_name"]
    batch_size = args.batch_size or cfg.get("batch_size", 128)
    num_workers = cfg.get("num_workers", 0)

    train_loader, _, _ = create_dataloaders_zipfian(
        data_path=data_path,
        data_name=data_name,
        batch_size=batch_size,
        num_workers=num_workers,
        test_size=cfg.get("test_size", 0.2),
        val_size=cfg.get("val_size", 0.1),
        random_state=cfg.get("random_state", 42),
        multimodal_flag=cfg.get("multimodal_flag", False),
    )

    freq = collect_label_counts(train_loader)
    total = freq.sum()
    classes = np.arange(len(freq))
    mask = freq > 0
    classes = classes[mask]
    empirical = freq[mask] / total

    theoretical = shifted_zipf_pmf(classes)
    theoretical = theoretical / theoretical.sum()

    out_path = Path(args.output) if args.output else None
    plot_distribution(classes, empirical, theoretical, out_path)


if __name__ == "__main__":
    main()

