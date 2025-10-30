from __future__ import annotations

import pickle
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, MutableMapping, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, confusion_matrix

from CCNL_readout_DBN import forwardDBN
from dunja_scripts.CLs import (
    Elastic_net_regression,
    Lasso_regression,
    Linear_regression,
    Logistic_regression_multiclass,
    Poisson_regression,
    Ridge_regression,
    SGD_regression,
)

plt.switch_backend("Agg")

CLASSIFIER_REGISTRY = {
    "SGD_regression": SGD_regression,
    "Ridge_regression": Ridge_regression,
    "Linear_regression": Linear_regression,
    "Logistic_regression_multiclass": Logistic_regression_multiclass,
    "Lasso_regression": Lasso_regression,
    "Elastic_net_regression": Elastic_net_regression,
    "Poisson_regression": Poisson_regression,
}


def _to_tensor(array, device: torch.device, dtype: torch.dtype = torch.float32) -> torch.Tensor:
    if isinstance(array, torch.Tensor):
        return array.to(device=device, dtype=dtype)
    return torch.tensor(array, device=device, dtype=dtype)


def load_estimation_dataset(path: Path, device: torch.device) -> Dict[str, torch.Tensor]:
    with open(path, "rb") as f:
        dataset = pickle.load(f)
    if not isinstance(dataset, Mapping):
        raise ValueError(f"[Estimation] Il file {path} non contiene un dict compatibile.")

    data = dataset.get("data")
    labels = dataset.get("labels")
    idxs = dataset.get("idxs")

    if data is None or labels is None:
        raise ValueError(f"[Estimation] Dataset {path} privo di chiavi 'data'/'labels'.")

    data_tensor = _to_tensor(data, device=device, dtype=torch.float32)
    labels_tensor = _to_tensor(labels, device=device, dtype=torch.float32)
    idxs_tensor = _to_tensor(idxs, device=device, dtype=torch.float32) if idxs is not None else None

    return {"data": data_tensor, "labels": labels_tensor, "idxs": idxs_tensor}


def weighted_mape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    absolute_errors = np.abs(y_true - y_pred)
    weights = np.abs(y_true)
    denom = np.sum(weights) + 1e-8
    return float(np.sum(absolute_errors) / denom)


def coefficient_of_variation(values: np.ndarray) -> float:
    mean = np.mean(values)
    if np.isclose(mean, 0.0):
        return 0.0
    return float(np.std(values) / mean)


def plot_metric(
    metric_by_group: Mapping[str, Mapping[int, float]],
    std_by_group: Mapping[str, Mapping[int, float]],
    title: str,
    ylabel: str,
    ylim: Optional[tuple[float, float]],
    out_path: Path,
) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    groups = list(metric_by_group.keys())
    if not groups:
        return

    colors = plt.cm.viridis(np.linspace(0, 1, len(groups)))
    fig, ax = plt.subplots(figsize=(10, 6))
    for color, group in zip(colors, groups):
        items = sorted(metric_by_group[group].items())
        if not items:
            continue
        numerosities = np.array([int(x[0]) for x in items])
        values = np.array([x[1] for x in items])
        stds = np.array([std_by_group.get(group, {}).get(int(n), 0.0) for n in numerosities])
        ax.plot(numerosities, values, marker="o", linewidth=2.5, color=color, label=group)
        ax.fill_between(numerosities, values - stds, values + stds, color=color, alpha=0.15)

    ax.set_title(title, fontsize=18)
    ax.set_xlabel("Numerosity", fontsize=16)
    ax.set_ylabel(ylabel, fontsize=16)
    if ylim is not None:
        ax.set_ylim(*ylim)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=12)
    ax.tick_params(axis="both", labelsize=12)
    fig.tight_layout()
    fig.savefig(out_path, dpi=220)
    plt.close(fig)


def plot_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    classifier_name: str,
    model_label: str,
    out_path: Path,
    max_display_classes: int = 32,
) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    labels = np.unique(np.concatenate([y_true, y_pred]))
    labels = np.sort(labels.astype(int))

    if labels.size > max_display_classes:
        labels = labels[:max_display_classes]

    cm = confusion_matrix(y_true, y_pred, labels=labels)
    with np.errstate(divide="ignore", invalid="ignore"):
        cm_percent = cm.astype(float) / cm.sum(axis=1, keepdims=True)
        cm_percent[np.isnan(cm_percent)] = 0.0

    fig, ax = plt.subplots(figsize=(8, 7))
    im = ax.imshow(cm_percent, cmap=plt.cm.viridis, vmin=0.0, vmax=1.0, aspect="auto")
    cbar = fig.colorbar(im, ax=ax)
    cbar.ax.set_ylabel("Proportion", rotation=270, labelpad=15)

    ax.set_xticks(np.arange(len(labels)))
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=10)
    ax.set_yticks(np.arange(len(labels)))
    ax.set_yticklabels(labels, fontsize=10)
    ax.set_xlabel("Predicted", fontsize=14)
    ax.set_ylabel("True", fontsize=14)
    ax.set_title(f"{classifier_name} — {model_label}", fontsize=16)
    fig.tight_layout()
    fig.savefig(out_path, dpi=220)
    plt.close(fig)


def _flatten_tensor(tensor: torch.Tensor) -> np.ndarray:
    return tensor.detach().cpu().view(-1, tensor.shape[-1]).numpy()


def _flatten_labels(tensor: torch.Tensor) -> np.ndarray:
    arr = tensor.detach().cpu().view(-1)
    return arr.numpy()


def _build_group_key(distribution: Optional[str], classifier_name: str) -> str:
    if distribution:
        return f"{distribution}-{classifier_name}"
    return classifier_name


def run_task_numerosity_estimation(
    model,
    train_dataset: Dict[str, torch.Tensor],
    test_dataset: Dict[str, torch.Tensor],
    output_dir: Path,
    model_label: str,
    classifiers: Optional[Iterable[str]] = None,
    label_mode: str = "int",
    scale_targets: bool = False,
    max_numerosity: Optional[int] = None,
    distribution: Optional[str] = None,
    max_display_classes: int = 32,
    wandb_run=None,
) -> Dict[str, object]:
    output_dir.mkdir(parents=True, exist_ok=True)

    X_train = train_dataset["data"]
    Y_train = train_dataset["labels"]
    X_test = test_dataset["data"]
    Y_test = test_dataset["labels"]

    classifiers = list(classifiers) if classifiers else ["SGD_regression"]

    valid_classifiers = []
    for name in classifiers:
        if name not in CLASSIFIER_REGISTRY:
            print(f"[Estimation] Classifier '{name}' non riconosciuto, salto.")
            continue
        valid_classifiers.append(name)
    if not valid_classifiers:
        raise ValueError("[Estimation] Nessun classifier valido specificato.")

    X_train_repr = forwardDBN(model, X_train).clone()
    X_test_repr = forwardDBN(model, X_test).clone()

    results_rows: List[MutableMapping[str, object]] = []
    prediction_records: List[MutableMapping[str, object]] = []

    accuracy_by_group: Dict[str, Dict[int, List[float]]] = {}
    wmape_by_group: Dict[str, Dict[int, List[float]]] = {}
    cov_by_group: Dict[str, Dict[int, List[float]]] = {}

    y_test_flat = _flatten_labels(Y_test)
    max_n = int(max_numerosity or (y_test_flat.max() if y_test_flat.size else 0))

    for name in valid_classifiers:
        clf_fn = CLASSIFIER_REGISTRY[name]
        acc_tr, pred_tr_cls, _, _, acc_te, pred_te_cls, pred_te_raw, _ = clf_fn(
            X_train_repr,
            X_test_repr,
            Y_train,
            Y_test,
            labels=label_mode,
            scale=scale_targets,
            last_layer_size=X_train_repr.shape[-1],
            MAX_NUM=max_n,
        )

        y_pred = np.asarray(pred_te_cls).reshape(-1)
        y_pred_raw = np.asarray(pred_te_raw).reshape(-1)
        min_cls = int(np.min(y_test_flat)) if y_test_flat.size else 0
        max_cls = int(np.max(y_test_flat)) if y_test_flat.size else 0
        y_pred = np.clip(y_pred, min_cls, max_cls)
        y_pred_raw = np.clip(y_pred_raw, min_cls, max_cls)
        acc_train = float(acc_tr)
        acc_test = float(acc_te)
        wmape = weighted_mape(y_test_flat, y_pred)

        group_key = _build_group_key(distribution, name)
        accuracy_by_group.setdefault(group_key, {})
        wmape_by_group.setdefault(group_key, {})
        cov_by_group.setdefault(group_key, {})

        unique_classes = np.unique(y_test_flat)
        for cls in unique_classes:
            mask = y_test_flat == cls
            cls_acc = accuracy_score(y_test_flat[mask], y_pred[mask]) if np.any(mask) else np.nan
            cls_wmape = weighted_mape(y_test_flat[mask], y_pred[mask]) if np.any(mask) else np.nan
            cls_cov = coefficient_of_variation(y_pred_raw[mask]) if np.any(mask) else 0.0

            accuracy_by_group[group_key].setdefault(int(cls), []).append(cls_acc)
            wmape_by_group[group_key].setdefault(int(cls), []).append(cls_wmape)
            cov_by_group[group_key].setdefault(int(cls), []).append(cls_cov)

        results_rows.append(
            {
                "model": model_label,
                "classifier": name,
                "distribution": distribution,
                "accuracy_train": acc_train,
                "accuracy_test": acc_test,
                "wmape": wmape,
            }
        )

        prediction_records.append(
            {
                "classifier": name,
                "distribution": distribution,
                "y_true": y_test_flat.copy(),
                "y_pred": y_pred.copy(),
                "y_pred_raw": y_pred_raw.copy(),
            }
        )

        plot_confusion_matrix(
            y_test_flat,
            y_pred,
            classifier_name=name,
            model_label=model_label,
            out_path=output_dir / f"confusion_{model_label}_{name}.png",
            max_display_classes=max_display_classes,
        )

        if wandb_run is not None:
            try:
                import wandb

                wandb_run.log(
                    {
                        f"behavioral/estimation/{name}/accuracy_test": acc_test,
                        f"behavioral/estimation/{name}/wmape": wmape,
                    }
                )
                wandb_run.log(
                    {
                        f"behavioral/estimation/{name}/confusion": wandb.Image(
                            str(output_dir / f"confusion_{model_label}_{name}.png")
                        )
                    }
                )
            except Exception:
                pass

    results_df = pd.DataFrame(results_rows)
    results_path = output_dir / f"numerosity_estimation_{model_label}.csv"
    results_df.to_csv(results_path, index=False)

    per_class_rows = []
    for group_key, cls_dict in accuracy_by_group.items():
        for cls, values in cls_dict.items():
            per_class_rows.append(
                {
                    "group": group_key,
                    "classifier": group_key.split("-", maxsplit=1)[-1],
                    "distribution": distribution,
                    "numerosity": int(cls),
                    "accuracy_mean": float(np.mean(values)),
                    "accuracy_std": float(np.std(values)),
                    "wmape_mean": float(np.mean(wmape_by_group[group_key][cls])),
                    "wmape_std": float(np.std(wmape_by_group[group_key][cls])),
                    "cov_mean": float(np.mean(cov_by_group[group_key][cls])),
                    "cov_std": float(np.std(cov_by_group[group_key][cls])),
                }
            )
    per_class_df = pd.DataFrame(per_class_rows)
    per_class_path = output_dir / f"numerosity_estimation_per_class_{model_label}.csv"
    per_class_df.to_csv(per_class_path, index=False)

    avg_accuracy = {
        group: {cls: float(np.mean(vals)) for cls, vals in cls_dict.items()}
        for group, cls_dict in accuracy_by_group.items()
    }
    std_accuracy = {
        group: {cls: float(np.std(vals)) for cls, vals in cls_dict.items()}
        for group, cls_dict in accuracy_by_group.items()
    }
    avg_wmape = {
        group: {cls: float(np.mean(wmape_by_group[group][cls])) for cls in cls_dict}
        for group, cls_dict in accuracy_by_group.items()
    }
    std_wmape = {
        group: {cls: float(np.std(wmape_by_group[group][cls])) for cls in cls_dict}
        for group, cls_dict in accuracy_by_group.items()
    }
    avg_cov = {
        group: {cls: float(np.mean(cov_by_group[group][cls])) for cls in cls_dict}
        for group, cls_dict in accuracy_by_group.items()
    }
    std_cov = {
        group: {cls: float(np.std(cov_by_group[group][cls])) for cls in cls_dict}
        for group, cls_dict in accuracy_by_group.items()
    }

    plot_metric(
        avg_accuracy,
        std_accuracy,
        title=f"Accuracy vs Numerosity ({model_label})",
        ylabel="Accuracy",
        ylim=(0.0, 1.05),
        out_path=output_dir / f"accuracy_vs_numerosity_{model_label}.png",
    )
    plot_metric(
        avg_wmape,
        std_wmape,
        title=f"WMAPE vs Numerosity ({model_label})",
        ylabel="WMAPE",
        ylim=None,
        out_path=output_dir / f"wmape_vs_numerosity_{model_label}.png",
    )
    plot_metric(
        avg_cov,
        std_cov,
        title=f"Coefficient of Variation vs Numerosity ({model_label})",
        ylabel="Coefficient of Variation",
        ylim=None,
        out_path=output_dir / f"cov_vs_numerosity_{model_label}.png",
    )

    if wandb_run is not None:
        try:
            import wandb

            wandb_run.log(
                {
                    "behavioral/estimation/accuracy_curve": wandb.Image(
                        str(output_dir / f"accuracy_vs_numerosity_{model_label}.png")
                    ),
                    "behavioral/estimation/wmape_curve": wandb.Image(
                        str(output_dir / f"wmape_vs_numerosity_{model_label}.png")
                    ),
                    "behavioral/estimation/cov_curve": wandb.Image(
                        str(output_dir / f"cov_vs_numerosity_{model_label}.png")
                    ),
                }
            )
        except Exception:
            pass

    summary = {
        "results_path": results_path,
        "per_class_path": per_class_path,
        "prediction_records": prediction_records,
    }
    return summary
