from __future__ import annotations

import os
import pickle
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pandas as pd
import torch
from scipy import io

from CCNL_readout_DBN import forwardDBN, classifier, beta_extraction


def _to_tensor(arr, device, dtype=torch.float32):
    return torch.tensor(arr, dtype=dtype, device=device)


def load_behavioral_inputs(
    train_pickle: Path,
    test_pickle: Path,
    mat_file: Path,
    device: torch.device,
) -> Dict[str, torch.Tensor]:
    with open(train_pickle, "rb") as f:
        train_dict = pickle.load(f)
    with open(test_pickle, "rb") as f:
        test_dict = pickle.load(f)

    X_train = _to_tensor(train_dict["data"], device=device, dtype=torch.float32) / 255.0
    Y_train = _to_tensor(train_dict["labels"], device=device, dtype=torch.float32)
    idx_train = _to_tensor(train_dict["idxs"], device=device, dtype=torch.float32)

    X_test = _to_tensor(test_dict["data"], device=device, dtype=torch.float32) / 255.0
    Y_test = _to_tensor(test_dict["labels"], device=device, dtype=torch.float32)
    idx_test = _to_tensor(test_dict["idxs"], device=device, dtype=torch.float32)

    mat_contents = io.loadmat(mat_file)

    return {
        "X_train": X_train,
        "Y_train": Y_train,
        "idx_train": idx_train,
        "X_test": X_test,
        "Y_test": Y_test,
        "idx_test": idx_test,
        "mat": mat_contents,
    }


def _forward_concat(model, X):
    part1 = forwardDBN(model, X[:, :, 0:10000]).clone()
    part2 = forwardDBN(model, X[:, :, 10000:20000]).clone()
    return torch.cat((part1, part2), dim=2)


def run_behavioral_analysis(
    model,
    inputs: Dict[str, torch.Tensor],
    output_dir: Path,
    model_label: str,
    guess_rate: float = 0.01,
) -> Dict[str, float]:
    output_dir.mkdir(parents=True, exist_ok=True)

    X_train = inputs["X_train"]
    Y_train = inputs["Y_train"]
    X_test = inputs["X_test"]
    Y_test = inputs["Y_test"]
    idx_test = inputs["idx_test"]
    mat = inputs["mat"]

    X_train_comp = _forward_concat(model, X_train)
    X_test_comp = _forward_concat(model, X_test)

    acc_train, pred_train, acc_test, pred_test, weights = classifier(
        X_train_comp, X_test_comp, Y_train, Y_test
    )

    weights_array = np.asarray(weights)
    np.save(output_dir / f"{model_label}_weights.npy", weights_array)

    choice_tensor = torch.tensor(pred_test, device=idx_test.device)
    model_fit, betas, weber, _ = beta_extraction(
        choice_tensor.cpu(),
        idx_test.cpu(),
        mat["N_list"],
        mat["TSA_list"],
        mat["FA_list"],
        guess_rate,
    )

    results = {
        "model": model_label,
        "accuracy_train": float(acc_train),
        "accuracy_test": float(acc_test),
        "beta_number": float(betas[0]) if len(betas) > 0 else np.nan,
        "beta_size": float(betas[1]) if len(betas) > 1 else np.nan,
        "beta_spacing": float(betas[2]) if len(betas) > 2 else np.nan,
        "weber_fraction": float(weber),
        "intercept": float(model_fit),
    }

    df = pd.DataFrame([results])
    df.to_excel(output_dir / f"behavioral_results_{model_label}.xlsx", index=False)

    return results
