from __future__ import annotations

import pickle
from pathlib import Path
from typing import Dict

import numpy as np
import pandas as pd
import torch
from scipy import io

from CCNL_readout_DBN import forwardDBN
from dunja_scripts.CLs import SGD_class_fixed, beta_extraction_ref_z


def _to_tensor(arr, device, dtype=torch.float32):
    return torch.tensor(arr, dtype=dtype, device=device)


def load_fixed_reference_inputs(
    train_pickle: Path,
    test_pickle: Path,
    mat_file: Path,
    device: torch.device,
) -> Dict[str, torch.Tensor]:
    with open(train_pickle, "rb") as f:
        train_dict = pickle.load(f)
    with open(test_pickle, "rb") as f:
        test_dict = pickle.load(f)

    X_train = _to_tensor(train_dict["data"], device=device, dtype=torch.float32)
    Y_train = _to_tensor(train_dict["labels"], device=device, dtype=torch.float32)
    idx_train = _to_tensor(train_dict["idxs"], device=device, dtype=torch.float32)

    X_test = _to_tensor(test_dict["data"], device=device, dtype=torch.float32)
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


def run_task_fixed_reference(
    model,
    inputs: Dict[str, torch.Tensor],
    output_dir: Path,
    model_label: str,
    ref_num: int,
    guess_rate: float = 0.01,
) -> Dict[str, float]:
    output_dir.mkdir(parents=True, exist_ok=True)

    X_train = inputs["X_train"]
    Y_train = inputs["Y_train"]
    X_test = inputs["X_test"]
    Y_test = inputs["Y_test"]
    idx_test = inputs["idx_test"]
    mat = inputs["mat"]

    X_train_comp = forwardDBN(model, X_train).clone()
    X_test_comp = forwardDBN(model, X_test).clone()

    acc_train, _, acc_test, pred_test = SGD_class_fixed(
        X_train_comp, X_test_comp, Y_train, Y_test
    )

    beta_results = beta_extraction_ref_z(
        pred_test,
        idx_test,
        mat["N_list"],
        mat["TSA_list"],
        mat["FA_list"],
        guessRate=guess_rate,
        ref_num=ref_num,
    )

    (
        intercept,
        betas,
        wf,
        X,
        num_ratios,
        prob_choice_higher,
        model_fit,
        numZ,
        sizeZ,
        spaceZ,
        num_list,
        t_stats,
        p_vals,
        standard_errors,
    ) = beta_results

    results = {
        "model": model_label,
        "reference": int(ref_num),
        "accuracy_train": float(acc_train),
        "accuracy_test": float(acc_test),
        "weber_fraction": float(wf),
        "intercept": float(intercept),
        "beta_number": float(betas[0]) if len(betas) > 0 else np.nan,
        "beta_size": float(betas[1]) if len(betas) > 1 else np.nan,
        "beta_spacing": float(betas[2]) if len(betas) > 2 else np.nan,
    }

    df = pd.DataFrame([results])
    df.to_excel(output_dir / f"fixed_reference_{model_label}_ref{ref_num}.xlsx", index=False)

    return results
