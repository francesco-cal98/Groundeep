from __future__ import annotations

import math
from pathlib import Path
from typing import Dict

import numpy as np
import pandas as pd
import torch
from scipy.stats import norm, t
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import accuracy_score

from CCNL_readout_DBN import forwardDBN


def _forward_concat(model, data: torch.Tensor) -> torch.Tensor:
    part1 = forwardDBN(model, data[:, :, 0:10000]).clone()
    part2 = forwardDBN(model, data[:, :, 10000:20000]).clone()
    return torch.cat((part1, part2), dim=2)


def _sgd_classifier(
    X_train: torch.Tensor,
    X_test: torch.Tensor,
    y_train: torch.Tensor,
    y_test: torch.Tensor,
):
    Xtr = X_train.view(-1, X_train.shape[2]).detach().cpu().numpy()
    Xte = X_test.view(-1, X_test.shape[2]).detach().cpu().numpy()

    Ytr = y_train.reshape(-1, y_train.shape[-1]).detach().cpu().numpy()
    Yte = y_test.reshape(-1, y_test.shape[-1]).detach().cpu().numpy()

    ytr = Ytr[:, 1].reshape(-1)
    yte = Yte[:, 1].reshape(-1)

    model = SGDClassifier(penalty="l2", max_iter=1000, random_state=42)
    model.fit(Xtr, ytr)

    pred_train = model.predict(Xtr)
    pred_test = model.predict(Xte)

    acc_train = accuracy_score(ytr, pred_train)
    acc_test = accuracy_score(yte, pred_test)

    return acc_train, pred_train, acc_test, pred_test


def _irls_fit(
    choice: np.ndarray,
    X: np.ndarray,
    guess_rate: float = 0.01,
    max_iter: int = 5000,
    tol: float = 1e-12,
):
    response_rate = 1 - guess_rate
    n_obs, n_features = X.shape
    beta = np.zeros(n_features + 1)
    X_design = np.column_stack((np.ones(n_obs), X))

    for _ in range(max_iter):
        linear = X_design @ beta
        prob = response_rate * (norm.cdf(linear) - 0.5) + 0.5
        prob = np.clip(prob, 1e-15, 1 - 1e-15)

        w = (response_rate * norm.pdf(linear)) ** 2 / prob / (1 - prob)
        z = linear + (choice - prob) / (response_rate * norm.pdf(linear))

        WX = w[:, None] * X_design
        lhs = WX.T @ X_design
        rhs = WX.T @ z

        # Piccola regolarizzazione per evitare matrici singolari
        reg = 1e-6 * np.eye(lhs.shape[0], dtype=lhs.dtype)
        lhs = lhs + reg
        try:
            beta_new = np.linalg.solve(lhs, rhs)
        except np.linalg.LinAlgError:
            beta_new = np.linalg.lstsq(lhs, rhs, rcond=None)[0]

        if np.linalg.norm(beta_new - beta) < tol:
            beta = beta_new
            break
        beta = beta_new
    else:
        print("[IRLS] Warning: max iterations reached without convergence")

    cov = np.linalg.pinv(X_design.T @ (w[:, None] * X_design))
    se = np.sqrt(np.diag(cov))
    t_stats = beta / se
    p_values = 2 * t.sf(np.abs(t_stats), df=max(n_obs - n_features - 1, 1))

    weber = 1 / (math.sqrt(2) * beta[1]) if len(beta) > 1 else np.nan
    return beta[0], beta[1:], weber, prob, t_stats, p_values, se


def _beta_extraction(choice, idxs, mat_contents, guess_rate: float = 0.01):
    idxs_np = idxs.view(-1, 2).detach().cpu().numpy().astype(int) - 1
    N_list = np.squeeze(mat_contents["N_list"])
    TSA_list = np.squeeze(mat_contents["TSA_list"])
    FA_list = np.squeeze(mat_contents["FA_list"])

    num_left, num_right = [], []
    isa_left, isa_right = [], []
    fa_left, fa_right = [], []

    for idx_left, idx_right in idxs_np:
        num_left.append(N_list[idx_left])
        num_right.append(N_list[idx_right])
        isa_left.append(TSA_list[idx_left] / N_list[idx_left])
        isa_right.append(TSA_list[idx_right] / N_list[idx_right])
        fa_left.append(FA_list[idx_left])
        fa_right.append(FA_list[idx_right])

    num_left = np.array(num_left)
    num_right = np.array(num_right)
    isa_left = np.array(isa_left)
    isa_right = np.array(isa_right)
    fa_left = np.array(fa_left)
    fa_right = np.array(fa_right)

    tsa_left = isa_left * num_left
    tsa_right = isa_right * num_right
    size_left = isa_left * tsa_left
    size_right = isa_right * tsa_right
    sparsity_left = fa_left / num_left
    sparsity_right = fa_right / num_right
    space_left = sparsity_left * fa_left
    space_right = sparsity_right * fa_right

    num_ratio = num_right / num_left
    size_ratio = size_right / size_left
    space_ratio = space_right / space_left

    X = np.column_stack((np.log2(num_ratio), np.log2(size_ratio), np.log2(space_ratio)))
    choice_np = np.asarray(choice)

    intercept, betas, weber, prob, t_stats, p_vals, se = _irls_fit(choice_np, X, guess_rate)

    return {
        "intercept": float(intercept),
        "betas": betas,
        "weber_fraction": float(weber),
        "num_ratio": num_ratio,
        "size_ratio": size_ratio,
        "space_ratio": space_ratio,
        "t_stats": t_stats,
        "p_values": p_vals,
        "standard_errors": se,
        "probabilities": prob,
    }


def run_task_comparison(
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

    acc_train, _, acc_test, pred_test = _sgd_classifier(
        X_train_comp, X_test_comp, Y_train, Y_test
    )

    beta_stats = _beta_extraction(pred_test, idx_test, mat, guess_rate)

    results = {
        "model": model_label,
        "accuracy_train": float(acc_train),
        "accuracy_test": float(acc_test),
        "weber_fraction": beta_stats["weber_fraction"],
        "intercept": beta_stats["intercept"],
        "beta_number": float(beta_stats["betas"][0]) if beta_stats["betas"].size > 0 else np.nan,
        "beta_size": float(beta_stats["betas"][1]) if beta_stats["betas"].size > 1 else np.nan,
        "beta_spacing": float(beta_stats["betas"][2]) if beta_stats["betas"].size > 2 else np.nan,
        "t_value_beta_number": float(beta_stats["t_stats"][1]) if beta_stats["t_stats"].size > 1 else np.nan,
        "p_value_beta_number": float(beta_stats["p_values"][1]) if beta_stats["p_values"].size > 1 else np.nan,
    }

    df = pd.DataFrame([results])
    df.to_excel(output_dir / f"task_comparison_{model_label}.xlsx", index=False)

    return results
