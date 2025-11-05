"""Rebuild behavioural datasets exactly as in Dunja's notebooks.

This script mirrors the dataset generation steps used in
`dunja_scripts/TASK_numerosity_estimation_exp.py` and
`dunja_scripts/TASK_fixed_reference_comparison_exp.py`, exporting the resulting
pickles into `behavioral_datasets/` for the offline analysis pipeline.
"""

from __future__ import annotations

import argparse
import pickle
import shutil
import sys
import warnings
from pathlib import Path
from typing import Dict, Optional

import numpy as np
from scipy.io import loadmat

# Ensure local modules are importable
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from dunja_scripts.datasets_utils import (
    single_stimuli_dataset_modified,
    single_stimuli_dataset_naming,
)

# Paths
SOURCE_DIR = PROJECT_ROOT / "circle_dataset_100x100"
DEST_DIR = PROJECT_ROOT / "behavioral_datasets"

TRAIN_MAT = SOURCE_DIR / "NumStim_1to32_100x100_TR.mat"
TEST_MAT = SOURCE_DIR / "NumStim_1to32_100x100_TE.mat"

# Optional: reduced numerosity range (7-28) if MAT archives exist
TRAIN_MAT_7_28 = SOURCE_DIR / "NumStim_7to28_100x100_TR.mat"
TEST_MAT_7_28 = SOURCE_DIR / "NumStim_7to28_100x100_TE.mat"

# Naming (estimation) parameters
MIN_NUM = 1
MAX_NUM = 32
SAMPLES_PER_NUM = 1500  # -> 480 batches (100 each)
LIMITS_NAMING = np.array([MIN_NUM - 1, MAX_NUM, 41])
PERCENTAGES_NAMING = np.array([0, 100, 0])
LIMIT_RADIUS = 45

# Reduced range parameters (7-28)
MIN_NUM_ALT = 7
MAX_NUM_ALT = 28
LIMITS_NAMING_ALT = np.array([MIN_NUM_ALT - 1, MAX_NUM_ALT, 41])
PERCENTAGES_NAMING_ALT = np.array([0, 100, 0])

# Fixed-reference parameters
REF_NUMS = (8, 14, 16, 20)
FIXED_LIMITS = np.array([0.49, 2.0, 4.0])
FIXED_PERC = np.array([0.0, 100.0, 0.0])
FIXED_NUM_SAMPLES = 15_200
BINARIZE_NAMING = True
BINARIZE_FIXED = True


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def move_if_exists(src: Path, dst_dir: Path, force: bool = False) -> None:
    if not src.exists():
        return
    dst = dst_dir / src.name
    ensure_dir(dst_dir)
    if dst.exists():
        if force:
            dst.unlink()
        else:
            return
    shutil.move(str(src), str(dst))
    print(f"[moved] {src} -> {dst}")


def dataset_to_numpy(
    dataset: Dict[str, "torch.Tensor"], *, binarize: bool = True
) -> Dict[str, np.ndarray]:
    """Convert torch tensors (batch, B, 10000) to numpy; optionally binarize to {0,1}."""

    def _to_numpy(t):
        if hasattr(t, "detach"):
            t = t.detach()
        if hasattr(t, "cpu"):
            t = t.cpu()
        return np.asarray(t)

    data = _to_numpy(dataset["data"])
    labels = _to_numpy(dataset["labels"])
    idxs = _to_numpy(dataset["idxs"])

    if binarize:
        data = np.where(data >= 0.5, 1, 0).astype(np.uint8)
    else:
        data = data.astype(np.float32)

    return {
        "data": data if data.dtype == np.uint8 else data.astype(np.float32),
        "labels": labels.astype(np.float32),
        "idxs": idxs.astype(np.float32),
    }


def save_pickle(path: Path, payload: Dict[str, np.ndarray], force: bool) -> None:
    ensure_dir(path.parent)
    if path.exists() and not force:
        print(f"[skip] {path} already exists (use --force to regenerate).")
        return
    with open(path, "wb") as f:
        pickle.dump(payload, f)
    print(f"[saved] {path}")


def _compute_percentages(
    unique_nums: np.ndarray,
    limits: np.ndarray,
    percentages: np.ndarray,
) -> Dict[float, float]:
    weights = {float(n): 0.0 for n in unique_nums}
    for i in range(len(limits) - 1):
        lower, upper = limits[i], limits[i + 1]
        mask = (unique_nums > lower) & (unique_nums <= upper)
        count = np.sum(mask)
        if count > 0:
            share = percentages[i + 1] / count
            for idx in np.where(mask)[0]:
                weights[float(unique_nums[idx])] = share
    return weights


def compute_naming_percentages(unique_nums: np.ndarray) -> Dict[float, float]:
    return _compute_percentages(unique_nums, LIMITS_NAMING, PERCENTAGES_NAMING)


def compute_naming_percentages_alt(unique_nums: np.ndarray) -> Dict[float, float]:
    return _compute_percentages(unique_nums, LIMITS_NAMING_ALT, PERCENTAGES_NAMING_ALT)


def compute_fixed_percentages(unique_nums: np.ndarray, ref_num: int) -> Dict[float, float]:
    weights = {float(n): 0.0 for n in unique_nums}
    ratios = unique_nums / ref_num
    for i in range(len(FIXED_LIMITS) - 1):
        lower, upper = FIXED_LIMITS[i], FIXED_LIMITS[i + 1]
        mask = (ratios > lower) & (ratios <= upper) & (unique_nums != ref_num)
        idxs = np.where(mask)[0]
        if idxs.size > 0:
            share = FIXED_PERC[i + 1] / idxs.size
            for j in idxs:
                weights[float(unique_nums[j])] = share
    weights[float(ref_num)] = 0.0
    return weights


def main(force: bool = False) -> None:
    ensure_dir(DEST_DIR)

    # Move legacy DeWind pickles (if any) into consolidated folder
    for name in ("binary_de_wind_train.pkl", "binary_de_wind_test.pkl"):
        move_if_exists(SOURCE_DIR / name, DEST_DIR, force=force)

    if not TRAIN_MAT.exists() or not TEST_MAT.exists():
        raise FileNotFoundError("Both TR and TE MAT archives are required.")

    train_mat = loadmat(TRAIN_MAT)
    test_mat = loadmat(TEST_MAT)
    unique_nums = np.unique(train_mat["N_list"])

    # 1) Numerosity naming datasets (train/test)
    naming_percentages = compute_naming_percentages(unique_nums)

    naming_train = single_stimuli_dataset_naming(
        str(TRAIN_MAT),
        num_samples=SAMPLES_PER_NUM * MAX_NUM,
        num_percentage_dict=naming_percentages,
        limit_FA=False,
        limit_radius=LIMIT_RADIUS,
    )
    naming_test = single_stimuli_dataset_naming(
        str(TEST_MAT),
        num_samples=SAMPLES_PER_NUM * MAX_NUM,
        num_percentage_dict=naming_percentages,
        limit_FA=False,
        limit_radius=LIMIT_RADIUS,
    )

    save_pickle(
        DEST_DIR / "naming_train.pkl",
        dataset_to_numpy(naming_train, binarize=BINARIZE_NAMING),
        force=force,
    )
    save_pickle(
        DEST_DIR / "naming_test.pkl",
        dataset_to_numpy(naming_test, binarize=BINARIZE_NAMING),
        force=force,
    )

    # 2) Fixed-reference datasets (train/test) for each reference numerosity
    for ref_num in REF_NUMS:
        ref_percentages = compute_fixed_percentages(unique_nums, ref_num)

        fixed_train = single_stimuli_dataset_modified(
            str(TRAIN_MAT),
            ref_num=ref_num,
            num_samples=FIXED_NUM_SAMPLES,
            num_percentage_dict=ref_percentages,
            binarize=BINARIZE_FIXED,
        )
        fixed_test = single_stimuli_dataset_modified(
            str(TEST_MAT),
            ref_num=ref_num,
            num_samples=FIXED_NUM_SAMPLES,
            num_percentage_dict=ref_percentages,
            binarize=BINARIZE_FIXED,
        )

        save_pickle(
            DEST_DIR / f"fixed_ref_REF_{ref_num}_train.pkl",
            dataset_to_numpy(fixed_train, binarize=BINARIZE_FIXED),
            force=force,
        )
        save_pickle(
            DEST_DIR / f"fixed_ref_REF_{ref_num}_test.pkl",
            dataset_to_numpy(fixed_test, binarize=BINARIZE_FIXED),
            force=force,
        )

    # Optional: datasets for numerosity range 7-28
    if TRAIN_MAT_7_28.exists() and TEST_MAT_7_28.exists():
        print("[info] Generating behavioural datasets for range 7-28.")
        train_alt_mat = loadmat(TRAIN_MAT_7_28)
        test_alt_mat = loadmat(TEST_MAT_7_28)
        unique_alt = np.unique(train_alt_mat["N_list"])

        alt_percentages = compute_naming_percentages_alt(unique_alt)
        naming_train_alt = single_stimuli_dataset_naming(
            str(TRAIN_MAT_7_28),
            num_samples=SAMPLES_PER_NUM * (MAX_NUM_ALT - MIN_NUM_ALT + 1),
            num_percentage_dict=alt_percentages,
            limit_FA=False,
            limit_radius=LIMIT_RADIUS,
        )
        naming_test_alt = single_stimuli_dataset_naming(
            str(TEST_MAT_7_28),
            num_samples=SAMPLES_PER_NUM * (MAX_NUM_ALT - MIN_NUM_ALT + 1),
            num_percentage_dict=alt_percentages,
            limit_FA=False,
            limit_radius=LIMIT_RADIUS,
        )

        save_pickle(
            DEST_DIR / "naming_train_7_28.pkl",
            dataset_to_numpy(naming_train_alt, binarize=BINARIZE_NAMING),
            force=force,
        )
        save_pickle(
            DEST_DIR / "naming_test_7_28.pkl",
            dataset_to_numpy(naming_test_alt, binarize=BINARIZE_NAMING),
            force=force,
        )

        for ref_num in REF_NUMS:
            alt_weights = compute_fixed_percentages(unique_alt, ref_num)
            fixed_train_alt = single_stimuli_dataset_modified(
                str(TRAIN_MAT_7_28),
                ref_num=ref_num,
                num_samples=FIXED_NUM_SAMPLES,
                num_percentage_dict=alt_weights,
                binarize=BINARIZE_FIXED,
            )
            fixed_test_alt = single_stimuli_dataset_modified(
                str(TEST_MAT_7_28),
                ref_num=ref_num,
                num_samples=FIXED_NUM_SAMPLES,
                num_percentage_dict=alt_weights,
                binarize=BINARIZE_FIXED,
            )
            save_pickle(
                DEST_DIR / f"fixed_ref_REF_{ref_num}_train_7_28.pkl",
                dataset_to_numpy(fixed_train_alt, binarize=BINARIZE_FIXED),
                force=force,
            )
            save_pickle(
                DEST_DIR / f"fixed_ref_REF_{ref_num}_test_7_28.pkl",
                dataset_to_numpy(fixed_test_alt, binarize=BINARIZE_FIXED),
                force=force,
            )
    else:
        warnings.warn(
            "NumStim_7to28 MAT archives not found; skipping 7-28 behavioural datasets."
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Assemble behavioural datasets (Dunja-style).")
    parser.add_argument("--force", action="store_true", help="Overwrite existing pickles.")
    args = parser.parse_args()
    main(force=args.force)
