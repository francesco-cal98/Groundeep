#!/usr/bin/env python
"""Rigenera rapidamente 2000 stimoli con numerosity 2 e li inserisce nel dataset."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Dict

import numpy as np
from scipy.spatial import ConvexHull


DATASET_DIR = Path("/home/student/Desktop/Groundeep/stimuli_dataset_10_10")
DATASET_FILE = DATASET_DIR / "stimuli_dataset.npz"
BACKUP_FILE = DATASET_DIR / "stimuli_dataset_before_level2_fast.npz"

TARGET_N = 2
NUM_SAMPLES = 2000
IMG_SIZE = 100
TARGET_AREA = 900.0  # area media approximate
STD_RADIUS = 0.15    # varianza della dimensione dei dots


def backup_dataset() -> None:
    if not DATASET_FILE.exists():
        raise FileNotFoundError(f"Dataset non trovato: {DATASET_FILE}")
    if not BACKUP_FILE.exists():
        np.savez_compressed(
            BACKUP_FILE,
            **np.load(DATASET_FILE)
        )
        print(f"[Backup] salvato in {BACKUP_FILE}")
    else:
        print(f"[Backup] già presente: {BACKUP_FILE}")


def trim_level_two(data: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
    mask = data["N_list"] != TARGET_N
    removed = data["N_list"].size - np.count_nonzero(mask)
    print(f"[Trim] rimossi {removed} campioni N=2")
    return {k: v[mask] for k, v in data.items()}


def sample_pair() -> tuple[np.ndarray, tuple[float, float, float], tuple[float, float, float]]:
    center_x = np.random.uniform(35, 65)
    center_y = np.random.uniform(35, 65)
    separation = np.random.uniform(18, 55)
    angle = np.random.uniform(0, 2 * math.pi)
    dx = separation * 0.5 * math.cos(angle)
    dy = separation * 0.5 * math.sin(angle)

    base_radius = np.random.uniform(6, 12)
    r1 = np.clip(np.random.normal(base_radius, base_radius * STD_RADIUS), 4, 16)
    r2 = np.clip(np.random.normal(base_radius, base_radius * STD_RADIUS), 4, 16)

    cx1, cy1 = center_x + dx, center_y + dy
    cx2, cy2 = center_x - dx, center_y - dy

    # ensure bounds
    attempts = 0
    while attempts < 30:
        if (r1 < cx1 < IMG_SIZE - r1 and r1 < cy1 < IMG_SIZE - r1 and
                r2 < cx2 < IMG_SIZE - r2 and r2 < cy2 < IMG_SIZE - r2):
            break
        center_x = np.random.uniform(35, 65)
        center_y = np.random.uniform(35, 65)
        separation = np.random.uniform(18, 55)
        angle = np.random.uniform(0, 2 * math.pi)
        dx = separation * 0.5 * math.cos(angle)
        dy = separation * 0.5 * math.sin(angle)
        cx1, cy1 = center_x + dx, center_y + dy
        cx2, cy2 = center_x - dx, center_y - dy
        attempts += 1

    yy, xx = np.mgrid[0:IMG_SIZE, 0:IMG_SIZE]
    mask = np.zeros((IMG_SIZE, IMG_SIZE), dtype=np.uint8)
    mask[((xx - cx1) ** 2 + (yy - cy1) ** 2) <= r1**2] = 255
    mask[((xx - cx2) ** 2 + (yy - cy2) ** 2) <= r2**2] = 255

    return mask.reshape(-1), (cx1, cy1, r1), (cx2, cy2, r2)


def adjust_area(img_flat: np.ndarray, r1: float, r2: float) -> tuple[np.ndarray, float, float]:
    area_now = math.pi * (r1**2 + r2**2)
    scale = math.sqrt(TARGET_AREA / max(area_now, 1e-6))
    r1 *= scale
    r2 *= scale

    mask = np.zeros(IMG_SIZE * IMG_SIZE, dtype=np.uint8)
    yy, xx = np.mgrid[0:IMG_SIZE, 0:IMG_SIZE]
    mask[((xx - 50 - r1) ** 2 + (yy - 50) ** 2) <= r1**2] = 255
    mask[((xx - 50 + r1) ** 2 + (yy - 50) ** 2) <= r2**2] = 255
    return mask, r1, r2


def compute_features(flat_img: np.ndarray) -> tuple[float, float, float, float, float]:
    mask = flat_img.reshape(IMG_SIZE, IMG_SIZE) > 0
    cum_area = float(mask.sum())
    coords = np.column_stack(np.nonzero(mask))
    if coords.shape[0] >= 3:
        try:
            ch = float(ConvexHull(coords).volume)
        except Exception:
            ch = cum_area
    else:
        ch = cum_area
    density = cum_area / ch if ch > 0 else 1.0
    mu = float(cum_area / (IMG_SIZE * IMG_SIZE))
    sd = float(mask.std())
    return cum_area, ch, density, mu, sd


def generate_level_two() -> Dict[str, np.ndarray]:
    rows = []
    images = []
    for _ in range(NUM_SAMPLES):
        flat, (cx1, cy1, r1), (cx2, cy2, r2) = sample_pair()
        cum_area, ch, density, mu, sd = compute_features(flat)
        rows.append([TARGET_N, cum_area, ch, density, mu, sd])
        images.append(flat)
    data = np.asarray(rows, dtype=np.float32)
    imgs = np.asarray(images, dtype=np.uint8)
    return {
        "D": imgs,
        "N_list": data[:, 0],
        "cumArea_list": data[:, 1],
        "CH_list": data[:, 2],
        "density": data[:, 3],
        "mean_item_size": data[:, 4],
        "std_item_size": data[:, 5],
    }


def merge(base: Dict[str, np.ndarray], level2: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
    merged = {}
    for key in base:
        merged[key] = np.concatenate([base[key], level2[key]], axis=0)
    return merged


def main():
    backup_dataset()
    original = np.load(DATASET_FILE)
    trimmed = trim_level_two({
        "D": original["D"],
        "N_list": original["N_list"],
        "cumArea_list": original["cumArea_list"],
        "CH_list": original["CH_list"],
        "density": original["density"],
        "mean_item_size": original["mean_item_size"],
        "std_item_size": original["std_item_size"],
    })
    np.savez_compressed(DATASET_FILE, **trimmed)
    print(f"[Trim] dataset senza N=2: {trimmed['D'].shape[0]} campioni")

    level2 = generate_level_two()
    merged = merge(trimmed, level2)
    np.savez_compressed(DATASET_FILE, **merged)
    print(f"[Done] dataset aggiornato: {merged['D'].shape[0]} campioni totali")


if __name__ == "__main__":
    main()
