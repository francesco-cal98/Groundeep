#!/usr/bin/env python
"""
Genera 2000 stimoli con numerosità 2 usando le stesse impostazioni
di data_generation/data_no_correlation_v3.py e li aggiunge al dataset
esistente stimuli_dataset_10_10/stimuli_dataset.npz
"""

from __future__ import annotations

import math
import numpy as np
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.append(str(REPO_ROOT))


SAMPLES_PER_LEVEL = 2000
TARGET_N = 2
IMG_SIZE = 100
DATASET_DIR = Path("/home/student/Desktop/Groundeep/stimuli_dataset_10_10")
DATASET_FILE = DATASET_DIR / "stimuli_dataset.npz"
BACKUP_FILE = DATASET_DIR / "stimuli_dataset_before_level2.npz"


def _place_disk_pair() -> tuple[np.ndarray, float, float]:
    """Genera due dischi con posizione/orientamento casuale simile al dataset originale."""
    center_x = np.random.uniform(40, 60)
    center_y = np.random.uniform(40, 60)
    separation = np.random.uniform(20, 55)
    angle = np.random.uniform(0, 2 * math.pi)
    dx = separation / 2 * math.cos(angle)
    dy = separation / 2 * math.sin(angle)

    base_radius = np.random.uniform(6, 12)
    r1 = np.clip(np.random.normal(base_radius, base_radius * 0.15), 4, 16)
    r2 = np.clip(np.random.normal(base_radius, base_radius * 0.15), 4, 16)

    cx1 = center_x + dx
    cy1 = center_y + dy
    cx2 = center_x - dx
    cy2 = center_y - dy

    # ensure within bounds; if not, resample
    attempts = 0
    while attempts < 50:
        if (
            0 + r1 < cx1 < IMG_SIZE - r1
            and 0 + r1 < cy1 < IMG_SIZE - r1
            and 0 + r2 < cx2 < IMG_SIZE - r2
            and 0 + r2 < cy2 < IMG_SIZE - r2
        ):
            break
        center_x = np.random.uniform(40, 60)
        center_y = np.random.uniform(40, 60)
        separation = np.random.uniform(20, 55)
        angle = np.random.uniform(0, 2 * math.pi)
        dx = separation / 2 * math.cos(angle)
        dy = separation / 2 * math.sin(angle)
        cx1 = center_x + dx
        cy1 = center_y + dy
        cx2 = center_x - dx
        cy2 = center_y - dy
        attempts += 1

    yy, xx = np.mgrid[0:IMG_SIZE, 0:IMG_SIZE]
    img = np.zeros((IMG_SIZE, IMG_SIZE), dtype=np.uint8)
    img[((xx - cx1) ** 2 + (yy - cy1) ** 2) <= r1**2] = 255
    img[((xx - cx2) ** 2 + (yy - cy2) ** 2) <= r2**2] = 255
    return img.reshape(-1), float(r1), float(r2)


def _compute_features(flat_img: np.ndarray, r1: float, r2: float) -> tuple[float, float, float, float, float]:
    mask = flat_img.reshape(IMG_SIZE, IMG_SIZE) > 0
    cum_area = float(mask.sum())

    coords = np.column_stack(np.nonzero(mask))
    if coords.shape[0] >= 3:
        try:
            from scipy.spatial import ConvexHull

            ch = float(ConvexHull(coords).volume)
        except Exception:
            ch = cum_area
    else:
        ch = cum_area

    density = cum_area / ch if ch > 0 else 0.0
    mu = float((math.pi * (r1**2 + r2**2)) / (IMG_SIZE * IMG_SIZE))
    sd = float(mask.std())
    return cum_area, ch, density, mu, sd


def generate_level_two(n_samples: int) -> tuple[np.ndarray, np.ndarray]:
    rows = []
    images = []
    for _ in range(n_samples):
        flat, r1, r2 = _place_disk_pair()
        cum_area, ch, density, mu, sd = _compute_features(flat, r1, r2)
        rows.append([TARGET_N, cum_area, ch, density, mu, sd])
        images.append(flat)
    return np.asarray(rows, dtype=np.float32), np.asarray(images, dtype=np.uint8)


def append_to_dataset(data: np.ndarray, images: np.ndarray) -> None:
    """Aggiunge i nuovi stimoli al dataset principale."""
    if not DATASET_FILE.exists():
        raise FileNotFoundError(f"File dataset non trovato: {DATASET_FILE}")

    original = np.load(DATASET_FILE)
    D = original["D"]
    N = original["N_list"]
    CA = original["cumArea_list"]
    CH = original["CH_list"]
    DEN = original["density"]
    MU = original["mean_item_size"]
    SD = original["std_item_size"]

    # backup
    if not BACKUP_FILE.exists():
        np.savez_compressed(
            BACKUP_FILE,
            D=D,
            N_list=N,
            cumArea_list=CA,
            CH_list=CH,
            density=DEN,
            mean_item_size=MU,
            std_item_size=SD,
        )
        print(f"[Backup] salvato in {BACKUP_FILE}")

    # concatena
    new_D = np.concatenate([D, images], axis=0)
    new_N = np.concatenate([N, data[:, 0]])
    new_CA = np.concatenate([CA, data[:, 1]])
    new_CH = np.concatenate([CH, data[:, 2]])
    new_DEN = np.concatenate([DEN, data[:, 3]])
    new_MU = np.concatenate([MU, data[:, 4]])
    new_SD = np.concatenate([SD, data[:, 5]])

    np.savez_compressed(
        DATASET_FILE,
        D=new_D,
        N_list=new_N,
        cumArea_list=new_CA,
        CH_list=new_CH,
        density=new_DEN,
        mean_item_size=new_MU,
        std_item_size=new_SD,
    )
    print(f"[Merge] nuovo dataset salvato in {DATASET_FILE}")
    print(f"[Info] dimensione totale: {new_D.shape[0]} campioni")


def main():
    print(f"Generazione di {SAMPLES_PER_LEVEL} stimoli con numerosità {TARGET_N}...")
    data, imgs = generate_level_two(SAMPLES_PER_LEVEL)
    print("Generazione completata. CumArea media:", data[:, 1].mean())
    append_to_dataset(data, imgs)


if __name__ == "__main__":
    main()
