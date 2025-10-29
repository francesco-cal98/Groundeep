#!/usr/bin/env python
"""
Rigenera i campioni con numerosità 2 nel dataset stimuli_dataset_10_10/stimuli_dataset.npz.

Passi:
 1. Backup del dataset esistente (se non già presente).
 2. Rimozione di tutti i campioni con N=2 dal file principale.
 3. Generazione di 2000 nuovi campioni con i vincoli originali usando il generatore
    data_no_correlation_v3 (limitato a N=2) sfruttando il multiprocessing.
 4. Merge dei nuovi campioni nel dataset principale.

Esegui con:
    /home/student/Desktop/Groundeep/groundeep/bin/python data_generation/regenerate_level2.py
"""

from __future__ import annotations

import shutil
import sys
from multiprocessing import Pool, cpu_count
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
from tqdm import tqdm

# Aggiungi la root del repo al PYTHONPATH
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.append(str(REPO_ROOT))

import data_generation.data_no_correlation_v3 as gen


DATASET_DIR = Path("/home/student/Desktop/Groundeep/stimuli_dataset_10_10")
DATASET_FILE = DATASET_DIR / "stimuli_dataset.npz"
BACKUP_FILE = DATASET_DIR / "stimuli_dataset_before_level2_regen.npz"
TARGET_N = 2
SAMPLES_PER_LEVEL = 2000


def backup_dataset() -> None:
    if not DATASET_FILE.exists():
        raise FileNotFoundError(f"Dataset non trovato: {DATASET_FILE}")
    if BACKUP_FILE.exists():
        print(f"[Backup] Già presente: {BACKUP_FILE}")
        return
    shutil.copy2(DATASET_FILE, BACKUP_FILE)
    print(f"[Backup] Salvato in {BACKUP_FILE}")


def trim_level_two() -> Dict[str, np.ndarray]:
    data = np.load(DATASET_FILE)
    mask = data["N_list"] != TARGET_N
    removed = data["N_list"].size - np.count_nonzero(mask)
    if removed == 0:
        print("[Trim] Nessun campione N=2 da rimuovere.")
    else:
        print(f"[Trim] Rimossi {removed} campioni con N=2")

    trimmed = {
        "D": data["D"][mask],
        "N_list": data["N_list"][mask],
        "cumArea_list": data["cumArea_list"][mask],
        "CH_list": data["CH_list"][mask],
        "density": data["density"][mask],
        "mean_item_size": data["mean_item_size"][mask],
        "std_item_size": data["std_item_size"][mask],
    }
    return trimmed


def save_dataset(values: Dict[str, np.ndarray]) -> None:
    np.savez_compressed(
        DATASET_FILE,
        D=values["D"],
        N_list=values["N_list"],
        cumArea_list=values["cumArea_list"],
        CH_list=values["CH_list"],
        density=values["density"],
        mean_item_size=values["mean_item_size"],
        std_item_size=values["std_item_size"],
    )


def _run_generate_level(args: Tuple[int, int]):
    return gen.generate_level(args)


def generate_level_two() -> Dict[str, np.ndarray]:
    gen.LOG_CSV = False

    chunk = 250
    tasks: List[Tuple[int, int]] = []
    remaining = SAMPLES_PER_LEVEL
    while remaining > 0:
        take = min(chunk, remaining)
        tasks.append((TARGET_N, take))
        remaining -= take

    data_list: List[np.ndarray] = []
    img_list: List[np.ndarray] = []

    with Pool(processes=min(cpu_count(), 12)) as pool:
        for level, data_lvl, imgs_lvl in tqdm(pool.imap_unordered(_run_generate_level, tasks), total=len(tasks)):
            if level != TARGET_N:
                raise RuntimeError(f"generate_level ha restituito livello inatteso: {level}")
            data_list.append(np.asarray(data_lvl, dtype=np.float32))
            img_list.append(np.asarray(imgs_lvl, dtype=np.uint8))

    data = np.concatenate(data_list, axis=0)
    imgs = np.concatenate(img_list, axis=0)
    print(f"[Gen] Raccolti {data.shape[0]} campioni")

    result = {
        "D": imgs,
        "N_list": data[:, 0],
        "cumArea_list": data[:, 1],
        "CH_list": data[:, 2],
        "density": data[:, 3],
        "mean_item_size": data[:, 4],
        "std_item_size": data[:, 5],
    }
    return result


def merge(base: Dict[str, np.ndarray], level2: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
    merged = {}
    for key in base:
        merged[key] = np.concatenate([base[key], level2[key]], axis=0)
    return merged


def main() -> None:
    print("== Rigenerazione livello N=2 ==")
    backup_dataset()
    trimmed = trim_level_two()
    save_dataset(trimmed)
    print(f"[Trim] Dataset senza N=2: {trimmed['D'].shape[0]} campioni")

    new_level2 = generate_level_two()
    print(f"[Gen] Generati {new_level2['D'].shape[0]} nuovi campioni N=2")

    merged = merge(trimmed, new_level2)
    save_dataset(merged)
    print(f"[Done] Dataset aggiornato: {merged['D'].shape[0]} campioni totali")


if __name__ == "__main__":
    main()
