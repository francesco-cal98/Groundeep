#!/usr/bin/env python
"""Quick zipfian dataloader histogram (temporary helper)."""
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from collections import Counter

from src.datasets.uniform_dataset import create_dataloaders_zipfian

plt.rcParams.update({
    "figure.figsize": (10, 5),
    "axes.titlesize": 14,
    "axes.labelsize": 12,
})

DATA_PATH = PROJECT_ROOT / "stimuli_dataset_10_10"
DATA_FILE = "stimuli_dataset.npz"

train_loader, val_loader, test_loader = create_dataloaders_zipfian(
    data_path=str(DATA_PATH),
    data_name=DATA_FILE,
    batch_size=256,
    num_workers=0,
    test_size=0.1,
    val_size=0.1,
    random_state=42,
)

label_counter = Counter()
for _, batch_labels in train_loader:
    label_counter.update(batch_labels.long().tolist())

labels_sorted = np.array(sorted(label_counter.keys()))
counts = np.array([label_counter[int(k)] for k in labels_sorted])

fig, ax = plt.subplots(figsize=(10, 5))
ax.bar(labels_sorted.astype(int), counts, color="steelblue")
ax.set_xlabel("Numerosity")
ax.set_ylabel("Sample count")
ax.set_title("Sampled numerosity histogram (Zipfian dataloader)")
ax.set_xticks(labels_sorted.astype(int))
plt.tight_layout()

out_dir = PROJECT_ROOT / "results" / "plots"
out_dir.mkdir(parents=True, exist_ok=True)
out_path = out_dir / "zipfian_train_histogram.png"
fig.savefig(out_path, dpi=300)
print(f"[saved] {out_path}")

plt.close(fig)
