import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import MDS
from matplotlib.cm import get_cmap
from PIL import Image
from glob import glob
import torch
import pickle as pkl
import gc
import sys

# Project setup
current_dir = os.getcwd()
project_root = os.path.abspath(os.path.join(current_dir, '..', '..'))
sys.path.append(project_root)
sys.path.append(current_dir)

from src.analyses.embedding_analysis import Embedding_analysis

# === Config for each distribution ===
configs = {
    "uniform": {
        "data_path": "/home/student/Desktop/Groundeep/circle_dataset_100x100/",
        "data_file": "circle_dataset_100x100_v2.npz",
        "network_dir": "/home/student/Desktop/Groundeep/networks/uniform/idbn_new_dataset"
    },
    "zipfian": {
        "data_path": "/home/student/Desktop/Groundeep/circle_dataset_100x100/",
        "data_file": "circle_dataset_100x100_v2.npz",
        "network_dir": "/home/student/Desktop/Groundeep/networks/zipfian/idbn_new_dataset"
    }
}

# ==== Plotting helper ====
def plot_3d_embeddings(embeddings, labels, title="", colormap="viridis", save_path=None):
    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111, projection='3d')
    num_classes = len(np.unique(labels))
    cmap = get_cmap(colormap, num_classes)
    norm = plt.Normalize(vmin=min(labels), vmax=max(labels))

    sc = ax.scatter(embeddings[:, 0], embeddings[:, 1], embeddings[:, 2],
                    c=labels, cmap=cmap, norm=norm, s=40, alpha=0.85)

    cbar = fig.colorbar(sc, ax=ax, shrink=0.6, pad=0.1)
    cbar.set_label("Class Index")
    ax.set_title(title)
    ax.set_xlabel("Dim 1")
    ax.set_ylabel("Dim 2")
    ax.set_zlabel("Dim 3")

    if save_path:
        print(f"Saving plot to: {save_path}")
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close(fig)

# ==== Main Loop for Uniform and Zipfian ====
for dist_name, cfg in configs.items():
    print(f"\nProcessing distribution: {dist_name.upper()}")

    data_path = cfg["data_path"]
    data_file = cfg["data_file"]
    network_dir = cfg["network_dir"]

    output_dir = f"/home/student/Desktop/Groundeep/outputs/images_combined/3D_MDS/{dist_name}/"
    os.makedirs(output_dir, exist_ok=True)

    pkl_files = glob(os.path.join(network_dir, "*.pkl"))
    image_paths = []

    for pkl_path in pkl_files:
        arch_name = os.path.splitext(os.path.basename(pkl_path))[0]
        save_path = os.path.join(output_dir, f"{arch_name}.jpg")

        print(f" - {arch_name}")
        dummy_other = pkl_path  # Only one is used per dist in your Embedding_analysis
        analyser = Embedding_analysis(data_path, data_file, 
                                      pkl_path if dist_name == "uniform" else dummy_other,
                                      dummy_other if dist_name == "uniform" else pkl_path,
                                      arch_name)

        output_dict = analyser._get_encodings()
        embeddings = np.array(output_dict[f'Z_{dist_name}'], dtype=np.float64)
        labels = output_dict[f'labels_{dist_name}']

        mds = MDS(n_components=3, max_iter=300, n_jobs=1, dissimilarity='euclidean', random_state=42)
        emb_3d = mds.fit_transform(embeddings)

        plot_3d_embeddings(emb_3d, labels, title=arch_name, save_path=save_path)
        image_paths.append(save_path)

    # Combine individual plots
    print(f"Combining {len(image_paths)} plots for {dist_name}...")
    cols = 3
    rows = int(np.ceil(len(image_paths) / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 6, rows * 5))

    for idx, img_path in enumerate(image_paths):
        row, col = divmod(idx, cols)
        ax = axes[row, col] if rows > 1 else axes[col]
        img = Image.open(img_path)
        ax.imshow(img)
        ax.set_title(os.path.basename(img_path), fontsize=10)
        ax.axis('off')

    for idx in range(len(image_paths), rows * cols):
        row, col = divmod(idx, cols)
        ax = axes[row, col] if rows > 1 else axes[col]
        ax.axis('off')

    final_grid = os.path.join(output_dir, f"all_{dist_name}_3D_MDS.jpg")
    print(f"Saving final grid: {final_grid}")
    fig.tight_layout()
    fig.savefig(final_grid, dpi=300, bbox_inches='tight')
    plt.close(fig)

print("\n✅ All 3D MDS visualizations complete.")
