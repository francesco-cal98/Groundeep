import os
import glob
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from embedding_analysis import Embedding_analysis

data_path = "/home/student/Desktop/Groundeep/training_tensors/uniform/"
data_file = "NumStim_1to40_100x100_TR_uniform.npz"
uniform_dir = "/home/student/Desktop/Groundeep/networks/uniform/idbn/"
zipfian_dir = "/home/student/Desktop/Groundeep/networks/zipfian/"
output_dir = "/home/student/Desktop/Groundeep/outputs/distortion_heatmaps"

os.makedirs(output_dir, exist_ok=True)

uniform_models = sorted(glob.glob(os.path.join(uniform_dir, "*.pkl")))

for uniform_path in uniform_models:
    arch_name = os.path.basename(uniform_path).replace("idbn_trained_uniform_", "").replace(".pkl", "")
    zipfian_path = os.path.join(zipfian_dir, f"idbn_trained_zipfian_{arch_name}.pkl")

    if not os.path.exists(zipfian_path):
        print(f"Skipping {arch_name}: corresponding Zipfian model not found.")
        continue

    print(f"Processing architecture: {arch_name}")

    embedder = Embedding_analysis(data_path, data_file, uniform_path, zipfian_path, arch_name)
    output_dict = embedder._get_encodings()
    
    Z_uniform = output_dict['Z_uniform']
    Z_zipfian = output_dict['Z_zipfian']
    labels = output_dict['labels_uniform']
    
    A = embedder.get_class_dist_matrix(Z_zipfian, Z_uniform, labels)

    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(
        A, ax=ax, cbar=True, cmap='bwr', annot=False,
        xticklabels=np.unique(labels),
        yticklabels=np.unique(labels)
    )
    ax.set_title(f"Distortion Matrix: {arch_name}")
    plt.xlabel("Class (Uniform)")
    plt.ylabel("Class (Zipfian)")
    plt.tight_layout()

    save_path = os.path.join(output_dir, f"{arch_name}_distortion_matrix.png")
    plt.savefig(save_path, dpi=300)
    plt.close(fig)
