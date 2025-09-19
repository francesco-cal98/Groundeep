import os
import glob
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from embedding_analysis import Embedding_analysis

# ==== Configuration ====
data_path = "/home/student/Desktop/Groundeep/stimuli_dataset_adaptive/"
data_file = "stimuli_dataset.npz"
uniform_dir = "/home/student/Desktop/Groundeep/networks/uniform/dataset_cum_area_0/"
zipfian_dir = "/home/student/Desktop/Groundeep/networks/zipfian/idbn_new_dataset2/"
output_dir = "/home/student/Desktop/Groundeep/outputs/distortion_heatmaps/cosine/"
use_ratio = False  # 🔁 Set to False to use raw distances instead of ratios

os.makedirs(output_dir, exist_ok=True)

# ==== Load all uniform models ====
uniform_models = sorted(glob.glob(os.path.join(uniform_dir, "*.pkl")))

for uniform_path in uniform_models:
    arch_name = os.path.basename(uniform_path).replace("idbn_new_dataset_idbn_trained_uniform", "").replace(".pkl", "")
    zipfian_path = os.path.join(zipfian_dir, f"idbn_new_dataset_idbn_trained_zipfian{arch_name}.pkl")

    if not os.path.exists(zipfian_path):
        print(f"❌ Skipping {arch_name}: corresponding Zipfian model not found.")
        continue

    print(f"🔍 Processing architecture: {arch_name}")

    embedder = Embedding_analysis(data_path, data_file, uniform_path, zipfian_path, arch_name)
    output_dict = embedder._get_encodings()

    Z_uniform = output_dict['Z_uniform']
    Z_zipfian = output_dict['Z_zipfian']
    labels = output_dict['labels_uniform']
    label_names = np.unique(labels)
    cum_area = output_dict['cumArea_uniform']
    convex_hull = output_dict['CH_uniform']

    #Bin the convex hull and Cum Area
    n_bins = len(label_names)
    bin_edges_cum_area = np.linspace(np.min(cum_area), np.max(cum_area), n_bins + 1)
    binned_cum_area = np.digitize(cum_area, bin_edges_cum_area, right=False)
    bin_edges_convex_hull = np.linspace(np.min(convex_hull), np.max(convex_hull), n_bins + 1)
    binned_convex_hull = np.digitize(convex_hull, bin_edges_convex_hull, right=False)

    # Setting up a config
    label = 'numerosity'  # Options: 'numerosity', 'cumArea', 'CH'

    if use_ratio:
        A_ratio = embedder.get_class_dist_matrix(Z_zipfian, Z_uniform, labels)

        fig, ax = plt.subplots(figsize=(8, 8))
        sns.heatmap(
            A_ratio, ax=ax, cbar=True, cmap='bwr', annot=False,
            xticklabels=label_names,
            yticklabels=label_names
        )
        ax.set_title(f"Distortion Ratio Matrix: {arch_name}")
        plt.xlabel("Class (Uniform)")
        plt.ylabel("Class (Zipfian)")
        plt.tight_layout()

        save_path = os.path.join(output_dir, f"{arch_name}_distortion_matrix.png")
        plt.savefig(save_path, dpi=300)
        plt.close(fig)
        print(f"✅ Saved ratio matrix for {arch_name}\n")

    else:
        if label == 'numerosity':
            A_zipfian, A_uniform = embedder.get_classwise_distance_matrices(Z_zipfian, Z_uniform, labels,metric='cosine')
        elif label == 'cumArea':
            A_zipfian, A_uniform = embedder.get_classwise_distance_matrices(Z_zipfian, Z_uniform, binned_cum_area, metric='cosine')
        elif label == 'CH':
            A_zipfian, A_uniform = embedder.get_classwise_distance_matrices(Z_zipfian, Z_uniform, binned_convex_hull, metric='cosine')


        vmin = min(A_zipfian.min(), A_uniform.min())
        vmax = max(A_zipfian.max(), A_uniform.max())

        for A, label, dist_type in zip([A_zipfian, A_uniform], ["Zipfian", "Uniform"], ["zipfian", "uniform"]):
            fig, ax = plt.subplots(figsize=(8, 6))
            sns.heatmap(
                A, ax=ax, cbar=True, cmap='viridis', annot=False,
                xticklabels=label_names,
                yticklabels=label_names,
            )
            ax.set_title(f"Distance Matrix ({label}): {arch_name}")
            plt.xlabel("Class")
            plt.ylabel("Class")
            plt.tight_layout()

            save_path = os.path.join(output_dir, f"{arch_name}_{dist_type}_distance_matrix.png")
            plt.savefig(save_path, dpi=300)
            plt.close(fig)

        print(f"✅ Saved separate distance matrices for {arch_name}\n")
