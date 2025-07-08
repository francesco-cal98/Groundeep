import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import MDS
from glob import glob
import sys
import pandas as pd
import seaborn as sns
from scipy.stats import spearmanr

# Path setup
current_dir = os.getcwd()
project_root = os.path.abspath(os.path.join(current_dir, '..', '..'))
sys.path.append(project_root)
sys.path.append(current_dir)

from src.analyses.embedding_analysis import Embedding_analysis


def plot_feature_correlation_matrix(features_dict, save_path, title="Feature Correlation Matrix"):
    df = pd.DataFrame(features_dict)
    corr_matrix = df.corr()

    plt.figure(figsize=(8, 6))
    sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap="coolwarm", square=True, cbar=True)
    plt.title(title)
    plt.tight_layout()
    
    plt.savefig(save_path, dpi=300)
    plt.close()


def plot_2d_embedding_and_correlations(emb_2d, features_dict, arch_name, save_path, method_name="2D Embedding"):
    fig, axs = plt.subplots(4, 3, figsize=(18, 16))
    axs = axs.flatten()

    numerosity = features_dict["N_list"]
    sc = axs[0].scatter(emb_2d[:, 0], emb_2d[:, 1], c=numerosity, cmap="viridis", s=40)
    axs[0].set_title(f"{method_name} – colored by Numerosity")
    axs[0].set_xlabel("Dim 1")
    axs[0].set_ylabel("Dim 2")
    plt.colorbar(sc, ax=axs[0], label="Numerosity")

    plot_idx = 1
    correlations = {}
    for feat_name, feat_values in features_dict.items():
        for i, dim_label in enumerate(["Dim 1", "Dim 2"]):
            dim = emb_2d[:, i]
            corr, _ = spearmanr(dim, feat_values)
            correlations[(feat_name, dim_label)] = corr
            axs[plot_idx].scatter(dim, feat_values, alpha=0.6)
            axs[plot_idx].set_title(f"{feat_name} vs {dim_label} (r = {corr:.2f})")
            axs[plot_idx].set_xlabel(dim_label)
            axs[plot_idx].set_ylabel(feat_name)
            plot_idx += 1

    for j in range(plot_idx, len(axs)):
        fig.delaxes(axs[j])

    fig.suptitle(f"{method_name} + Feature Correlations – {arch_name}", fontsize=18)
    fig.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(save_path, dpi=300)
    plt.close(fig)

    return correlations


def plot_all_pcs_vs_numerosity(embeddings, numerosities, arch_name, save_path, variance_threshold=0.01, max_components=20):
    pca = PCA(n_components=max_components)
    pcs = pca.fit_transform(embeddings)
    explained_variance = pca.explained_variance_ratio_

    num_components = np.sum(explained_variance >= variance_threshold)
    if num_components == 0:
        num_components = 1

    cols = 4
    rows = int(np.ceil(num_components / cols))
    fig, axs = plt.subplots(rows, cols, figsize=(5 * cols, 4 * rows), sharex=True)
    axs = axs.flatten()

    correlation_results = []

    for i in range(num_components):
        pc = pcs[:, i]
        corr, _ = spearmanr(numerosities, pc)

        axs[i].scatter(numerosities, pc, alpha=0.5)
        axs[i].set_title(f'PC{i+1} | ρ = {corr:.2f}, Var = {explained_variance[i]:.3f}')
        axs[i].set_xlabel('Numerosity')
        axs[i].set_ylabel(f'PC{i+1}')
        axs[i].grid(True)

        correlation_results.append({
            'arch': arch_name,
            'feature': 'N_list',
            'PC': f'PC{i+1}',
            'correlation': corr,
            'explained_variance': explained_variance[i]
        })

    for j in range(num_components, len(axs)):
        fig.delaxes(axs[j])

    fig.suptitle(f"PCs vs Numerosity – {arch_name}", fontsize=18)
    fig.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(save_path, dpi=300)
    plt.close()

    return correlation_results


# Configuration
configs = {
    "uniform": {
        "data_path": "/home/student/Desktop/Groundeep/circle_dataset_100x100/",
        "data_file": "circle_dataset_100x100_v2.npz",
        "network_dir": "/home/student/Desktop/Groundeep/networks/uniform/idbn_new_dataset",
        "use_mds": True,
        "output_subdir": "2D_MDS"
    },
    "zipfian": {
        "data_path": "/home/student/Desktop/Groundeep/circle_dataset_100x100/",
        "data_file": "circle_dataset_100x100_v2.npz",
        "network_dir": "/home/student/Desktop/Groundeep/networks/zipfian/idbn_new_dataset",
        "use_mds": True,
        "output_subdir": "2D_MDS"
    }
}

# Main loop
for dist_name, cfg in configs.items():
    print(f"\n📊 Processing 2D projection for distribution: {dist_name.upper()}")

    data_path = cfg["data_path"]
    data_file = cfg["data_file"]
    network_dir = cfg["network_dir"]
    use_mds = cfg.get("use_mds", False)
    method_name = "2D MDS" if use_mds else "2D PCA"
    output_dir = os.path.join(
        "/home/student/Desktop/Groundeep/outputs/images_combined/new_dataset/",
        cfg["output_subdir"], dist_name
    )
    os.makedirs(output_dir, exist_ok=True)

    pkl_files = glob(os.path.join(network_dir, "*.pkl"))
    correlations_list = []

    for pkl_path in pkl_files:
        arch_name = os.path.splitext(os.path.basename(pkl_path))[0]
        print(f" - {arch_name}")

        analyser = Embedding_analysis(
            data_path,
            data_file,
            pkl_path,
            pkl_path,
            arch_name
        )

        output_dict = analyser._get_encodings()
        embeddings = np.array(output_dict[f'Z_{dist_name}'], dtype=np.float64)

        features = {
            "N_list": np.array(output_dict[f'labels_{dist_name}']),
            "cumArea": np.array(output_dict[f'cumArea_{dist_name}']),
            "FA": np.array(output_dict[f'FA_{dist_name}']),
            "CH": np.array(output_dict[f'CH_{dist_name}'])
        }

        # Correlation matrix
        correlation_dir = os.path.join(output_dir, "correlation_matrix")
        os.makedirs(correlation_dir, exist_ok=True)
        matrix_path = os.path.join(correlation_dir, "feature_correlation_matrix.jpg")
        plot_feature_correlation_matrix(features, save_path=matrix_path)

        # 2D Embedding: PCA or MDS
        if use_mds:
            reducer = MDS(n_components=2, random_state=42, dissimilarity='euclidean', n_init=4, max_iter=300)
        else:
            reducer = PCA(n_components=2)
        emb_2d = reducer.fit_transform(embeddings)

        fig_path = os.path.join(output_dir, f"{arch_name}_{method_name.replace(' ', '_').lower()}_plot.jpg")
        correlations = plot_2d_embedding_and_correlations(emb_2d, features, arch_name, fig_path, method_name=method_name)

        for (feat_name, dim_label), corr_val in correlations.items():
            correlations_list.append({
                'arch': arch_name,
                'feature': feat_name,
                'PC': dim_label,
                'correlation': corr_val
            })

        # PCs vs numerosity (PCA only)
        if not use_mds:
            all_pcs_path = os.path.join(output_dir, f"all_pcs/{arch_name}_all_pcs_vs_numerosity.jpg")
            os.makedirs(os.path.dirname(all_pcs_path), exist_ok=True)
            pc_corrs = plot_all_pcs_vs_numerosity(
                embeddings,
                features["N_list"],
                arch_name,
                all_pcs_path,
                variance_threshold=0.03,
                max_components=20
            )
            correlations_list.extend(pc_corrs)

    # Save all correlations
    df_corr = pd.DataFrame(correlations_list)
    excel_path = os.path.join(output_dir, f"correlations_{dist_name}.xlsx")
    df_corr.to_excel(excel_path, index=False)

    print(f"✅ Finished {dist_name} — saved to: {output_dir}")

print("\n🎉 All visualizations complete.")
