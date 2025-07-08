import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from glob import glob
from scipy.spatial.distance import pdist, squareform
from scipy.stats import spearmanr, ttest_rel
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression
import umap
import sys

# === PATH SETUP ===
current_dir = os.getcwd()
project_root = os.path.abspath(os.path.join(current_dir, '..', '..'))
sys.path.append(project_root)
sys.path.append(current_dir)

from src.analyses.embedding_analysis import Embedding_analysis

# === RSA UTILITIES ===
def compute_brain_rdm(embeddings, metric="cosine"):
    if metric == "cosine":
        normed = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)
        return pdist(normed, metric='cosine')
    elif metric == "euclidean":
        return pdist(embeddings, metric='euclidean')
    else:
        raise ValueError("Unsupported distance metric")

def compute_model_rdm(values, metric='euclidean'):
    x = values.astype(np.float64)
    return pdist(x.reshape(-1, 1), metric=metric)

def correlation_test(brain_rdm, model_rdm1, model_rdm2):
    rho1, _ = spearmanr(brain_rdm, model_rdm1)
    rho2, _ = spearmanr(brain_rdm, model_rdm2)
    diff = rho1 - rho2
    z = (diff) / np.sqrt((1 - rho1 ** 2) / (len(brain_rdm) - 2) + (1 - rho2 ** 2) / (len(brain_rdm) - 2))
    return rho1, rho2, z

def plot_featurewise_projection_all_architectures_grid(embeddings_dict, features_dict, method_name, output_dir, dist_name):
    reducer = umap.UMAP(n_components=2, random_state=42) if method_name == "UMAP" else PCA(n_components=2)
    for feat_name in features_dict:
        n_archs = len(embeddings_dict)
        n_cols = 4
        n_rows = int(np.ceil(n_archs / n_cols))

        fig, axs = plt.subplots(n_rows, n_cols, figsize=(6 * n_cols, 5 * n_rows))
        axs = axs.flatten()

        for i, (arch_id, emb) in enumerate(embeddings_dict.items()):
            emb_2d = reducer.fit_transform(emb)
            if feat_name == "log":
                color_values = features_dict["log"][arch_id]
                tick_labels = features_dict["numerosity"][arch_id]
            else:
                color_values = features_dict[feat_name][arch_id]
                tick_labels = color_values

            sc = axs[i].scatter(emb_2d[:, 0], emb_2d[:, 1], c=color_values, cmap='viridis', s=40)
            axs[i].set_title(f"{arch_id}\nFeature: {feat_name}")
            axs[i].set_xlabel(f"{method_name}-1")
            axs[i].set_ylabel(f"{method_name}-2")
            cbar = fig.colorbar(sc, ax=axs[i], label=feat_name)
            if feat_name == "log":
                cbar.set_ticks(np.linspace(min(color_values), max(color_values), 5))
                cbar.set_ticklabels([str(int(v)) for v in np.linspace(min(tick_labels), max(tick_labels), 5)])

        for j in range(i + 1, len(axs)):
            axs[j].axis('off')

        plt.suptitle(f"{method_name} – {dist_name} – Feature: {feat_name}")
        plt.tight_layout()
        fname = f"{method_name.lower()}_plot_{dist_name}_{feat_name}_400.jpg"
        plt.savefig(os.path.join(output_dir, fname), dpi=300)
        plt.close()

def plot_boxplot_rsa_correlations(df, output_path):
    plt.figure(figsize=(10, 6))
    sns.boxplot(data=df, x="encoding", y="spearman_rho", hue="distribution")
    plt.title("Distribuzione delle correlazioni RSA per encoding e distribuzione")
    plt.xlabel("Encoding")
    plt.ylabel("Spearman Rho")
    plt.legend(title="Distribuzione")
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()

# === CONFIG ===
configs = {
    "uniform": {
        "data_path": "/home/student/Desktop/Groundeep/circle_dataset_100x100/",
        "data_file": "circle_dataset_100x100_v2.npz",
        "network_dir": "/home/student/Desktop/Groundeep/networks/uniform/idbn_new_dataset"
    },
    "zipfian": {
        "data_path": "/home/student/Desktop/Groundeep/circle_dataset_100x100/",
        "data_file": "circle_dataset_100x100_v2.npz",
        "network_dir": "/home/student/Desktop/Groundeep/networks/zipfian/idbn_400_runs"
    }
}

output_base = "/home/student/Desktop/Groundeep/outputs/rsa_analysis"
os.makedirs(output_base, exist_ok=True)

regression_results = []
rsa_results = []
corrtest_results = []

for dist_name, cfg in configs.items():
    data_path = cfg["data_path"]
    data_file = cfg["data_file"]
    network_dir = cfg["network_dir"]
    pkl_files = glob(os.path.join(network_dir, "*.pkl"))

    dist_out_dir = os.path.join(output_base, dist_name)
    os.makedirs(dist_out_dir, exist_ok=True)

    embeddings_dict = {}
    features_by_arch = {"linear": {}, "log": {}, "sqrt": {}, "numerosity": {}, "cumArea": {}, "FA": {}, "CH": {}}

    for pkl_path in pkl_files:
        arch_name = os.path.splitext(os.path.basename(pkl_path))[0]
        analyser = Embedding_analysis(data_path, data_file, pkl_path, pkl_path, arch_name)
        output_dict = analyser._get_encodings()

        embeddings = np.array(output_dict[f'Z_{dist_name}'], dtype=np.float64)
        numerosities = np.array(output_dict[f'labels_{dist_name}'])
        cumArea = np.array(output_dict[f'cumArea_{dist_name}'])
        FA = np.array(output_dict[f'FA_{dist_name}'])
        CH = np.array(output_dict[f'CH_{dist_name}'])

        embeddings_dict[arch_name] = embeddings
        features_by_arch["linear"][arch_name] = numerosities
        features_by_arch["log"][arch_name] = np.log1p(numerosities)
        features_by_arch["sqrt"][arch_name] = np.sqrt(numerosities)
        features_by_arch["numerosity"][arch_name] = numerosities
        features_by_arch["cumArea"][arch_name] = cumArea
        features_by_arch["FA"][arch_name] = FA
        features_by_arch["CH"][arch_name] = CH

        brain_rdm = compute_brain_rdm(embeddings)

        features = {
            "linear": numerosities,
            "log": np.log1p(numerosities),
            "sqrt": np.sqrt(numerosities),
            "numerosity": numerosities,
            "cumArea": cumArea,
            "FA": FA,
            "CH": CH
        }

        model_rdms = {k: compute_model_rdm(v) for k, v in features.items()}

        for name, rdm in model_rdms.items():
            rho, _ = spearmanr(brain_rdm, rdm)
            rsa_results.append({
                "arch": arch_name,
                "distribution": dist_name,
                "encoding": name,
                "spearman_rho": rho
            })

        for other in ["linear", "sqrt"]:
            rho_log, rho_other, z = correlation_test(brain_rdm, model_rdms["log"], model_rdms[other])
            corrtest_results.append({
                "arch": arch_name,
                "distribution": dist_name,
                "comparison": f"log_vs_{other}",
                "rho_log": rho_log,
                "rho_other": rho_other,
                "z_score": z
            })

        model_names = ["log", "cumArea", "FA", "CH"]
        model_rdms_list = [model_rdms[n] for n in model_names]
        X = np.stack(model_rdms_list, axis=1)
        y = brain_rdm
        reg = LinearRegression().fit(X, y)

        for name, coef in zip(model_names, reg.coef_):
            regression_results.append({
                "arch": arch_name,
                "distribution": dist_name,
                "feature": name,
                "regression_coef": coef
            })

    for method in ["UMAP", "PCA"]:
        plot_featurewise_projection_all_architectures_grid(
            embeddings_dict,
            features_by_arch,
            method,
            dist_out_dir,
            dist_name
        )

# === Salva risultati ===
df_reg = pd.DataFrame(regression_results)
df_reg.to_excel(os.path.join(output_base, "rsa_regression_feature_importance.xlsx"), index=False)
print("\n📄 Regressione multipla RSA salvata!")

df_rsa = pd.DataFrame(rsa_results)
df_rsa.to_excel(os.path.join(output_base, "rsa_featurewise_rhos.xlsx"), index=False)
print("📈 Correlazioni RSA salvate!")

plot_boxplot_rsa_correlations(df_rsa, os.path.join(output_base, "boxplot_rsa_correlations.jpg"))
print("📦 Boxplot delle correlazioni RSA salvato!")

df_corr = pd.DataFrame(corrtest_results)
df_corr.to_excel(os.path.join(output_base, "rsa_corrtest_log_vs_others.xlsx"), index=False)
print("🔬 CorrTest log vs altri salvato!")
