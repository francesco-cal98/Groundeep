# visualizer_with_logging.py
import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.manifold import MDS
from sklearn.metrics import pairwise_distances, mean_squared_error
from scipy.stats import spearmanr
import umap
import wandb
from itertools import product
import statsmodels.api as sm
from scipy.spatial.distance import pdist
from sklearn.linear_model import LinearRegression

class VisualizerWithLogging:

    def __init__(self, wandb_run, output_dir, embedding_analyzer):
        self.wandb_run = wandb_run
        self.output_dir = output_dir
        self.embedding_analyzer = embedding_analyzer
        os.makedirs(self.output_dir, exist_ok=True)
        self.rsa_result_list = []

    def reduce_dimensions(self, data, method='umap', **kwargs):
        if method == 'umap':
            reducer = umap.UMAP(n_components=2, random_state=42, **kwargs)
        elif method == 'pca':
            reducer = PCA(n_components=2, **kwargs)
        elif method == 'mds':
            reducer = MDS(n_components=2, random_state=42, **kwargs)
        else:
            raise ValueError(f"Unsupported reduction method: {method}")
        return reducer.fit_transform(data)

    def plot_feature_correlation_matrix(self, features_dict, arch_name, dist_name):
        df = pd.DataFrame(features_dict)
        corr_matrix = df.corr()

        plt.figure(figsize=(8, 6))
        sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap="coolwarm", square=True, cbar=True)
        plt.title(f"{arch_name} – {dist_name} – Feature Correlation Matrix")
        plt.tight_layout()

        save_path = os.path.join(self.output_dir, f"{arch_name}_correlation_matrix.jpg")
        plt.savefig(save_path, dpi=300)
        self.wandb_run.log({f"{arch_name}/{dist_name}/correlation_matrix": wandb.Image(plt.gcf())})
        plt.close()

    def plot_2d_embedding_and_correlations(self, emb_2d, features_dict, arch_name, dist_name, method_name='UMAP'):
        fig, axs = plt.subplots(4, 3, figsize=(18, 16))
        axs = axs.flatten()

        numerosity = features_dict["N_list"]
        sc = axs[0].scatter(emb_2d[:, 0], emb_2d[:, 1], c=numerosity, cmap="viridis", s=40)
        axs[0].set_title(f"2D {method_name} – colored by Numerosity")
        axs[0].set_xlabel(f"{method_name}-1")
        axs[0].set_ylabel(f"{method_name}-2")
        plt.colorbar(sc, ax=axs[0], label="Numerosity")

        plot_idx = 1
        correlations = {}
        for feat_name, feat_values in features_dict.items():
            for i, dim_label in enumerate([f"{method_name}-1", f"{method_name}-2"]):
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

        fig.suptitle(f"2D {method_name} + Feature Correlations – {arch_name}", fontsize=18)
        fig.tight_layout(rect=[0, 0.03, 1, 0.95])
        plot_path = os.path.join(self.output_dir, f"{arch_name}_{method_name.lower()}_2d_plot.jpg")
        plt.savefig(plot_path, dpi=300)
        self.wandb_run.log({f"{arch_name}/{dist_name}/2d_plot_{method_name.lower()}": wandb.Image(fig)})
        plt.close(fig)
        return correlations

    def rsa_analysis(self, arch_name, dist_name, metric="cosine"):
        output = self.embedding_analyzer.output_dict
        embeddings = output[f'Z_{dist_name}']
        labels = output[f'labels_{dist_name}']
        cumArea = np.array(output[f'cumArea_{dist_name}'])
        FA = np.array(output[f'FA_{dist_name}'])
        CH = np.array(output[f'CH_{dist_name}'])

        def compute_brain_rdm(emb):
            if metric == "cosine":
                normed = emb / np.linalg.norm(emb, axis=1, keepdims=True)
                return pdist(normed, metric='cosine')
            elif metric == "euclidean":
                return pdist(emb, metric='euclidean')
            else:
                raise ValueError("Unsupported metric")

        def compute_model_rdm(values):
            return pdist(values.reshape(-1, 1), metric='euclidean')

        brain_rdm = compute_brain_rdm(embeddings)

        features = {
            "log": np.log1p(labels),
            "cumArea": cumArea,
            "FA": FA,
            "CH": CH
        }

        model_rdms = {k: compute_model_rdm(v.astype(np.float64)) for k, v in features.items()}

        rsa_results = {}
        for name, model_rdm in model_rdms.items():
            rho, _ = spearmanr(brain_rdm, model_rdm)
            rsa_results[name] = rho
            self.rsa_result_list.append({
                "arch": arch_name,
                "distribution": dist_name,
                "encoding": name,
                "spearman_rho": rho
            })
            self.wandb_run.log({f"{arch_name}/{dist_name}/rsa_rho_{name}": rho})

        # Boxplot logging
        df_rsa = pd.DataFrame(self.rsa_result_list)
        plt.figure(figsize=(8, 6))
        sns.boxplot(data=df_rsa, x="encoding", y="spearman_rho", hue="distribution")
        plt.title("RSA Correlation Distribution")
        plt.xlabel("Encoding")
        plt.ylabel("Spearman Rho")
        plt.tight_layout()
        self.wandb_run.log({"rsa/boxplot_rsa_correlations": wandb.Image(plt.gcf())})
        plt.close()

        # Regressione lineare
        model_names = list(features.keys())
        X = np.stack([model_rdms[n] for n in model_names], axis=1)
        y = brain_rdm
        reg = LinearRegression().fit(X, y)

        for name, coef in zip(model_names, reg.coef_):
            self.wandb_run.log({f"{arch_name}/{dist_name}/rsa_regression_coef_{name}": coef})

        return rsa_results
