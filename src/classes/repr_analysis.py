# src/classes/repr_analysis.py

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import wandb
from scipy.spatial.distance import pdist
from scipy.stats import kendalltau
from sklearn.metrics import pairwise_distances
from statsmodels.stats.multitest import multipletests
import pickle
import sys

# === PATH SETUP ===
current_dir = os.getcwd()
project_root = os.path.abspath(os.path.join(current_dir, '..', '..'))
sys.path.append(project_root)
sys.path.append(current_dir) # Add current_dir if it contains other necessary modules

from src.analyses.embedding_analysis import Embedding_analysis




class RepresentationalAnalysis:
    """
    Una classe per eseguire le analisi di rappresentazione su un modello addestrato.
    Incapsula la logica per RSA, RDM e altri plot.
    """

    def __init__(self, model, output_dir="results/analysis"):
        self.model = model
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)
        self.embedding_analyzer = Embedding_analysis(self.model)

    def _compute_brain_rdm(self, embeddings, metric="cosine"):
        """Calcola la Matrice di Dissomiglianza (RDM) per gli embedding."""
        if not embeddings.shape[0] > 1:
            return np.array([])
        if metric == "cosine":
            normed = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)
            normed = np.nan_to_num(normed)
            return pdist(normed, metric='cosine')
        elif metric == "euclidean":
            return pdist(embeddings, metric='euclidean')
        else:
            raise ValueError("Unsupported distance metric. Use 'cosine' or 'euclidean'.")

    def _compute_model_rdm(self, values, metric='euclidean'):
        """Calcola la Matrice di Dissomiglianza (RDM) per i modelli di feature."""
        if not len(values) > 1:
            return np.array([])
        x = values.astype(np.float64)
        return pdist(x.reshape(-1, 1), metric=metric)

    def rsa_analysis(self, embeddings, features, dist_name, arch_name, metric="cosine"):
        """Esegue l'analisi RSA."""
        if not embeddings.shape[0] > 1 or not features.get('Labels', []).shape[0] > 1:
            print(f"Skipping RSA for {arch_name}/{dist_name}: Insufficient data.")
            return

        brain_rdm = self._compute_brain_rdm(embeddings, metric=metric)
        if len(brain_rdm) == 0:
            return

        features_for_rsa = {
            "numerosity_linear": features.get('Labels'),
            "numerosity_log": np.log(features.get('Labels')),
            "cumArea": features.get('Cumulative Area'),
            "CH": features.get('Convex Hull'),
        }

        results = []
        for name, values in features_for_rsa.items():
            if values is not None and len(values) > 1:
                model_rdm = self._compute_model_rdm(values, metric='euclidean')
                if len(model_rdm) == 0:
                    continue
                if len(brain_rdm) == len(model_rdm):
                    tau, p_two_sided = kendalltau(brain_rdm, model_rdm)
                    p_one_sided = p_two_sided / 2 if tau > 0 else 1 - (p_two_sided / 2)
                    results.append({
                        "Architecture": arch_name,
                        "Distribution": dist_name,
                        "Feature Model": name,
                        "Kendall Tau": tau,
                        "P-value (1-sided)": p_one_sided
                    })

        df_results = pd.DataFrame(results)
        if df_results.empty:
            print("No RSA results to analyze.")
            return

        reject, pvals_corrected, _, _ = multipletests(
            df_results["P-value (1-sided)"], alpha=0.01, method='fdr_bh'
        )
        df_results["Significant_FDR"] = reject
        df_results["P-value FDR"] = pvals_corrected

        excel_path = os.path.join(self.output_dir, f"rsa_results_{arch_name}_{dist_name}.xlsx")
        df_results.to_excel(excel_path, index=False)
        print(f"RSA results saved to {excel_path}")

        plt.figure(figsize=(12, 7))
        ax = sns.barplot(data=df_results, x='Feature Model', y='Kendall Tau', palette='deep')
        plt.title(f'RSA Correlation for {arch_name} ({dist_name})')
        plt.tight_layout()
        plot_path = os.path.join(self.output_dir, f"rsa_barplot_{arch_name}_{dist_name}.jpg")
        plt.savefig(plot_path, dpi=300)
        plt.close()
        print(f"RSA barplot saved to {plot_path}")

    def plot_pairwise_class_rdm(self, embeddings, features, dist_name, arch_name, metric="cosine"):
        """Calcola e plotta la Matrice di Dissomiglianza tra le classi."""
        print(f"Generating pairwise class RDM for {arch_name} - {dist_name}")
        labels = features.get("Labels")

        if embeddings.shape[0] < 2 or labels is None or labels.shape[0] < 2:
            print("Skipping class RDM: insufficient data")
            return

        unique_labels = np.unique(labels)
        class_means = []
        for ul in unique_labels:
            class_emb = embeddings[labels == ul]
            if class_emb.shape[0] > 0:
                class_means.append(class_emb.mean(axis=0))
            else:
                class_means.append(np.zeros(embeddings.shape[1]))

        class_means = np.vstack(class_means)
        dist_matrix = pairwise_distances(class_means, metric=metric, n_jobs=-1)

        plt.figure(figsize=(8, 6))
        sns.heatmap(dist_matrix, 
                    xticklabels=np.array(unique_labels, dtype=np.uint8), 
                    yticklabels=np.array(unique_labels, dtype=np.uint8),
                    cmap="viridis", square=True)
        plt.title(f"Pairwise class distances ({metric})\n{arch_name} - {dist_name}")
        plt.xlabel("Numerosity")
        plt.ylabel("Numerosity")
        plt.tight_layout()
        fpath = os.path.join(self.output_dir, f"pairwise_class_rdm_{arch_name}_{dist_name}.jpg")
        plt.savefig(fpath, dpi=300)
        plt.close()
        print(f"Pairwise class RDM saved: {fpath}")

    def run_all_analyses(self, data, arch_name, dist_name):
        """Metodo per eseguire tutte le analisi."""
        print(f"Running analyses for {arch_name} - {dist_name}...")
        
        embeddings = self.model.represent(data.val_batch)
        features = self.model.features
        
        self.rsa_analysis(
            embeddings=embeddings.detach().cpu().numpy(),
            features=features,
            dist_name=dist_name,
            arch_name=arch_name
        )
        self.plot_pairwise_class_rdm(
            embeddings=embeddings.detach().cpu().numpy(),
            features=features,
            dist_name=dist_name,
            arch_name=arch_name
        )
        
        # Esempio di utilizzo della classe Embedding_analysis
        output_dict = self.embedding_analyzer._get_encodings()
        print(f"Fetched encodings for analysis: {output_dict.keys()}")

        # Esegui la selettività neuronale
        spearman_rho, pval_corrected, significant = self.embedding_analyzer.compute_classwise_selectivity_with_fdr(source=dist_name)
        print(f"Computed neuronal selectivity for {dist_name} distribution.")

        print(f"Analyses for {arch_name} complete. Results are in the {self.output_dir} directory.")