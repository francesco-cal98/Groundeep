import os
import glob
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from statsmodels.stats.multitest import fdrcorrection
from scipy.stats import spearmanr
import numpy as np
from embedding_analysis import Embedding_analysis


def rank_selectivity_by_class(Z, labels, alpha=0.05):
    """
    Calcola la correlazione one-vs-all tra ciascun neurone e ciascuna classe.
    Applica FDR per ciascuna classe e restituisce ranking dei neuroni ordinati per rho.

    Args:
        Z (ndarray): attivazioni neuroni (n_samples, n_neurons)
        labels (ndarray): etichette di classe (numerosità)
        alpha (float): soglia FDR

    Returns:
        classwise_ranking: dict[num] -> DataFrame con colonne:
            ['neuron_id', 'rho', 'pval', 'pval_corrected', 'significant']
            ordinato per |rho| decrescente
    """
    n_samples, n_neurons = Z.shape
    unique_classes = np.unique(labels)
    classwise_ranking = {}

    for c in unique_classes:
        binary_mask = (labels == c).astype(int)
        rho = np.zeros(n_neurons)
        pvals = np.zeros(n_neurons)

        for i in range(n_neurons):
            r, p = spearmanr(Z[:, i], binary_mask)
            rho[i] = r
            pvals[i] = p

        # FDR correction
        reject, pval_corr = fdrcorrection(pvals, alpha=alpha)

        df = pd.DataFrame({
            'neuron_id': np.arange(n_neurons),
            'rho': rho,
            'pval': pvals,
            'pval_corrected': pval_corr,
            'significant': reject
        })

        # Ordina per valore assoluto di rho
        df = df.reindex(df['rho'].abs().sort_values(ascending=False).index)

        classwise_ranking[c] = df

    return classwise_ranking

import matplotlib.pyplot as plt
import math
import os

def plot_all_top_neurons(ranking_dict, arch_name, output_dir, top_n=5):
    nums = sorted(ranking_dict.keys())
    n = len(nums)

    # Calcola automaticamente le dimensioni della griglia
    n_cols = 4
    n_rows = math.ceil(n / n_cols)

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4 * n_cols, 4 * n_rows))
    axes = axes.flatten()

    for i, num in enumerate(nums):
        df = ranking_dict[num].head(top_n)
        ax = axes[i]
        ax.bar(
            df['neuron_id'].astype(str),
            df['rho'],
            color=['red' if sig else 'gray' for sig in df['significant']],
            edgecolor='black'
        )
        ax.axhline(0, color='black', linestyle='--', linewidth=0.8)
        ax.set_title(f'Numerosity {num}')
        ax.set_xlabel('Neuron')
        ax.set_ylabel('ρ')
        ax.grid(True)
        ax.set_xticks([])

    # Rimuovi subplot vuoti se ci sono
    for j in range(i + 1, len(axes)):
        fig.delaxes(axes[j])

    plt.suptitle(f'Top {top_n} Neuroni Selettivi per Numerosità ({arch_name})')
    plt.tight_layout(rect=[0, 0, 1, 0.95])

    save_path = os.path.join(output_dir, f"{arch_name}_top_neurons_all.png")
    plt.savefig(save_path)
    plt.close()
    print(f"✔️ Salvato barplot: {save_path}")


# === Config ===
data_path = "/home/student/Desktop/Groundeep/circle_dataset_100x100"
data_file = "circle_dataset_100x100_v2.npz"
uniform_dir = "/home/student/Desktop/Groundeep/networks/uniform/idbn/"
zipfian_dir = "/home/student/Desktop/Groundeep/networks/zipfian/idbn/"
output_dir = "/home/student/Desktop/Groundeep/outputs/neuron_activity/"
os.makedirs(output_dir, exist_ok=True)

uniform_models = sorted(glob.glob(os.path.join(uniform_dir, "*.pkl")))


# === Loop principale ===
for uniform_path in uniform_models:
    arch_name = os.path.basename(uniform_path).replace("idbn_trained_uniform_", "").replace(".pkl", "")
    zipfian_path = os.path.join(zipfian_dir, f"idbn_trained_zipfian_{arch_name}.pkl")

    if not os.path.exists(zipfian_path):
        print(f"❌ Skipping {arch_name}: Zipfian mancante.")
        continue

    print(f"\n▶️ Processing {arch_name}")
    embedder = Embedding_analysis(data_path, data_file, uniform_path, zipfian_path, arch_name)
    embedder._get_encodings()

    # --- Selettività one-vs-all per istogramma (opzionale) ---
    _, _, significant_df = embedder.compute_classwise_selectivity_with_fdr(source='uniform')
    active_neurons_count = significant_df.sum(axis=1)
    plt.figure(figsize=(10, 6))
    plt.bar(active_neurons_count.index, active_neurons_count.values, color='skyblue', edgecolor='black')
    plt.title(f'Neuroni Selettivi per Numerosità ({arch_name})')
    plt.xlabel('Numerosity')
    plt.ylabel('N. neuroni selettivi')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"{arch_name}_significant_histogram.png"))
    plt.close()

    # --- Ranking neuroni selettivi (per tuning / ablation) ---
    Z = embedder.output_dict['Z_uniform']
    labels = embedder.output_dict['labels_uniform']
    ranking = rank_selectivity_by_class(Z, labels)

    # Raccogli tutto in un unico DataFrame
    all_ranks = []
    for num, df in ranking.items():
        df = df.copy()
        df['numerosity'] = num
        all_ranks.append(df)
    ranking_all_df = pd.concat(all_ranks, ignore_index=True)

    # Salva ranking completo
    ranking_path = os.path.join(output_dir, f"{arch_name}_ranking_all.csv")
    ranking_all_df.to_csv(ranking_path, index=False)
    print(f"✔️ Salvato ranking: {ranking_path}")

    # Salva barplot combinato
    plot_all_top_neurons(ranking, arch_name, output_dir, top_n=5)
