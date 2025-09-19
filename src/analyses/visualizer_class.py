import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import wandb
from scipy.spatial.distance import pdist, squareform
from scipy.stats import spearmanr, ttest_rel
from sklearn.decomposition import PCA, FastICA
from sklearn.linear_model import LinearRegression
import umap
from scipy.fftpack import fft2
import statsmodels.api as sm
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.metrics import silhouette_score
from scipy.stats import spearmanr,kendalltau,rankdata
from src.utils.wandb_utils  import log_barplot



from skimage.metrics import structural_similarity as ssim # Assicurati che scikit-image sia installato

# ==============================================================================
# CLASSE: VisualizerWithLogging
# ==============================================================================

class VisualizerWithLogging:
    """
    Una classe per gestire la visualizzazione e il logging dei risultati delle analisi
    degli embedding, inclusa l'integrazione con Weights & Biases (WandB).
    """

    def __init__(self, wandb_run, global_output_dir):
        """
        Inizializza il Visualizer.

        Args:
            wandb_run: L'istanza di WandB Run per il logging.
            global_output_dir (str): La directory base dove salvare tutti gli output.
        """
        self.wandb_run = wandb_run
        self.global_output_dir = global_output_dir
        self.output_dir = global_output_dir # La directory di output predefinita per i plot

        # Liste per accumulare i risultati da tutte le analisi, per i plot combinati
        self.all_rsa_results = []
        self.all_mse_results = []
        self.all_afp_results = []
        self.all_ssim_results = []
        
        # Dizionario per accumulare i dati RSA per il report LaTeX (per facilità)
        self.rsa_data_for_latex = {} 
        self.all_reconstructions_info = [] # Per il report LaTeX, se applicabile

        print(f"VisualizerWithLogging initialized. Output directory: {self.output_dir}")

    def _compute_brain_rdm(self, embeddings):
        """Calcola Brain RDM: 1 - Pearson correlation tra embeddings"""
        if embeddings.shape[0] < 2:
            return np.array([])
        corr = np.corrcoef(embeddings)
        dist = 1 - corr
        # Restituisco il vettore upper triangular per correlazioni
        return squareform(dist, checks=False)

    def _compute_model_rdm(self, values):
        """Calcola RDM per feature modello (numerosità, cumArea, CH)"""
        values = np.array(values)
        if len(values) < 2:
            return np.array([])
        dist = pdist(values.reshape(-1, 1), metric='euclidean')
        return dist

    def _rank_normalize_rdm(self, rdm):
        """Rank transform + normalizzazione a [0,1]"""
        if len(rdm) < 2:
            return rdm
        ranks = rankdata(rdm)
        return (ranks - ranks.min()) / (ranks.max() - ranks.min())

    def _kendall_correlation(self, rdm1, rdm2):
        """Kendall tau tra due RDM vettorizzati"""
        if len(rdm1) != len(rdm2) or len(rdm1) < 2:
            return np.nan, np.nan
        tau, p = kendalltau(rdm1, rdm2)
        return tau, p

    # --- RSA Analysis ---
    def rsa_analysis(self, arch_name, dist_name, embedding_analyzer):
        print(f"Running RSA for {arch_name} - {dist_name}")
        output = embedding_analyzer._get_encodings()
        embeddings = np.array(output.get(f"Z_{dist_name}", []), dtype=np.float64)
        labels = np.array(output.get(f"labels_{dist_name}", []))
        cumArea = np.array(output.get(f"cumArea_{dist_name}", []))
        CH = np.array(output.get(f"CH_{dist_name}", []))

        if embeddings.shape[0] < 2 or labels.shape[0] < 2:
            print("Skipping RSA: insufficient data")
            return

        # Brain RDM
        brain_rdm = self._compute_brain_rdm(embeddings)
        brain_rdm_rank = self._rank_normalize_rdm(brain_rdm)

        # Feature model RDM
        features = {
            "numerosity": labels,
            "cumArea": cumArea,
            "CH": CH
        }

        # Calcolo RDM modello, rank transform, e correlazione Kendall
        tau_vals = {}
        for name, vals in features.items():
            model_rdm = self._compute_model_rdm(vals)
            model_rdm_rank = self._rank_normalize_rdm(model_rdm)
            if len(model_rdm_rank) != len(brain_rdm_rank):
                continue
            tau, pval = self._kendall_correlation(brain_rdm_rank, model_rdm_rank)
            tau_vals[name] = tau
            self.all_rsa_results.append({
                "Architecture": arch_name,
                "Distribution": dist_name,
                "Feature": name,
                "Kendall Tau": tau,
                "P-value": pval
            })

            # Heatmap RDM modello
            mat = squareform(model_rdm)
            plt.figure(figsize=(5,5))
            sns.heatmap(mat, cmap="viridis", square=True, cbar=True)
            plt.title(f"RDM: {name} ({arch_name}, {dist_name})")
            plt.tight_layout()
            fpath = os.path.join(self.output_dir, f"rdm_{arch_name}_{dist_name}_{name}.jpg")
            plt.savefig(fpath, dpi=300)
            self.wandb_run.log({f"rsa/rdm/{dist_name}/{arch_name}/{name}": wandb.Image(plt.gcf())})
            plt.close()

        # Barplot dei Kendall tau (come Figura 4B-C)
        if tau_vals:
            plt.figure(figsize=(6,4))
            sns.barplot(x=list(tau_vals.keys()), y=list(tau_vals.values()), palette="deep")
            plt.ylabel("Kendall Tau")
            plt.ylim(0,1)
            plt.title(f"RSA Kendall Tau ({arch_name}, {dist_name})")
            plt.tight_layout()
            fpath = os.path.join(self.output_dir, f"rsa_tau_{arch_name}_{dist_name}.jpg")
            plt.savefig(fpath, dpi=300)
            self.wandb_run.log({f"rsa/barplot/{dist_name}/{arch_name}": wandb.Image(plt.gcf())})
            plt.close()

        print(f"RSA completed for {arch_name} - {dist_name}")






    def plot_feature_correlation_matrix(self, features, arch_name, dist_name):
        """
        Genera e salva una matrice di correlazione tra le feature.
        
        Args:
            features (dict): Dizionario di nomi di feature e array di valori.
            arch_name (str): Nome dell'architettura.
            dist_name (str): Nome della distribuzione.
        """
        print(f"  Generating feature correlation matrix for {arch_name}/{dist_name}...")
        
        if not features or any(len(v) == 0 for v in features.values()):
            print(f"    Skipping feature correlation matrix for {arch_name}/{dist_name}: No feature data.")
            return

        df_features = pd.DataFrame(features)
        df_features = df_features.loc[:, df_features.apply(pd.Series.nunique) != 1]

        if df_features.empty:
            print(f"    Skipping feature correlation matrix for {arch_name}/{dist_name}: All features are constant or empty.")
            return

        try:
            correlation_matrix = df_features.corr(method='spearman')

            plt.figure(figsize=(8, 6))
            sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f", linewidths=.5)
            plt.title(f"Feature Correlation Matrix for {arch_name} ({dist_name})")
            plt.tight_layout()
            
            fname = f"feature_correlation_matrix_{arch_name}_{dist_name}.jpg"
            fpath = os.path.join(self.output_dir, fname)
            plt.savefig(fpath, dpi=300)
            self.wandb_run.log({f"feature_analysis/{dist_name}/{arch_name}/correlation_matrix": wandb.Image(plt.gcf())})
            plt.close()
            print(f"  Feature correlation matrix saved to: {fpath}")

        except Exception as e:
            print(f"    Error generating feature correlation matrix for {arch_name}/{dist_name}: {e}")

    def reduce_dimensions(self, embeddings, method='pca'):
        """
        Riduce la dimensionalità degli embedding utilizzando PCA, ICA o UMAP.

        Args:
            embeddings (np.array): Array degli embedding.
            method (str): Metodo di riduzione ('pca', 'ica' o 'umap').

        Returns:
            np.array: Embedding ridotti a 2 dimensioni.
        """
        if embeddings.shape[0] < 2:
            print(f"Warning: Not enough samples ({embeddings.shape[0]}) for dimensionality reduction with {method}. Returning empty array.")
            return np.array([])

        # n_neighbors per UMAP non deve essere maggiore di n_samples - 1
        n_neighbors_umap = min(embeddings.shape[0] - 1, 15) 

        method = method.lower()
        if method == "umap":
            reducer = umap.UMAP(n_components=2, random_state=42, n_neighbors=n_neighbors_umap)
        elif method == "pca":
            reducer = PCA(n_components=2, random_state=42)
        elif method == "ica":
            reducer = FastICA(n_components=2, random_state=42)
        else:
            raise ValueError(f"Unsupported dimensionality reduction method: {method}. Use 'pca', 'ica' or 'umap'.")
        
        try:
            emb_2d = reducer.fit_transform(embeddings)
            return emb_2d
        except Exception as e:
            print(f"Error during {method} reduction: {e}. Returning empty array.")
            return np.array([])

    def plot_2d_embedding_and_correlations(self, emb_2d, features, arch_name, dist_name, method_name):
        """
        Genera plot 2D degli embedding colorati per feature e calcola le correlazioni.

        Args:
            emb_2d (np.array): Embedding ridotti a 2 dimensioni.
            features (dict): Dizionario di nomi di feature e array di valori.
            arch_name (str): Nome dell'architettura.
            dist_name (str): Nome della distribuzione.
            method_name (str): Nome del metodo di riduzione (e.g., 'PCA', 'UMAP').

        Returns:
            dict: Correlazioni Spearman tra dimensioni e feature.
        """
        print(f"  Generating 2D embedding plot for {arch_name}/{dist_name} using {method_name}...")
        
        if emb_2d.shape[0] == 0 or emb_2d.shape[1] != 2:
            print(f"    Skipping 2D embedding plot for {arch_name}/{dist_name}: Invalid 2D embeddings.")
            return {}

        correlations = {}
        n_features = len(features)
        
        n_cols = 3
        n_rows = int(np.ceil(n_features / n_cols))
        
        fig, axs = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 4 * n_rows))
        axs = axs.flatten()

        i = 0
        for feat_name, values in features.items():
            if i >= len(axs): 
                break
            
            if len(values) != emb_2d.shape[0] or len(values) < 2:
                print(f"    Feature '{feat_name}' length mismatch or insufficient data for embeddings. Skipping plot for this feature.")
                # Assicurati che i valori di correlazione siano NaN se non calcolabili
                correlations[f"{feat_name}_dim1"] = np.nan
                correlations[f"{feat_name}_dim2"] = np.nan
                i += 1 # Vai all'asse successivo
                continue

            rho_dim1, _ = spearmanr(emb_2d[:, 0], values)
            correlations[f"{feat_name}_dim1"] = rho_dim1
            rho_dim2, _ = spearmanr(emb_2d[:, 1], values)
            correlations[f"{feat_name}_dim2"] = rho_dim2

            if feat_name == "N_list":
                color_values = np.log(values)
            else:
                color_values = values

            sc = axs[i].scatter(emb_2d[:, 0], emb_2d[:, 1], c=color_values, cmap='viridis', s=40, alpha=0.8)
            axs[i].set_title(f"Feature: {feat_name}\nDim1={correlations[f'{feat_name}_dim1']:.2f}, Dim2={correlations[f'{feat_name}_dim2']:.2f}")
            axs[i].set_xlabel(f"{method_name}-1")
            axs[i].set_ylabel(f"{method_name}-2")
            fig.colorbar(sc, ax=axs[i], label=feat_name)
            i += 1
        
        for j in range(i, len(axs)):
            axs[j].axis('off')

        plt.suptitle(f"{method_name} 2D Embedding for {arch_name} ({dist_name})", fontsize=16)
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])

        fname = f"{method_name.lower()}_2d_embedding_{arch_name}_{dist_name}.jpg"
        fpath = os.path.join(self.output_dir, fname)
        plt.savefig(fpath, dpi=300)
        self.wandb_run.log({f"embeddings/{dist_name}/{arch_name}/{method_name}_2d_embedding": wandb.Image(plt.gcf())})
        plt.close()
        print(f"  2D embedding plot saved to: {fpath}")
        return correlations

    def plot_combined_rsa_barplots(self):
        """
        Barplot RSA (Kendall tau) raggruppato per feature model e distribuzione.
        """
        if not self.all_rsa_results:
            print("No RSA results to plot.")
            return

        df = pd.DataFrame(self.all_rsa_results)
        df = df.rename(columns={"Feature Model": "Encoding", "Kendall Tau": "Tau"})

        plt.figure(figsize=(10, 6))
        sns.barplot(data=df, x="Encoding", y="Tau", hue="Distribution", palette="deep")
        plt.title("RSA results (Kendall Tau)")
        plt.ylim(-0.1, 0.7)
        plt.xlabel("Encoding")
        plt.ylabel("Kendall Tau")
        plt.legend(title="Distribution", bbox_to_anchor=(1.02, 1), loc="upper left")
        plt.grid(axis="y", linestyle="--", alpha=0.7)
        plt.tight_layout()

        fpath = os.path.join(self.output_dir, "rsa_barplot.jpg")
        plt.savefig(fpath, dpi=300)
        self.wandb_run.log({"rsa/barplot": wandb.Image(plt.gcf())})
        plt.close()
        print(f"  RSA barplot saved to {fpath}")



    def linear_separability(self, embeddings, labels, arch_name, dist_name):
        """
        Valuta la separabilità lineare delle numerosità negli embeddings
        usando Logistic Regression con cross-validation.
        """
        clf = LogisticRegression(max_iter=500, multi_class="multinomial")
        scores = cross_val_score(clf, embeddings, labels, cv=5, scoring="accuracy")
        mean_acc = np.mean(scores)

        print(f"[{arch_name}-{dist_name}] Linear separability (LogReg): {mean_acc:.3f}")
        wandb.log({f"{arch_name}_{dist_name}_linear_sep_acc": mean_acc})
        return mean_acc


    def distance_monotonicity_by_bin(self, embeddings, labels, arch_name, dist_name, bins, metric="cosine"):
        from sklearn.metrics import pairwise_distances
        results = {}
        for bmin, bmax in bins:
            idx = (labels >= bmin) & (labels <= bmax)
            if np.sum(idx) < 5:
                continue
            d_emb = pairwise_distances(embeddings[idx], metric=metric)
            d_num = np.abs(labels[idx][:, None] - labels[idx][None, :])

            triu_idx = np.triu_indices_from(d_emb, k=1)
            emb_flat = d_emb[triu_idx]
            num_flat = d_num[triu_idx]

            rho, _ = spearmanr(emb_flat, num_flat)
            results[f"{bmin}-{bmax}"] = rho
        
        df = pd.DataFrame(list(results.items()), columns=["bin", "spearman_rho"])
        wandb.log({f"{arch_name}_{dist_name}_monotonicity_bins": wandb.Table(dataframe=df)})
        log_barplot(results, "distance_monotonicity", arch_name, dist_name, ylabel="rho-spearman")

        return results

    

    def cluster_separability(self, embeddings, labels, arch_name, dist_name):
        """
        Calcola il silhouette score delle numerosità negli embeddings.
        """
        try:
            sil = silhouette_score(embeddings, labels, metric="euclidean")
            print(f"[{arch_name}-{dist_name}] Silhouette score: {sil:.3f}")
            wandb.log({f"{arch_name}_{dist_name}_silhouette": sil})
        except Exception as e:
            print(f"Silhouette score failed: {e}")
            sil = np.nan
        return sil


