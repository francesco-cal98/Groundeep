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
from scipy.stats import spearmanr
from src.utils.wandb_utils  import log_barplot
from sklearn.metrics import pairwise_distances



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

    # --- RSA UTILITIES ---
    def _compute_brain_rdm(self, embeddings, metric="cosine"):
        """Calcola la Matrice di Dissomiglianza (RDM) per gli embedding cerebrali."""
        if not embeddings.shape[0] > 1:
            return np.array([]) # Ritorna array vuoto se non ci sono abbastanza campioni
        if metric == "cosine":
            normed = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)
            # Gestisci NaN/Inf in normed se ci sono vettori zero
            normed = np.nan_to_num(normed)
            return pdist(normed, metric='cosine')
        elif metric == "euclidean":
            return pdist(embeddings, metric='euclidean')
        else:
            raise ValueError("Unsupported distance metric. Use 'cosine' or 'euclidean'.")

    def _compute_model_rdm(self, values, metric='euclidean'):
        """Calcola la Matrice di Dissomiglianza (RDM) per i modelli di feature."""
        if not len(values) > 1:
            return np.array([]) # Ritorna array vuoto se non ci sono abbastanza campioni
        x = values.astype(np.float64)
        return pdist(x.reshape(-1, 1), metric=metric)

    def _correlation_test(self, brain_rdm, model_rdm1, model_rdm2):
        """
        Esegue un test di correlazione di Spearman e un test di differenza tra correlazioni.
        """
        # Assicurati che gli RDMs non siano vuoti e abbiano abbastanza elementi
        if len(brain_rdm) < 3 or len(model_rdm1) < 3 or len(model_rdm2) < 3: 
            return np.nan, np.nan, np.nan, np.nan, np.nan # Aggiunto un valore di ritorno per il p-value

        # Calcola le correlazioni di Spearman
        rho1, p1 = spearmanr(brain_rdm, model_rdm1)
        rho2, p2 = spearmanr(brain_rdm, model_rdm2)
        
        # Fisher's Z-transformation for comparing correlations
        z = np.nan # Default to NaN
        if not (np.isnan(rho1) or np.isnan(rho2) or abs(rho1) == 1 or abs(rho2) == 1):
            n = len(brain_rdm)
            if n > 3: # Minimo n-3 per la varianza
                z1 = 0.5 * np.log((1 + rho1) / (1 - rho1))
                z2 = 0.5 * np.log((1 + rho2) / (1 - rho2))
                diff_z = z1 - z2
                se_diff_z = np.sqrt(1 / (n - 3) + 1 / (n - 3)) 
                if se_diff_z > 0:
                    z = diff_z / se_diff_z
                else:
                    z = np.nan
            else:
                z = np.nan 

        return rho1, p1, rho2, p2, z # Ritorna anche i p-values individuali e la Z-score


    # --- Metodi di analisi (prendono embedding_analyzer come parametro) ---

    def rsa_analysis(self, arch_name, dist_name, embedding_analyzer, metric="cosine"):
        """
        Esegue l'analisi RSA per una data architettura e distribuzione,
        e accumula i risultati.

        Args:
            arch_name (str): Nome dell'architettura.
            dist_name (str): Nome della distribuzione (e.g., 'uniform', 'zipfian').
            embedding_analyzer (Embedding_analysis): L'istanza dell'analizzatore di embedding.
            metric (str): La metrica di distanza da usare per gli RDM (e.g., 'cosine', 'euclidean').
        """
        print(f"  Running RSA for {arch_name} under {dist_name} distribution...")
        
        output_dict = embedding_analyzer._get_encodings()
        embeddings = np.array(output_dict.get(f'Z_{dist_name}', []), dtype=np.float64)
        labels = np.array(output_dict.get(f'labels_{dist_name}', []))
        cumArea = np.array(output_dict.get(f'cumArea_{dist_name}', []))
        FA = np.array(output_dict.get(f'FA_{dist_name}', []))
        CH = np.array(output_dict.get(f'CH_{dist_name}', []))

        if not embeddings.shape[0] > 1 or not labels.shape[0] > 1:
            print(f"    Skipping RSA for {arch_name}/{dist_name}: Insufficient embedding or label data.")
            return

        brain_rdm = self._compute_brain_rdm(embeddings, metric=metric)
        if len(brain_rdm) == 0:
            print(f"    Skipping RSA for {arch_name}/{dist_name}: Brain RDM could not be computed.")
            return

        features_for_rsa = {
            "numerosity_linear": labels,
            "numerosity_log": np.log(labels),
            "numerosity_sqrt": np.sqrt(labels),
            "cumArea": cumArea,
            "FA": FA,
            "CH": CH
        }

        current_arch_rsa_results = [] # Per tenere traccia dei risultati di questa architettura per i test e LaTeX
        for name, values in features_for_rsa.items():
            if len(values) > 1:
                model_rdm = self._compute_model_rdm(values, metric='euclidean')
                if len(model_rdm) == 0:
                    print(f"    Skipping RSA for feature '{name}' in {arch_name}/{dist_name}: Model RDM could not be computed.")
                    continue

                if len(brain_rdm) == len(model_rdm):
                    rho, p_value = spearmanr(brain_rdm, model_rdm)
                    self.all_rsa_results.append({
                        "Architecture": arch_name,
                        "Distribution": dist_name,
                        "Feature Model": name,
                        "Spearman Rho": rho,
                        "P-value": p_value # Aggiungi il p-value per completezza
                    })
                    current_arch_rsa_results.append({"Feature Model": name, "Rho": rho, "P-value": p_value})
                else:
                    print(f"    Skipping RSA for feature '{name}' in {arch_name}/{dist_name}: RDM dimensions mismatch ({len(brain_rdm)} vs {len(model_rdm)}).")
            else:
                print(f"    Skipping RSA for feature '{name}' in {arch_name}/{dist_name}: Insufficient feature values.")
        
        # Per i test di correlazione tra modelli (es. log vs linear/sqrt)
        rho_log_lin, p_log_lin, rho_lin_val, p_lin_val, z_log_lin = (np.nan,)*5 # Inizializza con NaN
        rho_log_sqrt, p_log_sqrt, rho_sqrt_val, p_sqrt_val, z_log_sqrt = (np.nan,)*5 # Inizializza con NaN

        if all(f in features_for_rsa for f in ["numerosity_log", "numerosity_linear", "numerosity_sqrt"]):
            log_rdm = self._compute_model_rdm(features_for_rsa["numerosity_log"])
            linear_rdm = self._compute_model_rdm(features_for_rsa["numerosity_linear"])
            sqrt_rdm = self._compute_model_rdm(features_for_rsa["numerosity_sqrt"])

            if len(log_rdm) > 0 and len(linear_rdm) > 0:
                rho_log_lin, p_log_lin, rho_lin_val, p_lin_val, z_log_lin = self._correlation_test(brain_rdm, log_rdm, linear_rdm)
            if len(log_rdm) > 0 and len(sqrt_rdm) > 0:
                rho_log_sqrt, p_log_sqrt, rho_sqrt_val, p_sqrt_val, z_log_sqrt = self._correlation_test(brain_rdm, log_rdm, sqrt_rdm)

            # Logga i risultati dei test di correlazione
            self.wandb_run.log({
                f"rsa_correlation_test/{dist_name}/{arch_name}/log_vs_linear_z": z_log_lin,
                f"rsa_correlation_test/{dist_name}/{arch_name}/log_vs_linear_rho_log": rho_log_lin,
                f"rsa_correlation_test/{dist_name}/{arch_name}/log_vs_linear_p_log": p_log_lin,
                f"rsa_correlation_test/{dist_name}/{arch_name}/log_vs_linear_rho_linear": rho_lin_val,
                f"rsa_correlation_test/{dist_name}/{arch_name}/log_vs_linear_p_linear": p_lin_val,
                f"rsa_correlation_test/{dist_name}/{arch_name}/log_vs_sqrt_z": z_log_sqrt,
                f"rsa_correlation_test/{dist_name}/{arch_name}/log_vs_sqrt_rho_log": rho_log_sqrt,
                f"rsa_correlation_test/{dist_name}/{arch_name}/log_vs_sqrt_p_log": p_log_sqrt,
                f"rsa_correlation_test/{dist_name}/{arch_name}/log_vs_sqrt_rho_sqrt": rho_sqrt_val,
                f"rsa_correlation_test/{dist_name}/{arch_name}/log_vs_sqrt_p_sqrt": p_sqrt_val,
            })
        
        # Per il report LaTeX (memorizza i valori più rilevanti)
        if dist_name not in self.rsa_data_for_latex:
            self.rsa_data_for_latex[dist_name] = {}
        
        # Recupera i rho specifici per il report LaTeX
        log_rho_val = next((item["Rho"] for item in current_arch_rsa_results if item["Feature Model"] == "numerosity_log"), np.nan)
        linear_rho_val = next((item["Rho"] for item in current_arch_rsa_results if item["Feature Model"] == "numerosity_linear"), np.nan)
        cumarea_rho_val = next((item["Rho"] for item in current_arch_rsa_results if item["Feature Model"] == "cumArea"), np.nan)
        
        self.rsa_data_for_latex[dist_name][arch_name] = {
            'numerosity_log_rho': log_rho_val,
            'numerosity_linear_rho': linear_rho_val,
            'cumArea_rho': cumarea_rho_val,
            'log_vs_linear_z': z_log_lin,
            'log_vs_sqrt_z': z_log_sqrt,
        }
        print(f"  RSA analysis for {arch_name}/{dist_name} completed.")
    
    def plot_pairwise_class_rdm(self, arch_name, dist_name, embedding_analyzer, feature="labels", metric="cosine"):
        """
        Calcola le distanze tra rappresentazioni medie per classe (numerosità)
        e le plotta come heatmap.
        """
        print(f"Generating pairwise class RDM for {arch_name} - {dist_name}")

        output = embedding_analyzer._get_encodings()
        embeddings = np.array(output.get(f"Z_{dist_name}", []), dtype=np.float64)
        labels = np.array(output.get(f"{feature}_{dist_name}", []))

        if embeddings.shape[0] < 2 or labels.shape[0] < 2:
            print("Skipping class RDM: insufficient data")
            return

        # Calcola rappresentazioni medie per ciascun livello di numerosity
        unique_labels = np.unique(labels)
        class_means = []
        for ul in unique_labels:
            class_emb = embeddings[labels == ul]
            if class_emb.shape[0] > 0:
                class_means.append(class_emb.mean(axis=0))
            else:
                class_means.append(np.zeros(embeddings.shape[1]))  # fallback se vuoto

        class_means = np.vstack(class_means)  # shape: (n_classes, embedding_dim)

        # Calcola pairwise distances tra rappresentazioni medie
        dist_matrix = pairwise_distances(class_means, metric=metric, n_jobs=-1)

        # Plot heatmap
        plt.figure(figsize=(8,6))
        sns.heatmap(dist_matrix, xticklabels=unique_labels, yticklabels=unique_labels,
                    cmap="viridis", square=True, annot=False, fmt=".2f")
        plt.title(f"Pairwise class distances ({metric})\n{arch_name} - {dist_name}")
        plt.xlabel("Numerosity")
        plt.ylabel("Numerosity")
        plt.tight_layout()

        fpath = os.path.join(self.output_dir, f"pairwise_class_rdm_{arch_name}_{dist_name}.jpg")
        plt.savefig(fpath, dpi=300)
        self.wandb_run.log({f"rsa/pairwise_class_rdm/{dist_name}/{arch_name}": wandb.Image(plt.gcf())})
        plt.close()
        print(f"Pairwise class RDM saved and logged: {fpath}")


    def mse_analysis(self, arch_name,embedding_analyzer):
        original_inputs, reconstructed = embedding_analyzer.reconstruct_input()
        out = embedding_analyzer.output_dict

        numerosities = out['labels_uniform']
        numerosities_bin = out['numerosity_bin_uniform']
        cumarea_bins = out['cumArea_bins_uniform']
        cumarea_bin_ids = out['cumArea_uniform']
        convex_hull_bins = out['convex_hull_bins_uniform']
        convex_hull_bin = out['convex_hull_uniform']


        mses = np.mean((original_inputs - reconstructed) ** 2, axis=1)

        df = pd.DataFrame({
            "numerosity": numerosities_bin,
            "cumarea_bin": cumarea_bin_ids,
            "convex_hull_bin": convex_hull_bin,
            "mse": mses
        })

        pivot_table = df.groupby(["cumarea_bin", "numerosity"])["mse"].mean().unstack(fill_value=np.nan)
        pivot_table = pivot_table.sort_index(ascending=False)  # Ordina cumulative area dal basso verso l'alto

        plt.figure(figsize=(10, 6))
        sns.heatmap(pivot_table, annot=True, fmt=".3f", cmap="viridis")
        plt.title(f"MSE Heatmap – {arch_name}")
        plt.xlabel("Numerosity_bin")
        plt.ylabel("Cum-Area bin")
        plt.tight_layout()
        self.wandb_run.log({f"{arch_name}/mse_heatmap_cumarea": wandb.Image(plt.gcf())})
        plt.close()

        pivot_hull = df.groupby(["convex_hull_bin", "numerosity"])["mse"].mean().unstack(fill_value=np.nan)
        pivot_hull = pivot_hull.sort_index(ascending=False)

        plt.figure(figsize=(10, 6))
        sns.heatmap(pivot_hull, annot=True, fmt=".3f", cmap="viridis")
        plt.title(f"MSE Heatmap – Convex Hull vs Numerosity – {arch_name}")
        plt.xlabel("Numerosity_bin")
        plt.ylabel("Convex Hull bin")
        plt.tight_layout()
        self.wandb_run.log({f"{arch_name}/mse_heatmap_convexhull": wandb.Image(plt.gcf())})
        plt.close()


            # === MSE vs Numerosity per ciascun livello di Cumulative Area + linea globale ===
        plt.figure(figsize=(8, 5))

        # Linea globale: MSE medio per numerosity su tutto il dataset
        mse_vs_numerosity = df.groupby("numerosity")["mse"].mean()
        plt.plot(
            mse_vs_numerosity.index, 
            mse_vs_numerosity.values, 
            label="All CumAreas", 
            color="black", 
            linestyle="--", 
            linewidth=2
        )

        # Linee per ogni livello di cumarea_bin
        for cumarea_level, group in df.groupby("cumarea_bin"):
            mse_by_numerosity = group.groupby("numerosity")["mse"].mean()
            plt.plot(
                mse_by_numerosity.index, 
                mse_by_numerosity.values, 
                label=f"CumArea Bin {cumarea_level}", 
                marker='o'
            )
            

        plt.title(f"MSE vs Numerosity per Cumulative Area – {arch_name}")
        plt.xlabel("Numerosity")
        plt.ylabel("Mean MSE")
        plt.legend(title="Legend")
        plt.grid(True)
        plt.tight_layout()

        # Log su Weights & Biases
        wandb.log({f"{arch_name}/mse_vs_numerosity_by_cumarea_with_total": wandb.Image(plt.gcf())})
        plt.close()

        reg_df = pd.DataFrame({
            "numerosity": out["labels_uniform"],
            "cumulative_area": out["cumArea_uniform"],
            "convex_hull": out["convex_hull_uniform"],
            "mse": mses
        })

        X = reg_df[["numerosity", "cumulative_area", "convex_hull"]]
        y = reg_df["mse"]
        X_const = sm.add_constant(X)
        model = sm.OLS(y, X_const).fit()

        coeffs = model.params
        pvals = model.pvalues
        conf_int = model.conf_int()

        self.wandb_run.log({
            f"{arch_name}/regression_coefficients": wandb.Table(
                columns=["Variable", "Coef", "P-value", "CI_lower", "CI_upper"],
                data=[
                    [var, float(coeffs[var]), float(pvals[var]), float(conf_int.loc[var][0]), float(conf_int.loc[var][1])]
                    for var in coeffs.index
                ]
            )
        })

    def compute_afp(self,img1, img2, img_shape=(100, 100)):
        # Reshape se necessario
        if img1.ndim == 1:
            img1 = img1.reshape(img_shape)
        if img2.ndim == 1:
            img2 = img2.reshape(img_shape)

        # Fourier amplitude
        fft1 = np.abs(fft2(img1))
        fft2_ = np.abs(fft2(img2))

        # Normalizza
        fft1 /= np.sum(fft1)
        fft2_ /= np.sum(fft2_)

        return np.sum(np.abs(fft1 - fft2_))

    def afp_analysis(self, arch_name,embedding_analyzer):
        original_inputs, reconstructed = embedding_analyzer.reconstruct_input()
        out = embedding_analyzer.output_dict

        # === Feature bins ===
        numerosities = out['labels_uniform']
        numerosities_bin = out['numerosity_bin_uniform']
        cumarea_bins = out['cumArea_bins_uniform']
        cumarea = out['cumArea_uniform']
        convex_hull_bins = out['convex_hull_bins_uniform']
        convex_hull = out['convex_hull_uniform']
        #items_uniform = out['Items_uniform']
        #items_uniform_bin = out['Items_bins_uniform']


        # === AFP per immagine ===
        afps = np.array([
            self.compute_afp(original_inputs[i], reconstructed[i])
            for i in range(len(original_inputs))
        ])

        df = pd.DataFrame({
            "numerosity": numerosities_bin,
            "cumarea": cumarea,
            "convex_hull": convex_hull,
            "afp": afps
        })

        # === Heatmap CumArea x Numerosity ===
        pivot_table = df.groupby(["cumarea", "numerosity"])["afp"].mean().unstack(fill_value=np.nan)
        pivot_table = pivot_table.sort_index(ascending=False)

        plt.figure(figsize=(10, 6))
        sns.heatmap(pivot_table, annot=True, fmt=".3f", cmap="plasma")
        plt.title(f"AFP Heatmap – {arch_name}")
        plt.xlabel("Numerosity_bin")
        plt.ylabel("Cum-Area bin")
        plt.tight_layout()
        self.wandb_run.log({f"{arch_name}/afp_heatmap_cumarea": wandb.Image(plt.gcf())})
        plt.close()

        # === Heatmap Convex Hull ===
        pivot_hull = df.groupby(["convex_hull", "numerosity"])["afp"].mean().unstack(fill_value=np.nan)
        pivot_hull = pivot_hull.sort_index(ascending=False)

        plt.figure(figsize=(10, 6))
        sns.heatmap(pivot_hull, annot=True, fmt=".3f", cmap="plasma")
        plt.title(f"AFP Heatmap – Convex Hull vs Numerosity – {arch_name}")
        plt.xlabel("Numerosity_bin")
        plt.ylabel("Convex Hull bin")
        plt.tight_layout()
        self.wandb_run.log({f"{arch_name}/afp_heatmap_convexhull": wandb.Image(plt.gcf())})
        plt.close()

        # === Line plot AFP vs Numerosity ===
        plt.figure(figsize=(8, 5))
        afp_vs_numerosity = df.groupby("numerosity")["afp"].mean()
        plt.plot(
            afp_vs_numerosity.index,
            afp_vs_numerosity.values,
            label="All CumAreas",
            color="black",
            linestyle="--",
            linewidth=2
        )
        self.wandb_run.log({f"{arch_name}/afp_vs_numerosity_by_cumarea_with_total": wandb.Image(plt.gcf())})

        for cumarea_level, group in df.groupby("cumarea"):
            afp_by_numerosity = group.groupby("numerosity")["afp"].mean()
            plt.plot(
                afp_by_numerosity.index,
                afp_by_numerosity.values,
                label=f"CumArea Bin {cumarea_level}",
                marker='o'
            )

        plt.title(f"AFP vs Numerosity per Cumulative Area – {arch_name}")
        plt.xlabel("Numerosity")
        plt.ylabel("Mean AFP")
        plt.legend(title="Legend")
        plt.grid(True)
        plt.tight_layout()
        self.wandb_run.log({f"{arch_name}/afp_vs_numerosity_by_cumarea_with_total": wandb.Image(plt.gcf())})
        plt.close()

        # === Regressione multipla (Numerosità, Area, CH → AFP) ===
        reg_df = pd.DataFrame({
            "numerosity": out["labels_uniform"],
            "cumulative_area": out["cumArea_uniform"],
            "convex_hull": out["convex_hull_uniform"],
            "afp": afps
        })

        X = reg_df[["numerosity", "cumulative_area", "convex_hull"]]
        y = reg_df["afp"]
        X_const = sm.add_constant(X)
        model = sm.OLS(y, X_const).fit()

        coeffs = model.params
        pvals = model.pvalues
        conf_int = model.conf_int()

        self.wandb_run.log({
            f"{arch_name}/afp_regression_coefficients": wandb.Table(
                columns=["Variable", "Coef", "P-value", "CI_lower", "CI_upper"],
                data=[
                    [var, float(coeffs[var]), float(pvals[var]), float(conf_int.loc[var][0]), float(conf_int.loc[var][1])]
                    for var in coeffs.index
                ]
            )
        })

        # === AFP vs features: plot separati ===
        panel_key_sep = f"{arch_name}/afp_vs_features"

        afp_scores = [
            self.compute_afp(orig, recon)
            for orig, recon in zip(original_inputs, reconstructed)
        ]

        afp_df = pd.DataFrame({"afp": afp_scores})
        feature_map = {
            "numerosity": numerosities,
            "cumarea": cumarea,
            "convex_hull": convex_hull,
            #"items_uniform": items_uniform
        }

        for feat_name, feat_values in feature_map.items():
            if feat_values is None:
                print(f"⚠️ Feature '{feat_name}' non trovata, skip.")
                continue

            afp_df[feat_name] = feat_values
            agg = afp_df.groupby(feat_name)["afp"].mean()

            plt.figure(figsize=(8, 5))
            plt.plot(
                agg.index,
                agg.values,
                marker="o",
                linestyle="-"
            )
            plt.title(f"AFP vs {feat_name.replace('_',' ').title()} – {arch_name}")
            plt.xlabel(feat_name.replace('_',' ').title())
            plt.ylabel("Mean AFP")
            plt.grid(True)
            plt.tight_layout()

            self.wandb_run.log({f"{panel_key_sep}/{arch_name}/AFP_vs_{feat_name}": wandb.Image(plt.gcf())})
            plt.close()


        # === AFP vs features: plot combinato ===
        panel_key_combined = f"{arch_name}/afp_vs_features_combined"

        plt.figure(figsize=(10, 6))
        for feat_name, feat_values in feature_map.items():
            if feat_values is None:
                continue

            afp_df[feat_name] = feat_values.astype(int)
            agg = afp_df.groupby(feat_name)["afp"].mean()

            plt.plot(
                agg.index,
                agg.values,
                marker="o",
                linestyle="-",
                label=feat_name.replace('_', ' ').title()
            )

        plt.title(f"AFP vs Features – {arch_name}")
        plt.xlabel("Feature Value")
        plt.ylabel("Mean AFP")
        plt.legend(title="Feature")
        plt.grid(True)
        plt.tight_layout()

        self.wandb_run.log({f"{panel_key_combined}": wandb.Image(plt.gcf())})
        plt.close()

    def ssim_analysis(self, arch_name,embedding_analyzer):

        original_inputs, reconstructed = embedding_analyzer.reconstruct_input()
        out = embedding_analyzer.output_dict

        numerosities = out['labels_uniform']
        cumarea = out['cumArea_uniform']
        convex_hull = out['convex_hull_uniform']
        #items_uniform = out['Items_uniform']

        # Calcolo SSIM
        ssim_scores = [
            ssim(orig, recon, data_range=1.0)
            for orig, recon in zip(original_inputs, reconstructed)
        ]

        ssim_df = pd.DataFrame({
            "ssim": ssim_scores
        })

        feature_map = {
            "numerosity": numerosities,
            "cumarea": cumarea,
            "convex_hull": convex_hull,
            #"items_uniform": items_uniform
        }

        reg_df = pd.DataFrame({
            "numerosity": out["labels_uniform"],
            "cumulative_area": out["cumArea_uniform"],
            "convex_hull": out["convex_hull_uniform"],
            "ssim": ssim_scores
        })

        X = reg_df[["numerosity", "cumulative_area", "convex_hull"]]
        y = reg_df["ssim"]
        X_const = sm.add_constant(X)
        model = sm.OLS(y, X_const).fit()

        coeffs = model.params
        pvals = model.pvalues
        conf_int = model.conf_int()

        self.wandb_run.log({
            f"{arch_name}/ssim_regression_coefficients": wandb.Table(
                columns=["Variable", "Coef", "P-value", "CI_lower", "CI_upper"],
                data=[
                    [var, float(coeffs[var]), float(pvals[var]), float(conf_int.loc[var][0]), float(conf_int.loc[var][1])]
                    for var in coeffs.index
                ]
            )
        })


        # === Plots separati per feature ===
        for feat_name, feat_values in feature_map.items():
            if feat_values is None:
                print(f"⚠️ Feature '{feat_name}' non trovata, skip.")
                continue

            ssim_df[feat_name] = feat_values.astype(int)
            agg = ssim_df.groupby(feat_name)["ssim"].mean()

            plt.figure(figsize=(8, 5))
            plt.plot(
                agg.index,
                agg.values,
                marker="o",
                linestyle="-"
            )
            plt.title(f"SSIM vs {feat_name.replace('_',' ').title()} – {arch_name}")
            plt.xlabel(feat_name.replace('_',' ').title())
            plt.ylabel("Mean SSIM")
            plt.grid(True)
            plt.tight_layout()

            self.wandb_run.log({
                f"{arch_name}/ssim_vs_{feat_name}": wandb.Image(plt.gcf())
            })
            plt.close()

        # === Plot combinato ===
        plt.figure(figsize=(10, 6))
        for feat_name, feat_values in feature_map.items():
            if feat_values is None:
                continue

            ssim_df[feat_name] = feat_values.astype(int)
            agg = ssim_df.groupby(feat_name)["ssim"].mean()

            plt.plot(
                agg.index,
                agg.values,
                marker="o",
                linestyle="-",
                label=feat_name.replace('_', ' ').title()
            )

        plt.title(f"SSIM vs Features – {arch_name}")
        plt.xlabel("Feature Value")
        plt.ylabel("Mean SSIM")
        plt.legend(title="Feature")
        plt.grid(True)
        plt.tight_layout()

        self.wandb_run.log({
            f"{arch_name}/ssim_vs_features_combined": wandb.Image(plt.gcf())
        })
        plt.close()



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
        Genera boxplot e barplot RSA raggruppati per encoding e distribuzione.
        Questa funzione dovrebbe essere chiamata DOPO aver eseguito rsa_analysis per tutte le combinazioni.
        """
        print("\n📊 Generating combined RSA plots (boxplot and barplots)...")
        if not self.all_rsa_results:
            print("Nessun dato RSA raccolto per i plot combinati.")
            return

        df_all_rsa = pd.DataFrame(self.all_rsa_results)
        print("  DEBUG: DataFrame for combined RSA plots created. Head:\n", df_all_rsa.head())
        print("  DEBUG: Unique Feature Models (Encodings):", df_all_rsa['Feature Model'].unique())

        # ====================================================================
        # >>> INIZIO SEZIONE: PLOT BOXPLOT RSA (come da tua immagine) <<<
        # ====================================================================
        print("\n  🔄 Generazione del Boxplot RSA combinato...")

        df_boxplot_rsa = df_all_rsa.rename(columns={'Feature Model': 'Encoding'})
        
        # Ordine per l'asse X, basato sul tuo plot d'esempio e i nomi delle feature
        encoding_order = ["numerosity_linear", "numerosity_log", "numerosity_sqrt", "cumArea", "FA", "CH"] 
        
        existing_encodings_for_plot = [enc for enc in encoding_order if enc in df_boxplot_rsa['Encoding'].unique()]
        
        if not existing_encodings_for_plot:
            print("  ⚠️ No valid encodings found for combined RSA boxplot. Check feature names in rsa_analysis.")
        else:
            df_boxplot_rsa['Encoding'] = pd.Categorical(df_boxplot_rsa['Encoding'], 
                                                        categories=existing_encodings_for_plot, 
                                                        ordered=True)
            df_boxplot_rsa = df_boxplot_rsa.sort_values(by='Encoding')

            plt.figure(figsize=(12, 7)) 
            sns.boxplot(data=df_boxplot_rsa, x='Encoding', y='Spearman Rho', hue='Distribution', palette='deep')
            
            plt.title('Distribuzione delle correlazioni RSA per encoding e distribuzione')
            plt.xlabel('Encoding')
            plt.ylabel("Spearman Rho")
            plt.ylim(-0.1, 0.7) 
            plt.legend(title='Distribuzione', bbox_to_anchor=(1.02, 1), loc='upper left')
            plt.grid(axis='y', linestyle='--', alpha=0.7) 
            plt.tight_layout()

            boxplot_path = os.path.join(self.output_dir, 'rsa_combined_boxplot.jpg')
            plt.savefig(boxplot_path, dpi=300) 
            self.wandb_run.log({"rsa/combined_boxplot": wandb.Image(plt.gcf())})
            plt.close()
            print(f"  📦 Combined RSA boxplot saved to: {boxplot_path}")



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


