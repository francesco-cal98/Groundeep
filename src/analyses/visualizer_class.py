import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import wandb
from scipy.spatial.distance import pdist, squareform
from scipy.stats import spearmanr, ttest_rel
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression
import umap
from skimage.metrics import structural_similarity as ssim_metric # Assicurati che scikit-image sia installato

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
            "numerosity_log": np.log1p(labels),
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


    def mse_analysis(self, arch_name, embedding_analyzer):
        """
        Calcola e logga l'Errore Quadratico Medio (MSE) per le ricostruzioni.

        Args:
            arch_name (str): Nome dell'architettura.
            embedding_analyzer (Embedding_analysis): L'istanza dell'analizzatore di embedding.
        """
        print(f"  Running MSE analysis for {arch_name}...")
        original_inputs, reconstructed = embedding_analyzer.reconstruct_input(embedding_analyzer.inputs_uniform)

        if original_inputs is None or reconstructed is None or len(original_inputs) == 0:
            print(f"    Skipping MSE for {arch_name}: No reconstruction data available.")
            return

        original_inputs = np.array(original_inputs)
        reconstructed = np.array(reconstructed)

        if original_inputs.shape != reconstructed.shape:
            print(f"    Shape mismatch for {arch_name}: Original {original_inputs.shape} vs Reconstructed {reconstructed.shape}. Skipping MSE.")
            return

        mse = np.mean((original_inputs - reconstructed) ** 2)
        self.all_mse_results.append({
            "Architecture": arch_name,
            "MSE": mse
        })
        self.wandb_run.log({f"reconstruction_metrics/{arch_name}/MSE": mse})
        print(f"  MSE for {arch_name}: {mse:.4f}")

    def afp_analysis(self, arch_name, embedding_analyzer):
        """
        Analisi AFP (Average Feature Presence) - Placeholder.
        Questo è un placeholder. Se hai un modo specifico per calcolare AFP,
        implementalo qui usando embedding_analyzer per accedere ai dati necessari.
        """
        print(f"  Running AFP analysis for {arch_name}...")
        
        # Esempio: se AFP dipendesse dagli activations del modello
        # activations = embedding_analyzer.get_activations() # Metodo ipotetico
        # if activations is not None and activations.shape[0] > 0:
        #     # Calcola AFP in base ai tuoi criteri
        #     dummy_afp_value = np.mean(activations > 0) # Esempio molto semplice
        # else:
        #     dummy_afp_value = np.nan
        #     print(f"    No activations available for AFP for {arch_name}.")
        
        # Per ora, registra un valore dummy
        dummy_afp_value = np.random.rand() * 10 
        
        self.all_afp_results.append({
            "Architecture": arch_name,
            "AFP": dummy_afp_value
        })
        self.wandb_run.log({f"feature_metrics/{arch_name}/AFP": dummy_afp_value})
        print(f"  AFP for {arch_name}: {dummy_afp_value:.4f} (Dummy Value)")


    def ssim_analysis(self, arch_name, embedding_analyzer):
        """
        Calcola e logga l'Indice di Similarità Strutturale (SSIM) per le ricostruzioni.

        Args:
            arch_name (str): Nome dell'architettura.
            embedding_analyzer (Embedding_analysis): L'istanza dell'analizzatore di embedding.
        """
        print(f"  Running SSIM analysis for {arch_name}...")
        original_inputs, reconstructed = embedding_analyzer.reconstruct_input(embedding_analyzer.inputs_uniform)

        if original_inputs is None or reconstructed is None or len(original_inputs) == 0:
            print(f"    Skipping SSIM for {arch_name}: No reconstruction data available.")
            return

        img_size = 100 # Assicurati che questa dimensione sia corretta per i tuoi dati
        
        ssim_values = []
        for i in range(min(len(original_inputs), len(reconstructed))):
            try:
                img1 = original_inputs[i].reshape(img_size, img_size)
                img2 = reconstructed[i].reshape(img_size, img_size)
                
                img1 = img1.astype(float)
                img2 = img2.astype(float)

                # data_range è la differenza tra il massimo e il minimo valore possibile dei pixel.
                # Per immagini binarie normalizzate tra 0 e 1, è 1.0.
                current_ssim = ssim_metric(img1, img2, data_range=1.0) 
                ssim_values.append(current_ssim)
            except Exception as e:
                print(f"    Error calculating SSIM for sample {i} of {arch_name}: {e}. Skipping.")
                continue
        
        if ssim_values:
            avg_ssim = np.mean(ssim_values)
            self.all_ssim_results.append({
                "Architecture": arch_name,
                "SSIM": avg_ssim
            })
            self.wandb_run.log({f"reconstruction_metrics/{arch_name}/SSIM": avg_ssim})
            print(f"  Avg SSIM for {arch_name}: {avg_ssim:.4f}")
        else:
            print(f"    No SSIM values calculated for {arch_name}.")


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
        Riduce la dimensionalità degli embedding utilizzando PCA o UMAP.

        Args:
            embeddings (np.array): Array degli embedding.
            method (str): Metodo di riduzione ('pca' o 'umap').

        Returns:
            np.array: Embedding ridotti a 2 dimensioni.
        """
        if embeddings.shape[0] < 2:
            print(f"Warning: Not enough samples ({embeddings.shape[0]}) for dimensionality reduction with {method}. Returning empty array.")
            return np.array([])
        
        # n_neighbors per UMAP non deve essere maggiore di n_samples - 1
        n_neighbors_umap = min(embeddings.shape[0] - 1, 15) 

        if method.lower() == "umap":
            reducer = umap.UMAP(n_components=2, random_state=42, n_neighbors=n_neighbors_umap)
        elif method.lower() == "pca":
            reducer = PCA(n_components=2)
        else:
            raise ValueError(f"Unsupported dimensionality reduction method: {method}. Use 'pca' or 'umap'.")
        
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

            sc = axs[i].scatter(emb_2d[:, 0], emb_2d[:, 1], c=values, cmap='viridis', s=40, alpha=0.8)
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


    def generate_latex_report_data(self):
        """
        Genera i dati formattati per il report LaTeX.
        Accede a `self.rsa_data_for_latex` e altri dati accumulati.
        """
        print("\n📝 Generating LaTeX report data...")
        
        latex_output = ""
        latex_output += "\\documentclass{article}\n"
        latex_output += "\\usepackage[utf8]{inputenc}\n"
        latex_output += "\\usepackage{amsmath}\n"
        latex_output += "\\usepackage{amssymb}\n"
        latex_output += "\\usepackage{graphicx}\n"
        latex_output += "\\usepackage{booktabs}\n" # For better table lines
        latex_output += "\\usepackage[T1]{fontenc}\n"
        latex_output += "\\usepackage{float}\n" # For [H] placement
        latex_output += "\\usepackage{hyperref}\n" # For hyperlinks
        latex_output += "\\hypersetup{colorlinks=true, linkcolor=blue, urlcolor=cyan}\n"
        latex_output += "\\title{Analisi degli Embedding delle Reti DBN}\n"
        latex_output += "\\author{}\n"
        latex_output += "\\date{"+ pd.Timestamp.now().strftime('%d %B %Y') +"}\n" # Automatically add current date
        latex_output += "\\begin{document}\n"
        latex_output += "\\maketitle\n"

        latex_output += "\\section*{Risultati RSA (Representational Similarity Analysis)}\n"
        latex_output += "Di seguito sono riportate le correlazioni di Spearman ($\\rho$) tra le RDM cerebrali "
        latex_output += "e le RDM dei modelli di feature per le diverse architetture e distribuzioni, "
        latex_output += "insieme ai valori Z di Fisher per il confronto tra i modelli di numerosità.\\\n"
        latex_output += "Un valore Z positivo indica che il primo modello correlato è più fortemente "
        latex_output += "correlato agli embedding rispetto al secondo modello.\\\n\n"


        for dist_name, arch_data in self.rsa_data_for_latex.items():
            latex_output += f"\\subsection*{{Distribuzione: {dist_name.capitalize()}}}\n"
            latex_output += "\\begin{table}[H]\n"
            latex_output += "\\centering\n"
            latex_output += "\\caption{Correlazioni RSA per la distribuzione " + dist_name.capitalize() + "}\n"
            latex_output += "\\label{tab:rsa_" + dist_name + "}\n"
            # Ho aggiunto un'ulteriore colonna per Z (Log vs Sqrt) per coerenza con rsa_analysis
            latex_output += "\\begin{tabular}{l c c c c c}\n" 
            latex_output += "\\toprule\n"
            latex_output += "Architettura & $\\rho$ (Log) & $\\rho$ (Linear) & $\\rho$ (CumArea) & Z (Log vs Linear) & Z (Log vs Sqrt) \\\\\n"
            latex_output += "\\midrule\n"
            for arch_name, metrics in arch_data.items():
                log_rho = f"{metrics.get('numerosity_log_rho', np.nan):.3f}" if not np.isnan(metrics.get('numerosity_log_rho', np.nan)) else "N/A"
                linear_rho = f"{metrics.get('numerosity_linear_rho', np.nan):.3f}" if not np.isnan(metrics.get('numerosity_linear_rho', np.nan)) else "N/A"
                cumarea_rho = f"{metrics.get('cumArea_rho', np.nan):.3f}" if not np.isnan(metrics.get('cumArea_rho', np.nan)) else "N/A"
                z_log_lin = f"{metrics.get('log_vs_linear_z', np.nan):.2f}" if not np.isnan(metrics.get('log_vs_linear_z', np.nan)) else "N/A"
                z_log_sqrt = f"{metrics.get('log_vs_sqrt_z', np.nan):.2f}" if not np.isnan(metrics.get('log_vs_sqrt_z', np.nan)) else "N/A"
                
                latex_output += f"{arch_name} & {log_rho} & {linear_rho} & {cumarea_rho} & {z_log_lin} & {z_log_sqrt} \\\\\n"
            latex_output += "\\bottomrule\n"
            latex_output += "\\end{tabular}\n"
            latex_output += "\\end{table}\n\n"

        # Aggiungi grafici RSA combinati
        latex_output += "\\begin{figure}[H]\n"
        latex_output += "\\centering\n"
        latex_output += "\\includegraphics[width=0.9\\textwidth]{rsa_combined_boxplot.jpg}\n"
        latex_output += "\\caption{Boxplot delle correlazioni RSA tra embedding e modelli di feature per diverse distribuzioni e encoding.}\n"
        latex_output += "\\label{fig:rsa_combined_boxplot}\n"
        latex_output += "\\end{figure}\n\n"
        
        latex_output += "\\begin{figure}[H]\n"
        latex_output += "\\centering\n"
        latex_output += "\\includegraphics[width=0.9\\textwidth]{rsa_barplot_zipfian.jpg}\n"
        latex_output += "\\caption{Correlazioni RSA per la distribuzione Zipfian (modelli di numerosità).}\n"
        latex_output += "\\label{fig:rsa_barplot_zipfian}\n"
        latex_output += "\\end{figure}\n\n"

        latex_output += "\\begin{figure}[H]\n"
        latex_output += "\\centering\n"
        latex_output += "\\includegraphics[width=0.9\\textwidth]{rsa_barplot_uniform.jpg}\n"
        latex_output += "\\caption{Correlazioni RSA per la distribuzione Uniforme (modelli di numerosità).}\n"
        latex_output += "\\label{fig:rsa_barplot_uniform}\n"
        latex_output += "\\end{figure}\n\n"
        
        latex_output += "\\begin{figure}[H]\n"
        latex_output += "\\centering\n"
        latex_output += "\\includegraphics[width=0.9\\textwidth]{rsa_barplot_combined_facetgrid.jpg}\n"
        latex_output += "\\caption{Correlazioni RSA attraverso architetture e distribuzioni (modelli di numerosità).}\n"
        latex_output += "\\label{fig:rsa_barplot_combined_facetgrid}\n"
        latex_output += "\\end{figure}\n\n"

        latex_output += "\\section*{Risultati MSE (Mean Squared Error)}\n"
        if self.all_mse_results:
            df_mse = pd.DataFrame(self.all_mse_results)
            latex_output += "\\begin{table}[H]\n"
            latex_output += "\\centering\n"
            latex_output += "\\caption{MSE delle ricostruzioni per architettura}\n"
            latex_output += "\\label{tab:mse_results}\n"
            latex_output += df_mse.to_latex(index=False, float_format="%.4f")
            latex_output += "\\end{table}\n\n"
        else:
            latex_output += "Nessun dato MSE disponibile.\\\n\n"

        latex_output += "\\section*{Risultati SSIM (Structural Similarity Index)}\n"
        if self.all_ssim_results:
            df_ssim = pd.DataFrame(self.all_ssim_results)
            latex_output += "\\begin{table}[H]\n"
            latex_output += "\\centering\n"
            latex_output += "\\caption{SSIM delle ricostruzioni per architettura}\n"
            latex_output += "\\label{tab:ssim_results}\n"
            latex_output += df_ssim.to_latex(index=False, float_format="%.4f")
            latex_output += "\\end{table}\n\n"
        else:
            latex_output += "Nessun dato SSIM disponibile.\\\n\n"

        latex_output += "\\end{document}\n"

        report_path = os.path.join(self.output_dir, "analysis_report.tex")
        with open(report_path, "w") as f:
            f.write(latex_output)
        print(f"  LaTeX report generated at: {report_path}")
        print("  Per compilare il PDF, esegui: pdflatex analysis_report.tex")