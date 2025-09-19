import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import wandb
from scipy.spatial.distance import pdist 
from scipy.stats import spearmanr
from sklearn.decomposition import PCA, FastICA
import umap
from scipy.stats import spearmanr,kendalltau
from sklearn.metrics import pairwise_distances
import os
import wandb
import numpy as np
import pandas as pd
import sys
from glob import glob
from statsmodels.stats.multitest import multipletests


# === PATH SETUP ===
current_dir = os.getcwd()
project_root = os.path.abspath(os.path.join(current_dir, '..', '..'))
sys.path.append(project_root)
sys.path.append(current_dir) # Add current_dir if it contains other necessary modules

# Import your classes
from src.analyses.embedding_analysis import Embedding_analysis
from src.utils.wandb_utils import log_reconstructions_to_wandb




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


    # --- Metodi di analisi (prendono embedding_analyzer come parametro) ---

    def rsa_analysis(self, arch_name, dist_name, embedding_analyzer, metric="cosine"):
        """
        Esegue l'analisi RSA con test one-sided + correzione FDR.
        """

        print(f"  Running RSA for {arch_name} under {dist_name} distribution...")

        output_dict = embedding_analyzer._get_encodings()
        embeddings = np.array(output_dict.get(f'Z_{dist_name}', []), dtype=np.float64)
        labels = np.array(output_dict.get(f'labels_{dist_name}', []))
        cumArea = np.array(output_dict.get(f'cumArea_{dist_name}', []))
        FA = np.array(output_dict.get(f'FA_{dist_name}', []))
        CH = np.array(output_dict.get(f'CH_{dist_name}', []))

        if not embeddings.shape[0] > 1 or not labels.shape[0] > 1:
            print(f"    Skipping RSA for {arch_name}/{dist_name}: Insufficient data.")
            return

        brain_rdm = self._compute_brain_rdm(embeddings, metric=metric)
        if len(brain_rdm) == 0:
            print(f"    Skipping RSA for {arch_name}/{dist_name}: Brain RDM could not be computed.")
            return

        features_for_rsa = {
            "numerosity_linear": labels,
            "numerosity_log": np.log(labels),
            "cumArea": cumArea,
            "FA": FA,
            "CH": CH
        }

        current_arch_rsa_results = []

        # === Step 1: compute tau + one-sided p ===
        for name, values in features_for_rsa.items():
            if len(values) > 1:
                model_rdm = self._compute_model_rdm(values, metric='euclidean')
                if len(model_rdm) == 0:
                    continue
                if len(brain_rdm) == len(model_rdm):
                    tau, p_two_sided = kendalltau(brain_rdm, model_rdm)

                    if tau > 0:
                        p_one_sided = p_two_sided / 2
                    else:
                        p_one_sided = 1 - (p_two_sided / 2)

                    self.all_rsa_results.append({
                        "Architecture": arch_name,
                        "Distribution": dist_name,
                        "Feature Model": name,
                        "Kendall Tau": tau,
                        "P-value (1-sided)": p_one_sided
                    })
                    current_arch_rsa_results.append({
                        "Feature Model": name,
                        "Tau": tau,
                        "P-value (1-sided)": p_one_sided
                    })

        # === Step 2: FDR correction ===
        df_all_rsa = pd.DataFrame(self.all_rsa_results)
        if df_all_rsa.empty:
            print("⚠️ No RSA results to analyze.")
            return

        reject, pvals_corrected, _, _ = multipletests(df_all_rsa["P-value (1-sided)"], 
                                                    alpha=0.01, method='fdr_bh')
        df_all_rsa["Significant_FDR"] = reject
        df_all_rsa["P-value FDR"] = pvals_corrected

        # Aggiorna la lista con info di significatività
        self.all_rsa_results = df_all_rsa.to_dict("records")

        # === Step 3: Save Excel locally ===
        excel_path = os.path.join(self.output_dir, f"rsa_results_{arch_name}_{dist_name}.xlsx")
        df_all_rsa.to_excel(excel_path, index=False)
        print(f"📂 RSA results saved to {excel_path}")

        # === Step 4: Plot barplot con * e ** ===
        df_barplot_rsa = df_all_rsa.rename(columns={'Feature Model': 'Encoding'})

        # Ordine asse X
        encoding_order = ["numerosity_linear", "numerosity_log",  "cumArea", "FA", "CH"] 
        existing_encodings_for_plot = [enc for enc in encoding_order if enc in df_barplot_rsa['Encoding'].unique()]

        df_barplot_rsa['Encoding'] = pd.Categorical(df_barplot_rsa['Encoding'], 
                                                    categories=existing_encodings_for_plot, 
                                                    ordered=True)
        df_barplot_rsa = df_barplot_rsa.sort_values(by='Encoding')

        plt.figure(figsize=(12, 7)) 
        ax = sns.barplot(data=df_barplot_rsa, x='Encoding', y='Kendall Tau', hue='Distribution', palette='deep')

        # Aggiungi asterischi sopra le barre significative
        for i, row in df_barplot_rsa.iterrows():
            if row['Significant_FDR']:  # se significativo dopo FDR
                # coord x del tick (asse X)
                x = i % len(existing_encodings_for_plot)  # calcolo semplificato se più distribuzioni
                y = row['Kendall Tau']
                ax.text(x, y + 0.02, "*", ha='center', va='bottom', fontsize=14, color='red')

        plt.title('Correlation plot of RSA (with significance)')
        plt.xlabel('Encoding')
        plt.ylabel("Kendall Tau")
        plt.ylim(-0.1, 0.7) 
        plt.legend(title='Statistical Distribution', bbox_to_anchor=(1.02, 1), loc='upper left')
        plt.grid(axis='y', linestyle='--', alpha=0.7) 
        plt.tight_layout()

        boxplot_path = os.path.join(self.output_dir, 'rsa_combined_boxplot_with_significance.jpg')
        plt.savefig(boxplot_path, dpi=300)
        plt.close()
        print(f"  📦 Combined RSA barplot with significance saved to: {boxplot_path}")

            
    
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
        sns.heatmap(dist_matrix, xticklabels=np.array(unique_labels,dtype=np.uint8)
, yticklabels=np.array(unique_labels,dtype=np.uint8)
,
                    cmap="viridis", square=True, annot=False, fmt=".2f")
        #plt.title(f"Pairwise class distances ({metric})\n{arch_name} - {dist_name}")
        #plt.xticks(rotation=45)
        plt.xlabel("Numerosity")
        plt.ylabel("Numerosity")
        plt.tight_layout()

        fpath = os.path.join(self.output_dir, f"pairwise_class_rdm_{arch_name}_{dist_name}.jpg")
        plt.savefig(fpath, dpi=300)
        #self.wandb_run.log({f"rsa/pairwise_class_rdm/{dist_name}/{arch_name}": wandb.Image(plt.gcf())})
        plt.close()
        print(f"Pairwise class RDM saved and logged: {fpath}")


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
                color_values = (values)
            else:
                color_values = values

            sc = axs[i].scatter(emb_2d[:, 0], emb_2d[:, 1], c=color_values, cmap='viridis', s=40, alpha=0.8)
            axs[i].set_title(f"Dim1={correlations[f'{feat_name}_dim1']:.2f}, Dim2={correlations[f'{feat_name}_dim2']:.2f}")
            axs[i].set_xlabel(f"{method_name}-1")
            axs[i].set_ylabel(f"{method_name}-2")
            fig.colorbar(sc, ax=axs[i], label=feat_name)
            i += 1
        
        for j in range(i, len(axs)):
            axs[j].axis('off')

        #plt.suptitle(f"{method_name} 2D Embedding for {arch_name} ({dist_name})", fontsize=16)
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])

        fname = f"{method_name.lower()}_2d_embedding_{arch_name}_{dist_name}.jpg"
        fpath = os.path.join(self.output_dir, fname)
        plt.savefig(fpath, dpi=300)
        self.wandb_run.log({f"embeddings/{dist_name}/{arch_name}/{method_name}_2d_embedding": wandb.Image(plt.gcf())})
        plt.close()
        print(f"  2D embedding plot saved to: {fpath}")
        return correlations



# /home/student/Desktop/Groundeep/src/analyses/main_analysis.py


# === Configuration ===
configs = {
    "uniform": {
        "data_path": "/home/student/Desktop/Groundeep/stimuli_dataset_adaptive",
        "data_file": "stimuli_dataset.npz",
        "network_dir": "/home/student/Desktop/Groundeep/networks/uniform/poster"
    },
    "zipfian": {
        "data_path": "/home/student/Desktop/Groundeep/stimuli_dataset_adaptive",
        "data_file": "stimuli_dataset.npz",
        "network_dir": "/home/student/Desktop/Groundeep/networks/zipfian/poster"
    }
}

# Define a base output directory for all analyses
global_output_base_dir = "/home/student/Desktop/Groundeep/outputs/analysis_results_poster/" 
os.makedirs(global_output_base_dir, exist_ok=True)

#os.environ['WANDB_API_KEY']='91fc380c0e7e443c45abd10540ae8e907d735e3a'

# === Init WandB ===
run = wandb.init(project="groundeep-visualization2")
print("WANDB mode:", wandb.run.settings.mode)
run.log({"debug/first": 1})


# === Initialize Visualizer ONLY ONCE outside the loops ===
# NON PASSARE embedding_analyzer qui!
visualizer = VisualizerWithLogging(run, global_output_base_dir)

# === Main loop ===
all_correlations = [] # To aggregate correlations from all architectures/distributions

for dist_name, cfg in configs.items():
    print(f"\n📊 Processing distribution: {dist_name.upper()}")

    data_path = cfg["data_path"]
    data_file = cfg["data_file"]
    network_dir = cfg["network_dir"]
    
    # Create a specific output directory for this distribution's plots
    dist_output_dir = os.path.join(global_output_base_dir, f"{dist_name}_plots")
    os.makedirs(dist_output_dir, exist_ok=True)
    
    pkl_files = glob(os.path.join(network_dir, "*.pkl"))
    
    if not pkl_files:
        print(f"⚠️ No .pkl files found in {network_dir}. Skipping this distribution.")
        continue

    for pkl_path in pkl_files:
        arch_name = os.path.splitext(os.path.basename(pkl_path))[0]
        print(f" - Analyzing architecture: {arch_name}")

        # === Load model and compute encodings ===
        embedding_analyzer = Embedding_analysis(data_path, data_file, pkl_path, pkl_path, arch_name)
        output_dict = embedding_analyzer._get_encodings()
        
        # get reconstructed inputs from DBN 
        original_inputs, reconstructed = embedding_analyzer.reconstruct_input(embedding_analyzer.inputs_uniform)
        
        # Prepare data for plotting and analysis
        embeddings = np.array(output_dict.get(f'Z_{dist_name}', []), dtype=np.float64)
        
        # normalization of embeddings
        embeddings_mean = np.mean(embeddings, axis=0)
        embeddings_std = np.std(embeddings, axis=0)
        embeddings = (embeddings - embeddings_mean) / (embeddings_std + 1e-10)  # Adding a small constant to avoid division by zero
        features = {
            "N_list": np.array(output_dict.get(f'labels_{dist_name}', [])),
            "cumArea": np.array(output_dict.get(f'cumArea_{dist_name}', [])),
            "CH": np.array(output_dict.get(f'CH_{dist_name}', []))
        }

        # === Call Visualizer methods ===
        # Ora passiamo embedding_analyzer ai metodi specifici che ne hanno bisogno
        # e usiamo dist_output_dir se vogliamo salvare i plot individuali in sottocartelle.
        # Attualmente i metodi del Visualizer non prendono plot_output_dir.

        visualizer.plot_feature_correlation_matrix(features, arch_name, dist_name) 
        # Se questo plot deve andare in dist_output_dir, la sua definizione in visualizer_class.py
        # dovrebbe prendere plot_output_dir come parametro e usarlo. Per ora, salverà in self.output_dir

        if embeddings.shape[0] > 1: 
            emb_2d = visualizer.reduce_dimensions(embeddings, method='pca') 
            correlations = visualizer.plot_2d_embedding_and_correlations(emb_2d, features, arch_name, dist_name, method_name="PC")
            
            correlations = visualizer.plot_2d_embedding_and_correlations(emb_2d, features, arch_name, dist_name, method_name="PC")
            
            # --- INIZIO BLOCCO DI CODICE CORRETTO ---
            # Itera sulle chiavi e valori del dizionario 'correlations'
            for key, corr_val in correlations.items():
                # Le chiavi sono nel formato "nome_feature_dimX" (es. "N_list_dim1")
                # Dobbiamo separare il nome della feature e la dimensione
                parts = key.rsplit('_', 1) # Divide la stringa dall'ultimo underscore, max 1 divisione
                if len(parts) == 2:
                    feat_name = parts[0]
                    dim_label = parts[1]
                else:
                    # Caso di fallback, se la chiave non è nel formato atteso
                    feat_name = key
                    dim_label = "unknown" 
                    print(f"Warning: Unexpected correlation key format: {key}. Assigning dim_label as 'unknown'.")

                all_correlations.append({
                    'arch': arch_name,
                    'feature': feat_name,
                    'dimension': dim_label,
                    'correlation': corr_val,
                    'distribution': dist_name
                })
            # --- FINE BLOCCO DI CODICE CORRETTO ---
        # Passa embedding_analyzer a questi metodi!
        visualizer.plot_pairwise_class_rdm( arch_name, dist_name, embedding_analyzer, feature="labels", metric="cosine")
        visualizer.rsa_analysis(arch_name, dist_name, embedding_analyzer=embedding_analyzer)
        
        #visualizer.mse_analysis(arch_name, embedding_analyzer=embedding_analyzer)
        #visualizer.afp_analysis(arch_name, embedding_analyzer=embedding_analyzer)
        #visualizer.ssim_analysis(arch_name, embedding_analyzer=embedding_analyzer)
        #visualizer.comprehensive_analysis(arch_name, embedding_analyzer=embedding_analyzer)
        # === Nuove analisi sugli embeddings ===
        #visualizer.linear_separability(embeddings, features["N_list"], arch_name, dist_name)
        #visualizer.distance_monotonicity(embeddings,embedding_analyzer, arch_name, dist_name)
        #visualizer.cluster_separability(embeddings, features["N_list"], arch_name, dist_name)


        # === Log reconstructions to WandB ===
        if original_inputs is not None and reconstructed is not None and len(original_inputs) > 0:
            log_reconstructions_to_wandb(original_inputs[:10], reconstructed[:10], name=arch_name)
        else:
            print(f"Skipping reconstruction logging for {arch_name}: original_inputs or reconstructed are empty.")

    print(f"✅ Finished processing all architectures for {dist_name} distribution.")

# === After all loops are complete, generate combined plots and reports ===
# Questi metodi ora accederanno a self.all_rsa_results che contiene i dati di ENTRAMBE le distribuzioni
print("\n🔄 Generating combined RSA barplots (and boxplot)...")
#visualizer.plot_combined_rsa_barplots() 

print("📝 Generating LaTeX report data...")
#visualizer.generate_latex_report_data()

# Save the aggregated correlation dataframe
if all_correlations:
    df_corr_all = pd.DataFrame(all_correlations)
    df_corr_all.to_excel(os.path.join(global_output_base_dir, "all_architectures_correlations.xlsx"), index=False)
    print(f"📊 Aggregated correlations saved to: {os.path.join(global_output_base_dir, 'all_architectures_correlations.xlsx')}")
else:
    print("No correlations data collected to save to Excel.")

print("\n🎉 All analyses complete and logged to WandB.")
wandb.finish()     



