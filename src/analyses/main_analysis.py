# /home/student/Desktop/Groundeep/src/analyses/main_analysis.py

import os
import wandb
import numpy as np
import pandas as pd
import sys
from glob import glob

# === PATH SETUP ===
current_dir = os.getcwd()
project_root = os.path.abspath(os.path.join(current_dir, '..', '..'))
sys.path.append(project_root)
sys.path.append(current_dir) # Add current_dir if it contains other necessary modules

# Import your classes
from src.analyses.embedding_analysis import Embedding_analysis
from src.analyses.visualizer_class import VisualizerWithLogging
from src.utils.wandb_utils import log_reconstructions_to_wandb

# === Configuration ===
configs = {
    "uniform": {
        "data_path": "/home/student/Desktop/Groundeep/circle_dataset_100x100",
        "data_file": "circle_dataset_100x100_v1.npz",
        "network_dir": "/home/student/Desktop/Groundeep/networks/uniform/idbn_binary"
    },
    "zipfian": {
        "data_path": "/home/student/Desktop/Groundeep/circle_dataset_100x100",
        "data_file": "circle_dataset_100x100_v1.npz",
        "network_dir": "/home/student/Desktop/Groundeep/networks/zipfian/idbn_binary"
    }
}

# Define a base output directory for all analyses
global_output_base_dir = "/home/student/Desktop/Groundeep/outputs/analysis_results/" 
os.makedirs(global_output_base_dir, exist_ok=True)

# === Init WandB ===
run = wandb.init(project="groundeep-visualization")

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
        
        features = {
            "N_list": np.array(output_dict.get(f'labels_{dist_name}', [])),
            "cumArea": np.array(output_dict.get(f'cumArea_{dist_name}', [])),
            "FA": np.array(output_dict.get(f'FA_{dist_name}', [])), # Assicurati che FA esista qui!
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
            correlations = visualizer.plot_2d_embedding_and_correlations(emb_2d, features, arch_name, dist_name, method_name="PCA")
            
            correlations = visualizer.plot_2d_embedding_and_correlations(emb_2d, features, arch_name, dist_name, method_name="PCA")
            
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
        visualizer.rsa_analysis(arch_name, dist_name, embedding_analyzer=embedding_analyzer, metric="cosine")
        visualizer.mse_analysis(arch_name, embedding_analyzer=embedding_analyzer)
        visualizer.afp_analysis(arch_name, embedding_analyzer=embedding_analyzer)
        visualizer.ssim_analysis(arch_name, embedding_analyzer=embedding_analyzer)

        # === Log reconstructions to WandB ===
        if original_inputs is not None and reconstructed is not None and len(original_inputs) > 0:
            log_reconstructions_to_wandb(original_inputs[:10], reconstructed[:10], name=arch_name)
        else:
            print(f"Skipping reconstruction logging for {arch_name}: original_inputs or reconstructed are empty.")

    print(f"✅ Finished processing all architectures for {dist_name} distribution.")

# === After all loops are complete, generate combined plots and reports ===
# Questi metodi ora accederanno a self.all_rsa_results che contiene i dati di ENTRAMBE le distribuzioni
print("\n🔄 Generating combined RSA barplots (and boxplot)...")
visualizer.plot_combined_rsa_barplots() 

print("📝 Generating LaTeX report data...")
visualizer.generate_latex_report_data()

# Save the aggregated correlation dataframe
if all_correlations:
    df_corr_all = pd.DataFrame(all_correlations)
    df_corr_all.to_excel(os.path.join(global_output_base_dir, "all_architectures_correlations.xlsx"), index=False)
    print(f"📊 Aggregated correlations saved to: {os.path.join(global_output_base_dir, 'all_architectures_correlations.xlsx')}")
else:
    print("No correlations data collected to save to Excel.")

print("\n🎉 All analyses complete and logged to WandB.")
wandb.finish()