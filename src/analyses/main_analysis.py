# main.py
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
sys.path.append(current_dir)

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

# === Init WandB ===
wandb.init(project="groundeep-visualization-2",)

# === Main loop ===
for dist_name, cfg in configs.items():
    print(f"\n📊 Processing distribution: {dist_name.upper()}")

    data_path = cfg["data_path"]
    data_file = cfg["data_file"]
    network_dir = cfg["network_dir"]
    output_dir = f"/home/student/Desktop/Groundeep/outputs/images_combined/new_dataset/2D_UMAP/{dist_name}/"
    os.makedirs(output_dir, exist_ok=True)

    pkl_files = glob(os.path.join(network_dir, "*.pkl"))
    correlations_list = []

    for pkl_path in pkl_files:
        arch_name = os.path.splitext(os.path.basename(pkl_path))[0]
        print(f" - Analyzing architecture: {arch_name}")

        # === Load model and compute encodings ===
        analyser = Embedding_analysis(data_path, data_file, pkl_path, pkl_path, arch_name)
        output_dict = analyser._get_encodings()
        
        # get reconstructed inputs from DBN 
        original_inputs,reconstructed = analyser.reconstruct_input(analyser.inputs_uniform)
        visualizer = VisualizerWithLogging(wandb.run, output_dir, analyser)

        embeddings = np.array(output_dict[f'Z_{dist_name}'], dtype=np.float64)
        features = {
            "N_list": np.array(output_dict[f'labels_{dist_name}']),
            "cumArea": np.array(output_dict[f'cumArea_{dist_name}']),
            "CH": np.array(output_dict[f'CH_{dist_name}'])
        }

        # === Plot correlation matrix ===
        visualizer.plot_feature_correlation_matrix(features, arch_name, dist_name)

        # === Plot 2D UMAP and correlations ===
        emb_2d = visualizer.reduce_dimensions(embeddings, method='pca')
        correlations = visualizer.plot_2d_embedding_and_correlations(emb_2d, features, arch_name, dist_name, method_name="umap")

        # === RSA analysis ===
        visualizer.rsa_analysis(arch_name, dist_name, metric="cosine")

        # === MSE analysis ===
        visualizer.mse_analysis(arch_name)

        # === AFP analysis ===
        visualizer.afp_analysis(arch_name)

        # === SSIM analysis ===
        visualizer.ssim_analysis(arch_name)


        # === Store correlations ===
        for (feat_name, dim_label), corr_val in correlations.items():
            correlations_list.append({
                'arch': arch_name,
                'feature': feat_name,
                'dimension': dim_label,
                'correlation': corr_val,
                'distribution': dist_name
            })

        # === Log reconstructions to WandB ===
        log_reconstructions_to_wandb(original_inputs[:10], reconstructed[:10], name=arch_name)


    # Save correlation dataframe
    df_corr = pd.DataFrame(correlations_list)
    df_corr.to_excel(os.path.join(output_dir, f"correlations_{dist_name}.xlsx"), index=False)
    print(f"✅ Finished {dist_name} — saved to: {output_dir}")
    


print("\n🎉 All visualizations complete.")
wandb.finish()
