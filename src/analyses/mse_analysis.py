import os
import sys
import numpy as np
import pandas as pd
import torch
from glob import glob
import wandb
from tqdm import tqdm
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
import statsmodels.api as sm


# === PATH SETUP ===
current_dir = os.getcwd()
project_root = os.path.abspath(os.path.join(current_dir, '..', '..'))
sys.path.append(project_root)
sys.path.append(current_dir)

from src.analyses.embedding_analysis import Embedding_analysis
from src.utils.wandb_utils import log_reconstructions_to_wandb

# === CONFIG ===
config = {
    "data_path": "/home/student/Desktop/Groundeep/circle_dataset_100x100/",
    "data_file": "circle_dataset_100x100_v2.npz",
    "network_dir": "/home/student/Desktop/Groundeep/networks/uniform/idbn_new_dataset",  # change for zipfian
    "distribution": "uniform"
}

# === WANDB INIT ===
wandb.init(project="groundeep-reconstruction-mse", config=config)
wandb_config = wandb.config

pkl_files = sorted(glob(os.path.join(config["network_dir"], "*.pkl")))

for model_path in tqdm(pkl_files, desc="Processing architectures"):
    arch_name = os.path.splitext(os.path.basename(model_path))[0]

    analyser = Embedding_analysis(
        config["data_path"],
        config["data_file"],
        model_path,
        model_path,
        arch_name,
        val_size=0.05
    )

    # === Get encodings and features ===
    out = analyser._get_encodings()
    #original_inputs = analyser.inputs_uniform.cpu().numpy()
    original_inputs,reconstructed = analyser.reconstruct_input(analyser.inputs_uniform)

    numerosities = out['labels_uniform']
    numerosities_bin = out['numerosity_bin_uniform']
    cumarea_bins = out['cumArea_bins_uniform']
    cumarea_bin_ids = out['cumArea_uniform']
    convex_hull_bins = out['convex_hull_bins_uniform']
    convex_hull_bin = out['convex_hull_uniform']

    # === Compute MSE per sample ===
    mses = np.mean((original_inputs - reconstructed) ** 2,axis =1 )  # shape: (N,)

    # === Build DataFrame ===
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

    pivot_hull = df.groupby(["convex_hull_bin", "numerosity"])["mse"].mean().unstack(fill_value=np.nan)
    pivot_hull = pivot_hull.sort_index(ascending=False)

    plt.figure(figsize=(10, 6))
    sns.heatmap(pivot_hull, annot=True, fmt=".3f", cmap="viridis")
    plt.title(f"MSE Heatmap – Convex Hull vs Numerosity – {arch_name}")
    plt.xlabel("Numerosity_bin")
    plt.ylabel("Convex Hull bin")
    plt.tight_layout()

    wandb.log({
        f"{arch_name}/mse_heatmap_convex_hull": wandb.Image(plt.gcf())
    })
    plt.close()

    # === Log results ===
    wandb.log({
        f"{arch_name}/mse_heatmap": wandb.Image(plt.gcf()),
        f"{arch_name}/mean_mse": df["mse"].mean()
    })
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

    # === Build regression DataFrame with continuous values ===
    reg_df = pd.DataFrame({
        "numerosity": out["labels_uniform"],
        "cumulative_area": out["cumArea_uniform"],
        "convex_hull": out["convex_hull_uniform"],
        "mse": mses
    })

    X = reg_df[["numerosity", "cumulative_area", "convex_hull"]]
    y = reg_df["mse"]


    # Fit regression model
    X_const = sm.add_constant(X)  # Add constant term for intercept
    model = sm.OLS(y, X_const).fit()

    # Log results
    # === Estrai risultati principali
    coeffs = model.params
    pvals = model.pvalues
    conf_int = model.conf_int()
    summary_text = model.summary().as_text()

    # === Logga i risultati come tabella su wandb
    wandb.log({
        f"{arch_name}/regression_coefficients": wandb.Table(
            columns=["Variable", "Coef", "P-value", "CI_lower", "CI_upper"],
            data=[
                [var, float(coeffs[var]), float(pvals[var]), float(conf_int.loc[var][0]), float(conf_int.loc[var][1])]
                for var in coeffs.index
            ]
        ),
        f"{arch_name}/regression_summary_text": wandb.Html(f"<pre>{summary_text}</pre>")
    })
  


    # Optional: log sample reconstructions
    log_reconstructions_to_wandb(original_inputs[:10], reconstructed[:10], name=arch_name)
wandb.finish()
