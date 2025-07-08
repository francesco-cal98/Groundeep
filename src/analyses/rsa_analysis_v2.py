import os, sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import wandb
from glob import glob
from scipy.stats import ttest_rel

from rsatoolbox.data.dataset import Dataset
from rsatoolbox.rdm.calc import calc_rdm
from rsatoolbox.model import ModelFixed, ModelWeighted
from rsatoolbox.inference.evaluate import eval_fixed
from rsatoolbox.vis.rdm_plot import show_rdm

# Setup paths
current_dir = os.getcwd()
project_root = os.path.abspath(os.path.join(current_dir, '..', '..'))
sys.path.append(project_root)
sys.path.append(current_dir)

from src.analyses.embedding_analysis import Embedding_analysis

def infer_n_pattern(rdm):
    n_dissim = rdm.dissimilarities.shape[1]
    n = (1 + np.sqrt(1 + 8 * n_dissim)) / 2
    return int(n)

# Config
configs = {
    "uniform": {
        "data_path": "/home/student/Desktop/Groundeep/circle_dataset_100x100/",
        "data_file": "circle_dataset_100x100_v2.npz",
        "network_dir": "/home/student/Desktop/Groundeep/networks/uniform/idbn_new_dataset"
    },
    "zipfian": {
        "data_path": "/home/student/Desktop/Groundeep/circle_dataset_100x100/",
        "data_file": "circle_dataset_100x100_v2.npz",
        "network_dir": "/home/student/Desktop/Groundeep/networks/zipfian/idbn_400_runs"
    }
}

wandb.init(project="rsa_analysis", name="embedding_rsa", config=configs)
output_base = "/home/student/Desktop/Groundeep/outputs/rsa_analysis_rsatoolbox"
os.makedirs(output_base, exist_ok=True)

rsa_results, regression_results = [], []
model_rhos_by_arch = {}

for dist_name, cfg in configs.items():
    for pkl_path in glob(os.path.join(cfg["network_dir"], "*.pkl")):
        arch = os.path.splitext(os.path.basename(pkl_path))[0]
        analyser = Embedding_analysis(cfg["data_path"], cfg["data_file"],
                                       pkl_path, pkl_path, arch)
        out = analyser._get_encodings()

        emb = np.array(out[f'Z_{dist_name}'], float)
        nums = np.array(out[f'labels_{dist_name}'])
        cumA = np.array(out[f'cumArea_{dist_name}'])
        FA = np.array(out[f'FA_{dist_name}'])
        CH = np.array(out[f'CH_{dist_name}'])

        ds = Dataset(emb)
        brain_rdm = calc_rdm(ds, method='euclidean')

        features = {
            'log':      np.log1p(nums),
            'sqrt':     np.sqrt(nums),
            'linear':   nums.astype(float),
            'cumArea':  cumA,
            'FA':       FA,
            'CH':       CH
        }

        models_fixed = []
        for name, feat in features.items():
            ds_mod = Dataset(feat.reshape(-1, 1))
            rdm_mod = calc_rdm(ds_mod, method='euclidean')
            models_fixed.append(ModelFixed(name, rdm_mod))

        model_rhos_by_arch[(dist_name, arch)] = {}

        res_fix = eval_fixed(models_fixed, brain_rdm, method='spearman')
        for model, score in zip(res_fix.models, res_fix.evaluations.squeeze()):
            model_rhos_by_arch[(dist_name, arch)][model.name] = score
            rsa_results.append({
                'arch': arch, 'distribution': dist_name, 'model': model.name, 'rho': score
            })            
            wandb.log({f"{dist_name}/{arch}/rsa/{model.name}": score})

        # Weighted regression (without bootstrap)
        model_w = ModelWeighted("weighted", models_fixed)
        model_w.fit(brain_rdm)
        weights = model_w.weights.squeeze()
        for i, name in enumerate(model_w.model_names):
            result = {
                'arch': arch, 'distribution': dist_name, 'model': name,
                'weight': weights[i]
            }
            regression_results.append(result)
            wandb.log({f"{dist_name}/{arch}/weight/{name}": weights[i]})

        # Plot RDMs
        plot_dir = os.path.join(output_base, dist_name, 'rdm_plots')
        os.makedirs(plot_dir, exist_ok=True)
        show_rdm(brain_rdm, fname=os.path.join(plot_dir, f"{arch}_brain.png"))
        wandb.log({"rdm_brain": wandb.Image(os.path.join(plot_dir, f"{arch}_brain.png"))})

        for m in models_fixed:
            rdm_path = os.path.join(plot_dir, f"{arch}_{m.name}.png")
            show_rdm(m.rdm, fname=rdm_path)
            wandb.log({f"rdm_{m.name}": wandb.Image(rdm_path)})

# Save results
rsa_df = pd.DataFrame(rsa_results)
rsa_df.to_excel(os.path.join(output_base, "rsa_fixed.xlsx"), index=False)
weights_df = pd.DataFrame(regression_results)
weights_df.to_excel(os.path.join(output_base, "rsa_weights.xlsx"), index=False)

# Heatmap RSA
pivot = rsa_df.pivot_table(index='model', columns='distribution', values='rho', aggfunc='mean')
sns.heatmap(pivot, annot=True, cmap='coolwarm')
plt.title("Mean RSA Correlation per Model and Distribution")
plt.tight_layout()
heatmap_path = os.path.join(output_base, "heatmap_rsa.png")
plt.savefig(heatmap_path)
wandb.log({"heatmap_rsa": wandb.Image(heatmap_path)})

# Weighted RSA Weights
for dist in weights_df['distribution'].unique():
    plt.figure(figsize=(10, 5))
    sub = weights_df[weights_df['distribution'] == dist]
    for arch in sub['arch'].unique():
        sub_arch = sub[sub['arch'] == arch]
        plt.bar(sub_arch['model'], sub_arch['weight'], alpha=0.6, label=arch)
    plt.title(f"Weighted RSA - Weights - {dist}")
    plt.xlabel("Model Feature")
    plt.ylabel("Weight")
    plt.legend()
    plt.tight_layout()
    fname = os.path.join(output_base, f"weights_{dist}.png")
    plt.savefig(fname)
    wandb.log({f"weights_{dist}": wandb.Image(fname)})

# Model comparison (t-test)
comparison_results = []
models = list(features.keys())
for i in range(len(models)):
    for j in range(i+1, len(models)):
        m1, m2 = models[i], models[j]
        rhos_1, rhos_2 = [], []
        for key, vals in model_rhos_by_arch.items():
            if m1 in vals and m2 in vals:
                rhos_1.append(vals[m1])
                rhos_2.append(vals[m2])
        if rhos_1 and rhos_2:
            stat, pval = ttest_rel(rhos_1, rhos_2)
            comparison_results.append({
                'model_1': m1, 'model_2': m2, 't_stat': stat, 'p_value': pval
            })
comparison_df = pd.DataFrame(comparison_results)
comparison_df.to_excel(os.path.join(output_base, "model_comparisons.xlsx"), index=False)
wandb.log({"model_comparisons": wandb.Table(dataframe=comparison_df)})

print("✅ Analisi RSA completata senza bootstrap, con W&B e visualizzazioni!")
