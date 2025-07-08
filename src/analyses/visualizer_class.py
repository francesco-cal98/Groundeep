# visualizer_with_logging.py
import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.manifold import MDS
from sklearn.metrics import pairwise_distances, mean_squared_error
from scipy.stats import spearmanr
import umap
import wandb
from itertools import product
from scipy.spatial.distance import pdist
from sklearn.linear_model import LinearRegression
from scipy.fftpack import fft2
import statsmodels.api as sm
from skimage.metrics import structural_similarity as ssim



class VisualizerWithLogging:

    def __init__(self, wandb_run, output_dir, embedding_analyzer):
        self.wandb_run = wandb_run
        self.output_dir = output_dir
        self.embedding_analyzer = embedding_analyzer
        
        os.makedirs(self.output_dir, exist_ok=True)

    def reduce_dimensions(self, data, method='umap', **kwargs):
        if method == 'umap':
            reducer = umap.UMAP(n_components=2, random_state=42, **kwargs)
        elif method == 'pca':
            reducer = PCA(n_components=2, **kwargs)
        elif method == 'mds':
            reducer = MDS(n_components=2, random_state=42, **kwargs)
        else:
            raise ValueError(f"Unsupported reduction method: {method}")
        return reducer.fit_transform(data)

    def plot_feature_correlation_matrix(self, features_dict, arch_name, dist_name):
        df = pd.DataFrame(features_dict)
        corr_matrix = df.corr()

        plt.figure(figsize=(8, 6))
        sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap="coolwarm", square=True, cbar=True)
        plt.title(f"{arch_name} – {dist_name} – Feature Correlation Matrix")
        plt.tight_layout()

        save_path = os.path.join(self.output_dir, f"{arch_name}_correlation_matrix.jpg")
        plt.savefig(save_path, dpi=300)
        self.wandb_run.log({f"{arch_name}/{dist_name}/correlation_matrix": wandb.Image(plt.gcf())})
        plt.close()

    def plot_2d_embedding_and_correlations(self, emb_2d, features_dict, arch_name, dist_name, method_name='UMAP'):
        fig, axs = plt.subplots(4, 3, figsize=(18, 16))
        axs = axs.flatten()

        numerosity = features_dict["N_list"]
        sc = axs[0].scatter(emb_2d[:, 0], emb_2d[:, 1], c=np.log(numerosity), cmap="viridis", s=40)
        axs[0].set_title("2D Embedding – colored by Numerosity")
        axs[0].set_xlabel("Dim-1")
        axs[0].set_ylabel("Dim-2")
        plt.colorbar(sc, ax=axs[0], label="Numerosity")

        plot_idx = 1
        correlations = {}
        for feat_name, feat_values in features_dict.items():
            for i, dim_label in enumerate(["Dim-1", "Dim-2"]):
                dim = emb_2d[:, i]
                corr, _ = spearmanr(dim, feat_values)
                correlations[(feat_name, dim_label)] = corr
                axs[plot_idx].scatter(dim, feat_values, alpha=0.6)
                axs[plot_idx].set_title(f"{feat_name} vs {dim_label} (ρ = {corr:.2f})")
                axs[plot_idx].set_xlabel(dim_label)
                axs[plot_idx].set_ylabel(feat_name)
                plot_idx += 1

        for j in range(plot_idx, len(axs)):
            fig.delaxes(axs[j])

        fig.suptitle(f"{arch_name} – {dist_name} – 2D {method_name} + Feature Correlations", fontsize=18)
        fig.tight_layout(rect=[0, 0.03, 1, 0.95])

        plot_path = os.path.join(self.output_dir, f"{arch_name}_{method_name.lower()}_2d_plot.jpg")
        plt.savefig(plot_path, dpi=300)
        self.wandb_run.log({f"{arch_name}/{dist_name}/2d_plot_{method_name.lower()}": wandb.Image(fig)})
        plt.close(fig)
        return correlations

    def rsa_analysis(self, arch_name, dist_name, metric="cosine"):
        output = self.embedding_analyzer.output_dict
        embeddings = output[f'Z_{dist_name}']
        labels = output[f'labels_{dist_name}']
        cumArea = np.array(output[f'cumArea_{dist_name}'])
        FA = np.array(output[f'FA_{dist_name}'])
        CH = np.array(output[f'CH_{dist_name}'])
        self.rsa_result_list = []

        def compute_brain_rdm(emb):
            if metric == "cosine":
                normed = emb / np.linalg.norm(emb, axis=1, keepdims=True)
                return pdist(normed, metric='cosine')
            elif metric == "euclidean":
                return pdist(emb, metric='euclidean')
            else:
                raise ValueError("Unsupported metric")

        def compute_model_rdm(values):
            return pdist(values.reshape(-1, 1), metric='euclidean')

        brain_rdm = compute_brain_rdm(embeddings)

        features = {
            "log": np.log1p(labels),
            "cumArea": cumArea,
            "FA": FA,
            "CH": CH
        }

        model_rdms = {k: compute_model_rdm(v.astype(np.float64)) for k, v in features.items()}

        rsa_results = {}
        for name, model_rdm in model_rdms.items():
            rho, _ = spearmanr(brain_rdm, model_rdm)
            rsa_results[name] = rho
            self.rsa_result_list.append({
                "arch": arch_name,
                "distribution": dist_name,
                "encoding": name,
                "spearman_rho": rho
            })

        # Boxplot logging
        df_rsa = pd.DataFrame(self.rsa_result_list)
        plt.figure(figsize=(8, 6))
        sns.boxplot(data=df_rsa, x="encoding", y="spearman_rho", hue="distribution")
        plt.title("RSA Correlation Distribution")
        plt.xlabel("Encoding")
        plt.ylabel("Spearman Rho")
        plt.tight_layout()
        self.wandb_run.log({"rsa/boxplot_rsa_correlations": wandb.Image(plt.gcf())})
        plt.close()

        # Regressione lineare
        model_names = list(features.keys())
        X = np.stack([model_rdms[n] for n in model_names], axis=1)
        y = brain_rdm
        reg = LinearRegression().fit(X, y)


        return rsa_results

    def mse_analysis(self, arch_name):
        original_inputs, reconstructed = self.embedding_analyzer.reconstruct_input()
        out = self.embedding_analyzer.output_dict

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



    def afp_analysis(self, arch_name):
        original_inputs, reconstructed = self.embedding_analyzer.reconstruct_input()
        out = self.embedding_analyzer.output_dict

        # === Feature bins ===
        numerosities = out['labels_uniform']
        numerosities_bin = out['numerosity_bin_uniform']
        cumarea_bins = out['cumArea_bins_uniform']
        cumarea = out['cumArea_uniform']
        convex_hull_bins = out['convex_hull_bins_uniform']
        convex_hull = out['convex_hull_uniform']
        items_uniform = out['Items_uniform']
        items_uniform_bin = out['Items_bins_uniform']


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
            "items_uniform": items_uniform
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

    def ssim_analysis(self, arch_name):

        original_inputs, reconstructed = self.embedding_analyzer.reconstruct_input()
        out = self.embedding_analyzer.output_dict

        numerosities = out['labels_uniform']
        cumarea = out['cumArea_uniform']
        convex_hull = out['convex_hull_uniform']
        items_uniform = out['Items_uniform']

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
            "items_uniform": items_uniform
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
