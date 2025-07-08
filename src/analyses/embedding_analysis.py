import torch
import pickle as pkl
import os 
import sys
import gc

from scipy.stats import spearmanr
from statsmodels.stats.multitest import fdrcorrection

current_dir = os.getcwd()
sys.path.append(current_dir)  # Add the project root to sys.path

from src.datasets.uniform_dataset import create_dataloaders_uniform,create_dataloaders_zipfian
from torch.utils.data import DataLoader
from sklearn.metrics import pairwise_distances
from itertools import product
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np 
from scipy.stats import spearmanr
from statsmodels.stats.multitest import multipletests
from sklearn.metrics.pairwise import cosine_similarity
from scipy.spatial import distance
from scipy.stats import spearmanr
from statsmodels.stats.multitest import fdrcorrection
import numpy as np
import pandas as pd




class Embedding_analysis:

    def __init__(self,path2data,data_name,model_uniform,model_zipfian,arch_name,val_size = 0.01):
        
        self.path2data = path2data
        self.data_name = data_name
        self.arch_name = arch_name

        _,self.val_dataloader_uniform,_ = create_dataloaders_uniform(self.path2data,self.data_name,batch_size = 128,val_size = val_size)
        
        _,self.val_dataloader_zipfian,_ = create_dataloaders_zipfian(self.path2data,self.data_name,batch_size = 128,val_size = val_size)
        
        # Override batch size for validation loader
        self.val_dataloader_uniform = DataLoader(
            self.val_dataloader_uniform.dataset, batch_size=len(self.val_dataloader_uniform.dataset), shuffle=False
        )

        self.val_dataloader_zipfian = DataLoader(
            self.val_dataloader_zipfian.dataset, batch_size=len(self.val_dataloader_zipfian.dataset), shuffle=False
        )

        # Subset object (e.g., torch.utils.data.Subset) – needed to access selected indices
        self.dataset_uniform_subset = self.val_dataloader_uniform.dataset

        # The original dataset (e.g., UniformDataset) – needed to access the feature lists
        self.dataset_uniform = self.dataset_uniform_subset.dataset


        self.dataset_zipfian_subset = self.val_dataloader_zipfian.dataset
        self.dataset_zipfian = self.dataset_zipfian_subset.dataset

        with open(model_uniform, 'rb') as f:
            self.model_uniform = pkl.load(f)

        with open(model_zipfian, 'rb') as f:
            self.model_zipfian = pkl.load(f)
    
    def _get_encodings(self):
        import gc
        import torch
        import pandas as pd

        if torch.cuda.is_available():
            device = torch.device('cuda:0')
        else:
            device = torch.device('cpu')

        self.output_dict = {}

        # === Estrai batch uniforme (intero dataset) ===
        batch_uniform = next(iter(self.val_dataloader_uniform))
        inputs_uniform, self.labels_uniform = batch_uniform
        self.inputs_uniform = inputs_uniform.to(device)

        with torch.no_grad():
            h1_probs_uniform = self.model_uniform.layers[0].forward(self.inputs_uniform)
            h2_probs_uniform = self.model_uniform.layers[1].forward(h1_probs_uniform)

            h1_probs_zipfian = self.model_zipfian.layers[0].forward(self.inputs_uniform)
            h2_probs_zipfian = self.model_zipfian.layers[1].forward(h1_probs_zipfian)

        Z_uniform = h2_probs_uniform.cpu().numpy()
        Z_zipfian = h2_probs_zipfian.cpu().numpy()

        #del self.model_uniform, self.model_zipfian
        self.inputs_uniform = inputs_uniform.detach().cpu()  # Store a copy for reconstruction analysis

        gc.collect()
        torch.cuda.empty_cache()

        self.output_dict['Z_uniform'] = Z_uniform
        self.output_dict['Z_zipfian'] = Z_zipfian
        self.output_dict['labels_uniform'] = self.labels_uniform.cpu().numpy()
        self.output_dict['labels_zipfian'] = self.labels_uniform.cpu().numpy()
        numerosity_bins_uniform = pd.cut(
        self.output_dict['labels_uniform'],
            bins=[0, 4, 8, 12, 16, 20, 24, 28,32],  # scegli i bin come preferisci
            labels=["1–4", "5–8", "9–12", "13–16", "17–20", "21–24", "25–28", "29–32"],
        )
        self.output_dict['numerosity_bin_uniform'] = numerosity_bins_uniform
        # === Uniform features ===
        indices_uniform = self.dataset_uniform_subset.indices
        cumArea_vals_uniform = [self.dataset_uniform.cumArea_list[i] for i in indices_uniform]
        convex_hull_uniform = [self.dataset_uniform.CH_list[i] for i in indices_uniform]
        items_uniform = [self.dataset_uniform.Items_list[i] for i in indices_uniform]
        self.output_dict['FA_uniform'] = [self.dataset_uniform.FA_list[i] for i in indices_uniform]
        self.output_dict['CH_uniform'] = [self.dataset_uniform.CH_list[i] for i in indices_uniform]
        self.output_dict['cumArea_uniform'] = cumArea_vals_uniform
        self.output_dict['Items_uniform'] = items_uniform
        # generating binned features
        cumArea_quantiles_uniform, cumArea_bins_uniform = pd.qcut(
            cumArea_vals_uniform, q=8, labels=False, retbins=True, duplicates='drop'
        )
        convex_hull_quantiles_uniform,  convex_hull_bins_uniform = pd.qcut(
            convex_hull_uniform, q=8, labels=False, retbins=True, duplicates='drop'
        )
        Items_quantiles_uniform, Items_bins_uniform = pd.qcut(
            items_uniform, q=8, labels=False, retbins=True, duplicates='drop'
        )
        
        self.output_dict['convex_hull_uniform'] = convex_hull_quantiles_uniform
        self.output_dict['convex_hull_bins_uniform'] = convex_hull_bins_uniform
        self.output_dict['cumArea_uniform'] = cumArea_quantiles_uniform
        self.output_dict['cumArea_bins_uniform'] = cumArea_bins_uniform
        self.output_dict['Items_uniform'] = Items_quantiles_uniform
        self.output_dict['Items_bins_uniform'] = Items_bins_uniform

        # === Zipfian features ===
        indices_zipfian = self.dataset_zipfian_subset.indices
        cumArea_vals_zipfian = [self.dataset_zipfian.cumArea_list[i] for i in indices_zipfian]
        self.output_dict['FA_zipfian'] = [self.dataset_zipfian.FA_list[i] for i in indices_zipfian]
        self.output_dict['CH_zipfian'] = [self.dataset_zipfian.CH_list[i] for i in indices_zipfian]

        cumArea_quantiles_zipfian, cumArea_bins_zipfian = pd.qcut(
            cumArea_vals_zipfian, q=10, labels=False, retbins=True, duplicates='drop'
        )
        self.output_dict['cumArea_zipfian'] = cumArea_quantiles_zipfian
        self.output_dict['cumArea_bins_zipfian'] = cumArea_bins_zipfian

        return self.output_dict
    


    def reconstruct_input(self, input_type="uniform"):
        """
        Ricostruisce gli input a partire dai codici latenti del modello specificato.
        input_type: 'uniform' o 'zipfian'
        """

        if torch.cuda.is_available():
            device = torch.device('cuda:0')
        else:
            device = torch.device('cpu')

        dataloader = self.val_dataloader_uniform 
        model = self.model_uniform if input_type == "uniform" else self.model_zipfian

        batch = next(iter(dataloader))
        inputs, _ = batch
        inputs = inputs.to(device)

        with torch.no_grad():
            temp_data = inputs
            for rbm in model.layers:
                temp_data = rbm.forward(temp_data)
            for rbm in reversed(model.layers):
                temp_data = rbm.backward(temp_data)
        
        reconstruction = temp_data.cpu().numpy()
        original = inputs.cpu().numpy()
        return original, reconstruction


    def get_classwise_distance_matrices(self, Z1, Z2, labels, metric='euclidean'):
        """
        Computes class-to-class average distance matrices for two embedding spaces (Z1 and Z2).
        
        Parameters:
            Z1, Z2: Embeddings (n_samples x n_features)
            labels: Class labels (n_samples,)
            metric: 'euclidean' or 'mahalanobis'
        
        Returns:
            A1, A2: Class-to-class average distance matrices (n_classes x n_classes)
        """
        
        if metric == 'mahalanobis':
            # Compute inverse covariance matrices
            VI1 = np.linalg.pinv(np.cov(Z1.T))
            VI2 = np.linalg.pinv(np.cov(Z2.T))
            
            # Compute pairwise Mahalanobis distance matrices manually
            n = Z1.shape[0]
            dist1 = np.zeros((n, n))
            dist2 = np.zeros((n, n))
            for i in range(n):
                for j in range(n):
                    dist1[i, j] = distance.mahalanobis(Z1[i], Z1[j], VI1)
                    dist2[i, j] = distance.mahalanobis(Z2[i], Z2[j], VI2)
        else:
            # Default to Euclidean
            dist1 = pairwise_distances(Z1, metric=metric, n_jobs=20)
            dist2 = pairwise_distances(Z2, metric=metric, n_jobs=20)

        # Compute class-to-class average distances
        unique_labels = np.unique(labels)
        n_classes = len(unique_labels)
        A1 = np.empty((n_classes, n_classes))
        A2 = np.empty((n_classes, n_classes))

        for i_idx, i in enumerate(unique_labels):
            for j_idx, j in enumerate(unique_labels):
                if i == j:
                    idx = np.where(labels == i)[0]
                    sel = [(u, v) for u, v in product(idx, idx) if u != v]
                else:
                    idx_i = np.where(labels == i)[0]
                    idx_j = np.where(labels == j)[0]
                    sel = list(product(idx_i, idx_j))
                if sel:
                    values1 = [dist1[u, v] for u, v in sel]
                    values2 = [dist2[u, v] for u, v in sel]
                    A1[i_idx, j_idx] = np.mean(values1)
                    A2[i_idx, j_idx] = np.mean(values2)
                else:
                    A1[i_idx, j_idx] = np.nan
                    A2[i_idx, j_idx] = np.nan

        return A1, A2


    def get_class_dist_matrix(self,Z_dist, Z_base, labels):
        dist1 = pairwise_distances(Z_dist, metric='euclidean', n_jobs=20)
        dist2 = pairwise_distances(Z_base, metric='euclidean', n_jobs=20)
        ro = np.divide(dist1, dist2)
        ro[np.isnan(ro)] = 1

        ro = (ro - ro.mean())  # Optionally normalize further

        unique_labels = np.unique(labels)
        n = len(unique_labels)
        A = np.empty((n, n))

        for i_idx, i in enumerate(unique_labels):
            for j_idx, j in enumerate(unique_labels):
                if i == j:
                    idx = np.where(labels == i)[0]
                    sel = [(u, v) for u, v in product(idx, idx) if u != v]
                else:
                    idx_i = np.where(labels == i)[0]
                    idx_j = np.where(labels == j)[0]
                    sel = list(product(idx_i, idx_j))

                l = [ro[u, v] for u, v in sel]
                A[i_idx, j_idx] = np.median(l) if l else np.nan

        return A
    
    def compute_classwise_selectivity_with_fdr(self, source='uniform', alpha=0.05, method='fdr_bh'):
        """
        Per ogni classe, calcola la correlazione di Spearman (ρ) tra ciascun neurone e una maschera binaria (one-vs-all),
        poi applica una correzione FDR globale su tutti i test neurone-classe.

        Args:
            source (str): 'uniform' o 'zipfian'
            alpha (float): livello di significatività per la FDR
            method (str): metodo di correzione FDR (default: 'fdr_bh')

        Returns:
            spearman_rho_df: DataFrame [classi x neuroni] con i valori di ρ
            pval_corrected_df: p-values corretti per FDR (stessa forma)
            significant_df: boolean mask con True per i test significativi (stessa forma)
        """


        if not hasattr(self, 'output_dict'):
            raise RuntimeError("Run _get_encodings() first.")

        Z = self.output_dict[f'Z_{source}']          # shape: (n_samples, n_neurons)
        labels = self.output_dict[f'labels_{source}']
        num_neurons = Z.shape[1]
        classes = np.unique(labels)

        rhos = []
        pvals = []

        for cls in classes:
            binary_labels = (labels == cls).astype(int)
            class_rhos = []
            class_pvals = []
            for i in range(num_neurons):
                rho, p = spearmanr(Z[:, i], binary_labels)
                class_rhos.append(rho)
                class_pvals.append(p)
            rhos.append(class_rhos)
            pvals.append(class_pvals)

        spearman_rho_df = pd.DataFrame(rhos, index=classes)
        pval_df = pd.DataFrame(pvals, index=classes)
        spearman_rho_df.index.name = 'numerosity'
        pval_df.index.name = 'numerosity'

        # Flatten all p-values for FDR correction globale
        pval_array = pval_df.values.flatten()
        rejected, pvals_corrected, _, _ = multipletests(pval_array, alpha=alpha, method=method)

        # Reshape back
        pval_corrected_df = pd.DataFrame(
            pvals_corrected.reshape(pval_df.shape),
            index=pval_df.index,
            columns=pval_df.columns
        )
        significant_df = pd.DataFrame(
            rejected.reshape(pval_df.shape),
            index=pval_df.index,
            columns=pval_df.columns
        )

        return spearman_rho_df, pval_corrected_df, significant_df


