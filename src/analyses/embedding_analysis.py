import os
import sys
import gc
from pathlib import Path
import pickle as pkl
from typing import Dict, Any, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.datasets.uniform_dataset import create_dataloaders_uniform, create_dataloaders_zipfian  # noqa


class Embedding_analysis:
    """
    Estrae embeddings e feature in modo coerente per confronto per-stimolo:
    - usa il dataloader 'uniform' come base inputs per entrambi i modelli (uniform & zipfian),
      così Z_* e le feature/labels sono allineate 1:1.
    - (opzionale) puoi aggiungere una seconda vista zipf-on-zipf se servono analisi per-classe non allineate.
    """

    def __init__(self, path2data: str, data_name: str,
                 model_uniform: str, model_zipfian: str, arch_name: str, val_size: float = 0.05):
        self.path2data = path2data
        self.data_name = data_name
        self.arch_name = arch_name

        # Dataloaders
        self.train_dataloader_uniform, self.val_dataloader_uniform, self.test_dataloader_uniform = create_dataloaders_uniform(
            self.path2data, self.data_name, batch_size=128, val_size=val_size
        )
        self.train_dataloader_zipfian, self.val_dataloader_zipfian, self.test_dataloader_zipfian = create_dataloaders_zipfian(
            self.path2data, self.data_name, batch_size=128, val_size=val_size
        )

        # Unico batch (intero val) per semplicità/consistenza
        self.val_dataloader_uniform = DataLoader(
            self.val_dataloader_uniform.dataset, batch_size=len(self.val_dataloader_uniform.dataset), shuffle=False
        )
        self.val_dataloader_zipfian = DataLoader(
            self.val_dataloader_zipfian.dataset, batch_size=len(self.val_dataloader_zipfian.dataset), shuffle=False
        )

        # Accesso a dataset e indici
        self.dataset_uniform_subset = self.val_dataloader_uniform.dataset
        self.dataset_uniform = self.dataset_uniform_subset.dataset

        self.dataset_zipfian_subset = self.val_dataloader_zipfian.dataset
        self.dataset_zipfian = self.dataset_zipfian_subset.dataset

        # Carica modelli
        self.model_uniform = self._load_model(model_uniform)
        self.model_zipfian = self._load_model(model_zipfian)

        self.output_dict: Dict[str, Any] = {}

    @staticmethod
    def _load_model(path: str):
        target_device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        orig_restore = getattr(torch.serialization, "default_restore_location", None)
        if orig_restore is not None:
            torch.serialization.default_restore_location = (
                lambda storage, loc: storage.cuda() if target_device.type == 'cuda' else storage.cpu()
            )
        try:
            with open(path, 'rb') as f:
                m = pkl.load(f)
        finally:
            if orig_restore is not None:
                torch.serialization.default_restore_location = orig_restore
        # accetta dict pickled {"layers": [...]} oppure oggetti
        if isinstance(m, dict) and 'layers' in m:
            class _Wrapper:
                def __init__(self, d):
                    self.layers = d.get('layers', [])
                    self.params = d.get('params', {})

                def decode(self, top):
                    with torch.no_grad():
                        cur = top
                        for rbm in reversed(self.layers):
                            cur = rbm.backward(cur)
                        return cur
            return _Wrapper(m)
        return m

    def _get_encodings(self) -> Dict[str, Any]:
        def _infer_device(model):
            layers = getattr(model, "layers", [])
            for layer in layers:
                for attr in ("W", "hid_bias", "vis_bias", "hbias", "vbias"):
                    tensor = getattr(layer, attr, None)
                    if isinstance(tensor, torch.Tensor):
                        return tensor.device
            return torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

        device_uniform = _infer_device(self.model_uniform)
        device_zipf = _infer_device(self.model_zipfian)
        self.output_dict = {}

        # === Batch UNIFORM (base per-stimolo coerente) ===
        batch_uniform = next(iter(self.val_dataloader_uniform))
        inputs_uniform, self.labels_uniform = batch_uniform
        inputs_uniform = inputs_uniform.to(torch.float32)
        self.inputs_uniform = inputs_uniform.detach().cpu()

        # Forward top-hidden
        with torch.no_grad():
            # Uniform model on uniform inputs
            cur = inputs_uniform.to(device_uniform)
            for rbm in self.model_uniform.layers:
                cur = rbm.forward(cur)
            Z_uniform = cur.detach().cpu().numpy()

            # Zipf model on the SAME uniform inputs
            cur2 = inputs_uniform.to(device_zipf)
            for rbm in self.model_zipfian.layers:
                cur2 = rbm.forward(cur2)
            Z_zipf_on_uniform = cur2.detach().cpu().numpy()

        self.inputs_uniform = inputs_uniform.detach().cpu()

        # Feature/labels coerenti con inputs_uniform
        indices_uniform = self.dataset_uniform_subset.indices
        labels_uniform = self.labels_uniform.cpu().numpy()
        cumArea_uniform = np.array([self.dataset_uniform.cumArea_list[i] for i in indices_uniform])
        CH_uniform = np.array([self.dataset_uniform.CH_list[i] for i in indices_uniform])

        # Salva (stesso set per entrambe le viste)
        self.output_dict['Z_uniform'] = Z_uniform
        self.output_dict['Z_zipfian'] = Z_zipf_on_uniform
        self.output_dict['labels_uniform'] = labels_uniform
        self.output_dict['labels_zipfian'] = labels_uniform

        self.output_dict['cumArea_uniform'] = cumArea_uniform
        self.output_dict['cumArea_zipfian'] = cumArea_uniform

        self.output_dict['CH_uniform'] = CH_uniform
        self.output_dict['CH_zipfian'] = CH_uniform

        gc.collect()
        torch.cuda.empty_cache()

        # === (OPZIONALE) Zipf-on-zipf per analisi non allineate per-stimolo ===
        # batch_zipf = next(iter(self.val_dataloader_zipfian))
        # inputs_zipf, labels_zipf = batch_zipf
        # with torch.no_grad():
        #     cur = inputs_zipf.to(device)
        #     for rbm in self.model_zipfian.layers:
        #         cur = rbm.forward(cur)
        #     Z_zipf_on_zipf = cur.detach().cpu().numpy()
        # indices_zipf = self.dataset_zipfian_subset.indices
        # cumArea_zipf = np.array([self.dataset_zipfian.cumArea_list[i] for i in indices_zipf])
        # CH_zipf = np.array([self.dataset_zipfian.CH_list[i] for i in indices_zipf])
        # self.output_dict['Z_zipf_on_zipf'] = Z_zipf_on_zipf
        # self.output_dict['labels_zipf'] = labels_zipf.numpy()
        # self.output_dict['cumArea_zipf'] = cumArea_zipf
        # self.output_dict['CH_zipf'] = CH_zipf

        return self.output_dict

    def reconstruct_input(self, input_type: str = "uniform") -> Tuple[np.ndarray, np.ndarray]:
        """
        Ricostruisce gli input passando su/giù la pila RBM del modello scelto.
        input_type: 'uniform' | 'zipfian'  (usa gli inputs_uniform come base per semplicità)
        """
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        dataloader = self.val_dataloader_uniform
        model = self.model_uniform if input_type == "uniform" else self.model_zipfian

        batch = next(iter(dataloader))
        inputs, _ = batch
        inputs = inputs.to(device)

        with torch.no_grad():
            cur = inputs
            for rbm in model.layers:
                cur = rbm.forward(cur)
            for rbm in reversed(model.layers):
                cur = rbm.backward(cur)
            reconstruction = cur.cpu().numpy()
        original = inputs.cpu().numpy()
        return original, reconstruction

    # ====== utilità addizionali già presenti nel tuo progetto (lasciate intatte) ======

    def get_classwise_distance_matrices(self, Z1, Z2, labels, metric='euclidean'):
        from itertools import product
        from scipy.spatial import distance
        if metric == 'mahalanobis':
            VI1 = np.linalg.pinv(np.cov(Z1.T))
            VI2 = np.linalg.pinv(np.cov(Z2.T))
            n = Z1.shape[0]
            dist1 = np.zeros((n, n))
            dist2 = np.zeros((n, n))
            for i in range(n):
                for j in range(n):
                    dist1[i, j] = distance.mahalanobis(Z1[i], Z1[j], VI1)
                    dist2[i, j] = distance.mahalanobis(Z2[i], Z2[j], VI2)
        else:
            from sklearn.metrics import pairwise_distances
            dist1 = pairwise_distances(Z1, metric=metric, n_jobs=20)
            dist2 = pairwise_distances(Z2, metric=metric, n_jobs=20)

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

    def get_class_dist_matrix(self, Z_dist, Z_base, labels):
        from itertools import product
        from sklearn.metrics import pairwise_distances
        dist1 = pairwise_distances(Z_dist, metric='euclidean', n_jobs=20)
        dist2 = pairwise_distances(Z_base, metric='euclidean', n_jobs=20)
        ro = np.divide(dist1, dist2)
        ro[np.isnan(ro)] = 1
        ro = (ro - ro.mean())

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
                l = [ro[u, v] for u, v in sel] if sel else []
                A[i_idx, j_idx] = np.median(l) if l else np.nan
        return A

    def compute_classwise_selectivity_with_fdr(self, source='uniform', alpha=0.05, method='fdr_bh'):
        if not hasattr(self, 'output_dict'):
            raise RuntimeError("Run _get_encodings() first.")
        Z = self.output_dict[f'Z_{source}']
        labels = self.output_dict[f'labels_{source}']
        num_neurons = Z.shape[1]
        classes = np.unique(labels)
        from scipy.stats import spearmanr
        from statsmodels.stats.multitest import multipletests

        rhos, pvals = [], []
        for cls in classes:
            binary_labels = (labels == cls).astype(int)
            class_rhos, class_pvals = [], []
            for i in range(num_neurons):
                rho, p = spearmanr(Z[:, i], binary_labels)
                class_rhos.append(rho); class_pvals.append(p)
            rhos.append(class_rhos); pvals.append(class_pvals)

        spearman_rho_df = np.array(rhos)
        pval_df = np.array(pvals)

        flat = pval_df.flatten()
        rejected, pvals_corrected, _, _ = multipletests(flat, alpha=alpha, method=method)
        pval_corrected = pvals_corrected.reshape(pval_df.shape)
        significant = rejected.reshape(pval_df.shape)
        import pandas as pd
        return (
            pd.DataFrame(spearman_rho_df, index=classes),
            pd.DataFrame(pval_corrected, index=classes),
            pd.DataFrame(significant, index=classes),
        )
