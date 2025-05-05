import torch
import pickle as pkl
import os 
import sys
import gc
current_dir = os.getcwd()
sys.path.append(current_dir)  # Add the project root to sys.path

from src.datasets.uniform_dataset import create_dataloaders_uniform,create_dataloaders_zipfian
from torch.utils.data import DataLoader
from sklearn.metrics import pairwise_distances
from itertools import product
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np 


class Embedding_analysis:

    def __init__(self,path2data,data_name,model_uniform,model_zipfian,arch_name):
        
        self.path2data = path2data
        self.data_name = data_name
        self.arch_name = arch_name

        _,self.val_dataloader_uniform,_ = create_dataloaders_uniform(self.path2data,self.data_name,batch_size = 128,val_size = 0.005)
        
        _,self.val_dataloader_zipfian,_ = create_dataloaders_zipfian(self.path2data,self.data_name,batch_size = 128,val_size = 0.005)
        
        # Override batch size for validation loader
        self.val_dataloader_uniform = DataLoader(
            self.val_dataloader_uniform.dataset, batch_size=len(self.val_dataloader_uniform.dataset), shuffle=False
        )

        self.val_dataloader_zipfian = DataLoader(
            self.val_dataloader_zipfian.dataset, batch_size=len(self.val_dataloader_zipfian.dataset), shuffle=False
        )

        with open(model_uniform, 'rb') as f:
            self.model_uniform = pkl.load(f)

        with open(model_zipfian, 'rb') as f:
            self.model_zipfian = pkl.load(f)

    def _get_encodings(self):
        if torch.cuda.is_available():
            device = torch.device('cuda:0')
        else:
            device = torch.device('cpu')

        output_dict = {}

        
        batch_uniform = next(iter(self.val_dataloader_uniform))  # Only one batch since batch_size = dataset size
        inputs_uniform, self.labels_uniform = batch_uniform
        inputs_uniform = inputs_uniform.to(device)

        batch_zipfian = next(iter(self.val_dataloader_zipfian))  # Only one batch since batch_size = dataset size
        inputs_zipfian, self.labels_zipfian = batch_zipfian
        inputs_zipfian = inputs_zipfian.to(device)


        with torch.no_grad():  # No gradients needed for inference

           h1_probs_uniform =  self.model_uniform.layers[0].forward(inputs_uniform)
           h2_probs_uniform = self.model_uniform.layers[1].forward(h1_probs_uniform)

           h1_probs_zipfian =  self.model_zipfian.layers[0].forward(inputs_zipfian)
           h2_probs_zipfian = self.model_zipfian.layers[1].forward(h1_probs_zipfian)

        Z_uniform = h2_probs_uniform.cpu().numpy()

        Z_zipfian = h2_probs_zipfian.cpu().numpy()

        del inputs_uniform,inputs_zipfian,self.model_uniform,self.model_zipfian

        # Run garbage collection
        gc.collect()

        # Free unused CUDA memory
        torch.cuda.empty_cache()

        output_dict['Z_uniform'] = Z_uniform
        output_dict['Z_zipfian'] = Z_zipfian
        output_dict['labels_uniform'] = self.labels_uniform.cpu().numpy()
        output_dict['labels_zipfian'] = self.labels_zipfian.cpu().numpy()

        return output_dict
    
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
                A[i_idx, j_idx] = np.mean(l) if l else np.nan

        return A

            
       


