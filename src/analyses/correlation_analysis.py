import torch
import pickle as pkl
import os 
import sys
import gc
import numpy as np


import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import spearmanr
import torch

import numpy as np
import torch

def compute_avg_pearson_with_numerosity(data_loader):
    """
    Computes the average Pearson correlation matrix including:
    [TSA, cumArea, FA, CH, N_list]
    """
    num_features = 5  # TSA, cumArea, FA, CH, N_list
    accumulated_corr = np.zeros((num_features, num_features))
    batch_count = 0

    for batch in data_loader:
        _, N_list, TSA, cumArea, FA, CH = batch  # Unpack tensors

        # Stack features and numerosity: shape [batch_size, 5]
        features = torch.stack([TSA, cumArea, FA, CH, N_list], dim=1).cpu().numpy()

        # Compute Pearson correlation matrix for this batch
        corr_matrix = np.corrcoef(features, rowvar=False)  # rowvar=False to treat columns as variables

        accumulated_corr += corr_matrix
        batch_count += 1

    # Average over batches
    avg_corr_matrix = accumulated_corr / batch_count
    return avg_corr_matrix


def compute_avg_spearman_with_numerosity(data_loader):
    """
    Computes the average Spearman correlation matrix including:
    [TSA, cumArea, FA, CH, N_list]
    """
    num_features = 5  # TSA, cumArea, FA, CH, N_list
    accumulated_corr = np.zeros((num_features, num_features))
    batch_count = 0

    for batch in data_loader:
        _, N_list, TSA, cumArea, FA, CH = batch  # Unpack tensors

        # Stack features and numerosity: shape [batch_size, 5]
        features = torch.stack([N_list,TSA, cumArea, FA, CH], dim=1).cpu().numpy()

        # Compute Spearman correlation matrix for this batch
        corr_matrix = np.zeros((num_features, num_features))
        for i in range(num_features):
            for j in range(num_features):
                corr = np.corrcoef(features[:, i], features[:, j])
                corr_matrix[i, j] = corr

        accumulated_corr += corr_matrix
        batch_count += 1

    # Average over batches
    avg_corr_matrix = accumulated_corr / batch_count
    return avg_corr_matrix


def plot_spearman_heatmap(corr_matrix, feature_names, save_path=None):
    """Plots and saves the Spearman correlation heatmap."""
    plt.figure(figsize=(6, 5))
    sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap="coolwarm",
                xticklabels=feature_names, yticklabels=feature_names,
                vmin=0, vmax=1)
    plt.title("Average Spearman Correlation Matrix")
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300)
    plt.show()


current_dir = os.getcwd()
sys.path.append(current_dir)  # Add the project root to sys.path
from src.datasets.uniform_dataset import create_dataloaders_uniform,create_dataloaders_zipfian

train_loader, val_loader, test_loader = create_dataloaders_uniform("/home/student/Desktop/Groundeep/training_tensors/uniform/","NumStim_1to40_100x100_TR_uniform_non_numerical.npz", batch_size = 128, num_workers = 1,)

# Assuming you have a data loader already:

# Compute and plot
feature_names = ['N_list','TSA', 'cumArea', 'FA', 'CH']
corr_matrix = compute_avg_pearson_with_numerosity(train_loader)
plot_spearman_heatmap(corr_matrix, feature_names, save_path="/home/student/Desktop/Groundeep/outputs/correlation_analysis/correlation_uniform.jpg")
