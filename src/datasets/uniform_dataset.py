import torch

import numpy as np
import pandas as pd 
import torch.nn.functional as F

import pickle as pkl

from pathlib import Path
import torch
from torch.utils.data import Dataset, DataLoader, Subset, random_split
import numpy as np
from sklearn.model_selection import train_test_split
import functools
from torch.utils.data import WeightedRandomSampler
from collections import Counter


class UniformDataset(Dataset):

    def __init__(self, data_path,dataset_name,non_numerical=False, batch_size=128, num_workers=4,multimodal_flag=False):
        super().__init__()
        self.data_path = data_path
        self.dataset_name = dataset_name
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.non_numerical = non_numerical

        data_name = f"{self.data_path}/{self.dataset_name}"

        data = np.load(data_name)

        self.multimodal_flag = multimodal_flag
        self.data = torch.tensor(data['D'], dtype = torch.float32)  # shape: [batch_size, 10000]
        self.labels = data['N_list']

        self.cumArea_list = data['cumArea_list']
        #self.FA_list = data['FA_list']
        self.CH_list = data['CH_list']
        #self.Items_list = data['item_size']


        # create one-hot encoded label_list
        labels = torch.tensor(self.labels, dtype=torch.long)  # shape: [batch_size]
        labels_shifted = labels - 1
        self.one_hot = F.one_hot(labels_shifted, num_classes=32).float()  # shape: [batch_size, 32]



        """
        labels_mask = self.labels <= 4
        self.data = self.data[labels_mask]
        self.TSA_list = self.TSA_list[labels_mask]
        self.cumArea_list = self.cumArea_list[labels_mask]
        self.FA_list = self.FA_list[labels_mask]
        self.CH_list = self.CH_list[labels_mask]
        self.labels = self.labels[labels_mask]
        self.sparsity = self.FA_list /self.labels
        self.ISA = self.TSA_list /self.labels
        self.size = self.TSA_list + self.ISA
        """



    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        
        data = (torch.tensor(self.data[idx])!=0 ).float()  # normalize the images 
        labels = torch.tensor(self.labels[idx]) if not self.multimodal_flag else self.one_hot[idx] # if multimodal, return one-hot encoded labels
        if self.non_numerical:
            #TSA = torch.tensor(float(self.TSA_list[idx]))
            cumArea = torch.tensor(float(self.cumArea_list[idx]))
            FA = torch.tensor(float(self.FA_list[idx]))
            #ISA = torch.tensor(float(self.ISA[idx]))
            CH = torch.tensor(float(self.CH_list[idx]))
            return data, labels, cumArea, FA, CH
        else:
            return data, labels


def create_dataloaders_uniform(data_path,data_name, batch_size=32, num_workers=4, test_size=0.2, val_size=0.1, random_state=42,multimodal_flag=False):
    dataset = UniformDataset(data_path, data_name,multimodal_flag=multimodal_flag)
    total_samples = len(dataset)
    indices = np.arange(total_samples)
    labels = np.array(dataset.labels)
    
    # Splitting data into train, validation, and test sets
    train_idx, temp_idx, _, temp_labels = train_test_split(
        indices, labels, test_size=(test_size + val_size), stratify=labels, random_state=random_state
    )
    val_idx, test_idx = train_test_split(
        temp_idx, test_size=(test_size / (test_size + val_size)), stratify=temp_labels, random_state=random_state
    )
    
    train_dataset = Subset(dataset, train_idx)
    val_dataset = Subset(dataset, val_idx)
    test_dataset = Subset(dataset, test_idx)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=len(val_dataset), shuffle=False, num_workers=num_workers)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    
    return train_loader, val_loader, test_loader


def create_dataloaders_zipfian(data_path, data_name, batch_size=32, num_workers=4,
                               test_size=0.2, val_size=0.1, random_state=42,multimodal_flag=False):
    # Step 1: Build the zipfian_probs dictionary
    def shifted_zipf_pmf(k, a, s):
        return (k + s)**(-a) / np.sum((np.arange(1, max(k)+1) + s)**(-a))

    a = 112.27
    s = 714.33
    k_vals = np.arange(1, 41)
    zipfian_raw = shifted_zipf_pmf(k_vals, a, s)
    zipfian_probs = {i: zipfian_raw[i - 1] for i in range(1, 33)}  # class labels 1-based

    dataset = UniformDataset(data_path, data_name,multimodal_flag=multimodal_flag)
    total_samples = len(dataset)
    indices = np.arange(total_samples)
    labels = np.array(dataset.labels)

    train_idx, temp_idx, _, temp_labels = train_test_split(
        indices, labels, test_size=(test_size + val_size), stratify=labels, random_state=random_state
    )
    val_idx, test_idx = train_test_split(
        temp_idx, test_size=(test_size / (test_size + val_size)), stratify=temp_labels, random_state=random_state
    )

    train_labels = labels[train_idx]
    
    # Adjust class index for label range (1–40 vs 0–39)
    sample_weights = torch.DoubleTensor([zipfian_probs[int(label)] for label in train_labels])

    train_sampler = WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(train_idx),
        replacement=True
    )

    train_dataset = Subset(dataset, train_idx)
    val_dataset = Subset(dataset, val_idx)
    test_dataset = Subset(dataset, test_idx)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, sampler=train_sampler, num_workers=num_workers, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=len(val_dataset), shuffle=False, num_workers=num_workers)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    return train_loader, val_loader, test_loader
