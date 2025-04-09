import torch

import numpy as np
import pandas as pd 


from pathlib import Path
import torch
from torch.utils.data import Dataset, DataLoader, Subset, random_split
import numpy as np
from sklearn.model_selection import train_test_split
import functools



class UniformDataset(Dataset):

    def __init__(self, data_path,dataset_name, batch_size=128, num_workers=4):
        super().__init__()
        self.data_path = data_path
        self.dataset_name = dataset_name
        self.batch_size = batch_size
        self.num_workers = num_workers

        data_name = f"{self.data_path}/{self.dataset_name}"

        data = np.load(data_name)

        self.data = data['D']
        self.labels = data['N_list']

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        data = (torch.tensor(self.data[idx])) / 255 # normalize the images 
        labels = torch.tensor(self.data[idx])
        return data, labels



def create_dataloaders_uniform(data_path,data_name, batch_size=32, num_workers=4, test_size=0.2, val_size=0.1, random_state=42):
    dataset = UniformDataset(data_path, data_name)
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
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    
    return train_loader, val_loader, test_loader