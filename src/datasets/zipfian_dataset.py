

from pathlib import Path
import torch
from torch.utils.data import Dataset, DataLoader, Subset, random_split
import numpy as np
from sklearn.model_selection import train_test_split
import functools


class ZipfianDataset(Dataset):
    def __init__(self, directory):
        self.directory = Path(directory)
        self.files = sorted(self.directory.glob("**/*.pt"))

        # Extract labels from filenames (assumes "NumX_" pattern)
        self.labels = sorted({int(f.stem.split("Num")[1].split("_")[0]) for f in self.files})
        self.label_to_index = {label: idx for idx, label in enumerate(self.labels)}

        # Create mapping of sample indices and labels for stratified sampling
        self.sample_index = []
        self.sample_labels = []
        self.sample_labels_onehot = []

        for file_path in self.files:
            data = torch.load(file_path).float()
            num_samples = data.size(0) if isinstance(data, torch.Tensor) else len(data)

            label = int(file_path.stem.split("Num")[1].split("_")[0])
            label_idx = self.label_to_index[label]
            one_hot = torch.zeros(len(self.labels))
            one_hot[label_idx] = 1

            for sample_idx in range(num_samples):
                self.sample_index.append((file_path, sample_idx))
                self.sample_labels.append(label_idx)
                self.sample_labels_onehot.append(one_hot)

        self.sample_labels_onehot = torch.stack(self.sample_labels_onehot)

    def __len__(self):
        return len(self.sample_index)

    @functools.lru_cache(maxsize=128)  # Cache file loads
    def load_file(self, file_path):
        return torch.load(file_path).float()

    def __getitem__(self, idx):
        file_path, sample_idx = self.sample_index[idx]
        data = self.load_file(str(file_path))
        sample = data[sample_idx] / 255.0  # Normalize to [0,1]
        return sample, self.sample_labels_onehot[idx]


def create_dataloaders_zipfian(directory, batch_size=32, subsample_fraction=None, num_workers=4, random_state=42):
    dataset = ZipfianDataset(directory)
    total_samples = len(dataset)
    indices = np.arange(total_samples)

    # Subsampling (optional)
    if subsample_fraction is not None and subsample_fraction < 1.0:
        np.random.seed(random_state)
        num_subsample = int(total_samples * subsample_fraction)
        indices = np.random.choice(indices, num_subsample, replace=False)

    labels = np.array(dataset.sample_labels)[indices]

    # Stratified split: 80% train, 10% validation, 10% test
    train_idx, temp_idx, _, temp_labels = train_test_split(
        indices, labels, test_size=0.02, stratify=labels, random_state=random_state
    )
    val_idx, test_idx = train_test_split(
        temp_idx, test_size=0.1, stratify=temp_labels, random_state=random_state
    )

    train_dataset = Subset(dataset, train_idx)
    val_dataset = Subset(dataset, val_idx)
    test_dataset = Subset(dataset, test_idx)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers,pin_memory = True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    return train_loader, val_loader, test_loader












"""
from pathlib import Path
import torch
from torch.utils.data import Dataset,DataLoader,Subset
from sklearn.model_selection import train_test_split
import pytorch_lightning as pl
from collections import Counter
import numpy as  np 

from pathlib import Path
import torch
from torch.utils.data import Dataset, DataLoader, Subset
import numpy as np
from sklearn.model_selection import train_test_split
import pytorch_lightning as pl


class ZipfianDataset(Dataset):
    def __init__(self, directory):
        self.directory = Path(directory)
        self.files = sorted(list(self.directory.glob("**/*.pt")))
        
        # Determine unique labels from filenames (assumes "NumX_" pattern)
        self.labels = sorted({int(f.stem.split("Num")[1].split("_")[0]) for f in self.files})
        self.label_to_index = {label: idx for idx, label in enumerate(self.labels)}
        
        # Build a list of sample indices and corresponding label indices
        self.sample_index = []   # list of tuples (file_index, sample_index)
        self.sample_labels = []  # parallel list of label indices for each sample
        self.loaded_data = []  # each element corresponds to one file's data

        for file_idx, file_path in enumerate(self.files):
            data = torch.load(file_path)
            self.loaded_data.append(data)
            num_samples = data.size(0) if isinstance(data, torch.Tensor) else len(data)
            
            # Extract label from file name (assumes pattern "NumX_")
            label = int(file_path.stem.split("Num")[1].split("_")[0])
            label_idx = self.label_to_index[label]
            
            for sample_idx in range(num_samples):
                self.sample_index.append((file_idx, sample_idx))
                self.sample_labels.append(label_idx)
                
    def __len__(self):
        return len(self.sample_index)
    
    def __getitem__(self, idx):
        file_idx, sample_idx = self.sample_index[idx]
        file_path = self.files[file_idx]
        data = self.loaded_data[file_idx].float()
        sample = data[sample_idx] if isinstance(data, torch.Tensor) else data[sample_idx]
        
        # Create one-hot encoding for the label
        one_hot = torch.zeros(len(self.labels))
        one_hot[self.sample_labels[idx]] = 1
        
        return sample, one_hot
"""

"""

class ZipfianDataModule(pl.LightningDataModule):
    def __init__(self, directory, batch_size=32, subsample_fraction=None, num_workers=0, random_state=42):
        super().__init__()
        self.directory = directory
        self.batch_size = batch_size
        self.subsample_fraction = subsample_fraction
        self.num_workers = num_workers
        self.random_state = random_state

    def setup(self, stage=None):
        # Load the full dataset
        full_dataset = ZipfianDataset(self.directory)
        total_samples = len(full_dataset)
        indices = np.arange(total_samples)

        
        # Optionally subsample the dataset
        if self.subsample_fraction is not None and self.subsample_fraction < 1.0:
            np.random.seed(self.random_state)
            num_subsample = int(total_samples * self.subsample_fraction)
            indices = np.random.choice(indices, num_subsample, replace=False)
        
        # Get labels corresponding to the chosen indices
        labels = np.array(full_dataset.sample_labels)[indices]
        
        # Split into train (80%), validation (10%), and test (10%) sets
        train_idx, temp_idx, _, temp_labels = train_test_split(
            indices, labels, test_size=0.2, stratify=labels, random_state=self.random_state
        )
        val_idx, test_idx = train_test_split(
            temp_idx, test_size=0.5, stratify=temp_labels, random_state=self.random_state
        )
        
        # Wrap the full dataset with subsets (this is enough if you don't need extra methods)
        self.train_dataset = Subset(full_dataset, train_idx)
        self.val_dataset   = Subset(full_dataset, val_idx)
        self.test_dataset  = Subset(full_dataset, test_idx)
        
        # Optionally, save the shape of one sample for later use
        self.data_shape = full_dataset[0][0].shape[0]

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers
        )
    
    def val_dataloader(self):
        return DataLoader(
            self.val_dataset, batch_size=self.batch_size, shuffle=False
        )
    
    def test_dataloader(self):
        return DataLoader(
            self.test_dataset, batch_size=self.batch_size, shuffle=False
        )
"""



