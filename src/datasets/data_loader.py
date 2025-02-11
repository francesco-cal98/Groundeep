import pytorch_lightning as pl
import torch
from torch.utils.data import Dataset, DataLoader
from scipy import io
import numpy as np
from sklearn.model_selection import train_test_split
import torch.nn.functional as F

class NumerosityDataset(Dataset):
    def __init__(self, data, labels):
        self.data = data
        self.labels = labels

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]

class NumerosityDataModule(pl.LightningDataModule):
    def __init__(self, data_path, batch_size=128, num_workers=4):
        super().__init__()
        self.data_path = data_path
        self.batch_size = batch_size
        self.num_workers = num_workers

    def setup(self, stage=None):
        # Load MATLAB data
        matlab_data = io.loadmat(self.data_path)
        D = matlab_data['D']  # Images data
        N_list = matlab_data['N_list'].flatten()  # Numerosity list

        # Shuffle and split the data
        D_train, D_test, N_list_train, N_list_test = train_test_split(D.T, N_list, test_size=0.2, random_state=42)

        # Reshape the data
        NUM_CHANNELS = 1
        IMAGE_WIDTH = 100
        IMAGE_HEIGHT = 100

        self.train_data = torch.from_numpy(D_train).float().view(-1, NUM_CHANNELS, IMAGE_WIDTH, IMAGE_HEIGHT)
        self.train_labels = torch.from_numpy(N_list_train).float().unsqueeze(1)

        self.test_data = torch.from_numpy(D_test).float().view(-1, NUM_CHANNELS, IMAGE_WIDTH, IMAGE_HEIGHT)
        self.test_labels = torch.from_numpy(N_list_test).float().unsqueeze(1)

        self.train_labels_one_hot = F.one_hot(self.train_labels.long() - 1, num_classes=int(np.max(N_list_train))).float()
        self.test_labels_one_hot = F.one_hot(self.test_labels.long() - 1, num_classes=int(np.max(N_list_test))).float()


        # Create datasets
        self.train_dataset = NumerosityDataset(self.train_data, self.train_labels_one_hot)
        self.test_dataset = NumerosityDataset(self.test_data, self.test_labels_one_hot)

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers)

    def val_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size, num_workers=self.num_workers)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size, num_workers=self.num_workers)