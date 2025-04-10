import torch
import pickle as pkl
import os 
import sys
import gc
current_dir = os.getcwd()
sys.path.append(current_dir)  # Add the project root to sys.path

from src.datasets.uniform_dataset import create_dataloaders_uniform
from torch.utils.data import DataLoader


class Embedding_analysis:

    def __init__(self,path2data,data_name,model):
        
        self.path2data = path2data
        self.data_name = data_name

        _,self.val_dataloader,_ = create_dataloaders_uniform(self.path2data,self.data_name,batch_size = 128)
        
        
        # Override batch size for validation loader
        self.val_dataloader = DataLoader(
            self.val_dataloader.dataset, batch_size=len(self.val_dataloader.dataset), shuffle=False
        )
        with open(model, 'rb') as f:
            self.model = pkl.load(f)

    def _get_encodings(self):
        if torch.cuda.is_available():
            device = torch.device('cuda:0')
        else:
            device = torch.device('cpu')

        


        batch = next(iter(self.val_dataloader))  # Only one batch since batch_size = dataset size
        inputs, labels = batch
        inputs = inputs.to(device)

        with torch.no_grad():  # No gradients needed for inference

           h1_probs =  self.model.layers[0].forward(inputs)
           h2_probs = self.model.layers[1].forward(h1_probs)

        Z = h2_probs.cpu().numpy()

        del inputs,self.model

        # Run garbage collection
        gc.collect()

        # Free unused CUDA memory
        torch.cuda.empty_cache()

        return Z, labels.cpu().numpy()
    
            
       


