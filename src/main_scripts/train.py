# src/main_scripts/train.py

import torch
import os
import yaml
import wandb
from pathlib import Path
import sys

# === PATH SETUP ===
current_dir = os.getcwd()
project_root = os.path.abspath(os.path.join(current_dir, '..', '..'))
sys.path.append(project_root)
sys.path.append(current_dir) # Add current_dir if it contains other necessary modules

from src.classes.gdbn_model import gDBN, iDBN
from src.datasets.uniform_dataset import create_dataloaders_uniform, create_dataloaders_zipfian


def run_training():
    """
    Script principale per l'addestramento del modello.
    Legge la configurazione e avvia il processo di training.
    """
    
    config_path = Path("src/configs/training_config.yaml")
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found at {config_path}")
        
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    params = {
        "ALGORITHM": config['algorithm'],
        "LEARNING_RATE": config['learning_rate'],
        "WEIGHT_PENALTY": config['weight_penalty'],
        "INIT_MOMENTUM": config['init_momentum'],
        "FINAL_MOMENTUM": config['final_momentum'],
        "LEARNING_RATE_DYNAMIC": True,
        "CD": config['cd_k'],
        "EPOCHS": config['epochs'],
        "SPARSITY": config['sparsity'],
        "SPARSITY_FACTOR": config['sparsity_factor'],
        "SAVE_PATH": config['save_path'],
        "SAVE_NAME": config['save_name'],
    }
    layer_sizes_list = config['layer_sizes_list']
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    train_loader, val_loader, test_loader = create_dataloaders_uniform(
        data_path=config['dataset_path'],
        data_name=config['dataset_name'],
        batch_size=config['batch_size'],
        num_workers=config['num_workers'],
        multimodal_flag=config['multimodal_flag']
    )
    
    wandb_run = wandb.init(project="groundeep-diagnostics")

    for layer_sizes in layer_sizes_list:
        arch_name = '_'.join(map(str, layer_sizes))
        full_arch_name = f"{params['ALGORITHM']}_{params['SAVE_NAME']}_{arch_name}"
        save_path = f"{params['SAVE_PATH']}_{full_arch_name}.pkl"
        os.makedirs(os.path.dirname(save_path), exist_ok=True)

        if params["ALGORITHM"] == "g":
            dbn = gDBN(
                layer_sizes=[10000] + layer_sizes, 
                params=params, 
                dataloader=train_loader, 
                device=device
            )                 
            dbn.train(epochs=params["EPOCHS"])
            dbn.save(save_path)
            
        elif params["ALGORITHM"] == "i":
            dbn = iDBN(
                layer_sizes=[10000] + layer_sizes,
                params=params,
                dataloader=train_loader,
                val_loader=val_loader,
                device=device,
                wandb_run=wandb_run
            )
            print(f"Starting training for architecture: {full_arch_name}")
            dbn.train(epochs=params["EPOCHS"])
            dbn.save_model(save_path)

    print("✅ Training complete for all architectures.")

if __name__ == "__main__":
    run_training()