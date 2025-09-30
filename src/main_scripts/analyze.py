# src/main_scripts/analyze.py

import pickle
import os
import torch
from pathlib import Path
import yaml

from src.classes.repr_analysis import RepresentationalAnalysis
#from src.classes.beh_analysis import BehavioralAnalysis
from src.datasets.uniform_dataset import create_dataloaders_uniform


def run_analysis_pipeline():
    """
    Script principale per l'analisi dei modelli addestrati.
    Carica i modelli in base al file di configurazione e lancia le analisi.
    """
    
    # === 1. Caricamento della configurazione ===
    config_path = Path("src/configs/analysis_config.yaml")
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found at {config_path}")
        
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    # === 2. Preparazione dei dati per l'analisi (comuni a tutti i modelli) ===
    train_loader, val_loader, test_loader = create_dataloaders_uniform(
        data_path=config['dataset_path'],
        data_name=config['dataset_name'],
        batch_size=config['batch_size'],
        num_workers=config['num_workers'],
        multimodal_flag=config['multimodal_flag']
    )

    # === 3. Ciclo di analisi per ogni modello nel file di configurazione ===
    for model_params in config['model_to_analyze']:
        arch_name = model_params['arch']
        dist_name = model_params['distribution']
        epochs = model_params['epochs']

        print(f"🔄 Starting analysis for model: {arch_name}, distribution: {dist_name}")

        # Percorso del modello salvato
        model_path = os.path.join(
            config['dataset_path'].replace("stimuli_dataset_adaptive", "networks"),
            dist_name,
            f"idbn_new_dataset_{epochs}_epochs",
            f"i_dbn_trained_{dist_name}_{arch_name}.pkl"
        )
        # Nota: Ho reso il path più dinamico. Dovrai adattarlo al tuo percorso specifico.

        if not os.path.exists(model_path):
            print(f"⚠️ Model not found at: {model_path}. Skipping analysis for this model.")
            continue

        print(f"Loading model from {model_path}...")
        with open(model_path, 'rb') as f:
            model = pickle.load(f)

        # Aggiungo i dataloader al modello, se non sono già presenti
        if not hasattr(model, 'val_loader'):
            model.val_loader = val_loader
        if not hasattr(model, 'dataloader'):
            model.dataloader = train_loader

        # === 4. Esecuzione delle analisi ===
        repr_analysis = RepresentationalAnalysis(model=model)
        repr_analysis.run_all_analyses(data=model, arch_name=arch_name, dist_name=dist_name)

        #beh_analysis = BehavioralAnalysis(model=model)
        #beh_analysis.run_analysis(test_loader=test_loader)
        
        print(f"✅ Analysis for model {arch_name} complete.")


if __name__ == "__main__":
    run_analysis_pipeline()