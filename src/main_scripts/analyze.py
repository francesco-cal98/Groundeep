# src/main_scripts/analyze.py

import pickle
import os
import sys
import torch
from pathlib import Path
import yaml

CURRENT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = CURRENT_DIR.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.classes.repr_analysis import RepresentationalAnalysis
#from src.classes.beh_analysis import BehavioralAnalysis
from src.datasets.uniform_dataset import create_dataloaders_uniform


def _move_to_device(model, device):
    """Recursively move RBM weights to the desired device and update references."""
    if model is None:
        return model

    if hasattr(model, "device"):
        model.device = device

    layers = getattr(model, "layers", None)
    if layers is not None:
        for rbm in layers:
            if hasattr(rbm, "to"):
                rbm.to(device)

    joint_rbm = getattr(model, "joint_rbm", None)
    if joint_rbm is not None and hasattr(joint_rbm, "to"):
        joint_rbm.to(device)

    # Recursively handle multimodal branches
    for attr_name in ("image_idbn", "text_idbn"):
        branch = getattr(model, attr_name, None)
        if branch is not None and branch is not model:
            _move_to_device(branch, device)

    return model


def _build_feature_cache(val_loader):
    """Reconstruct feature tensors (labels, cumArea, CH, density) from the val_loader."""
    if val_loader is None:
        return {}

    subset = getattr(val_loader, "dataset", None)
    indices = getattr(subset, "indices", None)
    base_dataset = getattr(subset, "dataset", None)

    if indices is None or base_dataset is None:
        return {}

    features = {
        "Cumulative Area": torch.tensor([base_dataset.cumArea_list[i] for i in indices], dtype=torch.float32),
        "Convex Hull": torch.tensor([base_dataset.CH_list[i] for i in indices], dtype=torch.float32),
        "Labels": torch.tensor([base_dataset.labels[i] for i in indices], dtype=torch.float32),
    }

    density_source = getattr(base_dataset, "density_list", None)
    if density_source is not None:
        features["Density"] = torch.tensor([density_source[i] for i in indices], dtype=torch.float32)

    return features


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

    # Cache dei dataloader in base a (dataset_path, dataset_name, batch_size, num_workers, multimodal_flag)
    dataloader_cache = {}

    def get_dataloaders(dataset_path, dataset_name, batch_size, num_workers, multimodal_flag):
        key = (dataset_path, dataset_name, batch_size, num_workers, bool(multimodal_flag))
        if key not in dataloader_cache:
            dataloader_cache[key] = create_dataloaders_uniform(
                data_path=dataset_path,
                data_name=dataset_name,
                batch_size=batch_size,
                num_workers=num_workers,
                multimodal_flag=multimodal_flag,
            )
        return dataloader_cache[key]

    # === 3. Ciclo di analisi per ogni modello nel file di configurazione ===
    for model_params in config['model_to_analyze']:
        arch_name = model_params.get('arch', 'unknown_arch')
        dist_name = model_params.get('distribution', 'unknown_distribution')
        model_type = model_params.get('model_type', 'idbn').lower()

        dataset_path = model_params.get('dataset_path', config['dataset_path'])
        dataset_name = model_params.get('dataset_name', config['dataset_name'])
        batch_size = model_params.get('batch_size', config['batch_size'])
        num_workers = model_params.get('num_workers', config['num_workers'])
        multimodal_flag = model_params.get('multimodal_flag', config.get('multimodal_flag', False))

        train_loader, val_loader, test_loader = get_dataloaders(
            dataset_path, dataset_name, batch_size, num_workers, multimodal_flag
        )

        print(f"🔄 Starting analysis for model: {arch_name} (type={model_type}), distribution: {dist_name}")

        model_path = model_params.get('model_path')
        if not model_path:
            if model_type != 'idbn':
                print("⚠️ Please provide 'model_path' for non-standard models. Skipping.")
                continue

            epochs = model_params.get('epochs')
            if epochs is None:
                print("⚠️ Missing 'epochs' in config entry; cannot infer model path. Skipping.")
                continue

            networks_root = Path(model_params.get('networks_root', config.get('networks_root', Path(dataset_path).parent / 'networks')))
            model_subdir = model_params.get('model_subdir', f"idbn_new_dataset_{epochs}_epochs")
            model_filename = model_params.get('model_filename', f"i_dbn_trained_{dist_name}_{arch_name}.pkl")
            model_path = networks_root / dist_name / model_subdir / model_filename
        else:
            model_path = Path(model_path)

        if not model_path.exists():
            print(f"⚠️ Model not found at: {model_path}. Skipping analysis for this model.")
            continue

        print(f"Loading model from {model_path}...")
        with open(model_path, 'rb') as f:
            model = pickle.load(f)

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        _move_to_device(model, device)

        # Attach dataloaders for downstream analyses
        if getattr(model, 'val_loader', None) is None:
            model.val_loader = val_loader
        if getattr(model, 'dataloader', None) is None:
            model.dataloader = train_loader

        # Ensure val_batch is present and consistent with the current loader
        try:
            batch_from_val = next(iter(val_loader))
        except StopIteration:
            batch_from_val = None

        if model_type == 'imdbn' and batch_from_val is not None:
            model.val_batch = batch_from_val
        elif batch_from_val is not None:
            if isinstance(batch_from_val, (tuple, list)):
                model.val_batch = batch_from_val[0]
            else:
                model.val_batch = batch_from_val

        if not getattr(model, 'features', None):
            model.features = _build_feature_cache(val_loader)
        else:
            # Guarantee Labels are numeric tensors for downstream metrics
            labels_feat = model.features.get("Labels")
            if isinstance(labels_feat, torch.Tensor) and labels_feat.dim() > 1:
                model.features["Labels"] = labels_feat.argmax(dim=1).float() + 1

        # === 4. Esecuzione delle analisi ===
        repr_analysis = RepresentationalAnalysis(model=model)
        repr_analysis.run_all_analyses(data=model, arch_name=arch_name, dist_name=dist_name)

        #beh_analysis = BehavioralAnalysis(model=model)
        #beh_analysis.run_analysis(test_loader=test_loader)

        print(f"✅ Analysis for model {arch_name} complete.")


if __name__ == "__main__":
    run_analysis_pipeline()
