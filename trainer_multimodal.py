from pathlib import Path

import argparse
import sys

import torch
import wandb
import yaml

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
SRC_ROOT = PROJECT_ROOT / 'src'
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from src.classes.gdbn_model import iDBN, iMDBN
from src.datasets.uniform_dataset import create_dataloaders_uniform


DEFAULT_CONFIG_PATH = Path("src/configs/multimodal_training_config.yaml")


def parse_args():
    parser = argparse.ArgumentParser(description="Train a multimodal iDBN")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH,
                        help="Path to the multimodal training configuration file")
    return parser.parse_args()


def load_config(config_path: Path = DEFAULT_CONFIG_PATH) -> dict:
    if not config_path.exists():
        raise FileNotFoundError(f"Multimodal config not found at {config_path}")
    with config_path.open("r") as fp:
        return yaml.safe_load(fp)


def build_params(cfg: dict) -> dict:
    training_cfg = cfg.get("training", {})
    paths_cfg = cfg.get("paths", {})
    wandb_cfg = cfg.get("wandb", {})

    params = {
        "LEARNING_RATE": training_cfg.get("learning_rate", 0.1),
        "WEIGHT_PENALTY": training_cfg.get("weight_penalty", 0.0001),
        "INIT_MOMENTUM": training_cfg.get("init_momentum", 0.5),
        "FINAL_MOMENTUM": training_cfg.get("final_momentum", 0.95),
        "LEARNING_RATE_DYNAMIC": training_cfg.get("learning_rate_dynamic", True),
        "CD": training_cfg.get("cd", 1),
        "EPOCHS": training_cfg.get("epochs_image", 100),
        "EPOCHS TEXT": training_cfg.get("epochs_text", 30),
        "EPOCHS JOINT": training_cfg.get("epochs_joint", 200),
        "JOINT_MASK_P": training_cfg.get("joint_mask_p", 0.1),
        "JOINT_WARMUP_EPOCHS": training_cfg.get("joint_warmup_epochs", 0),
        "LOG_EVERY_PCA": training_cfg.get("log_every_pca", 10),
        "LOG_EVERY_PROBE": training_cfg.get("log_every_probe", training_cfg.get("log_every_pca", 10)),
        "W_REC": training_cfg.get("w_rec", 1.0),
        "W_SUP": training_cfg.get("w_sup", 1.0),
        "JOINT_CD": training_cfg.get("joint_cd", training_cfg.get("cd", 1)),
        "CROSS_GIBBS_STEPS": training_cfg.get("cross_gibbs_steps", 10),
        "TEXT_LEARNING_RATE": training_cfg.get("text_learning_rate", training_cfg.get("learning_rate", 0.1)),
        "JOINT_LEARNING_RATE": training_cfg.get("joint_learning_rate", training_cfg.get("learning_rate", 0.1)),
        "SAVE_PATH": paths_cfg.get("save_dir", "networks/zipfian/imdbn"),
        "SAVE_NAME": paths_cfg.get("save_name", "imdbn_trained"),
        "ENABLE_WANDB": wandb_cfg.get("enable", False),
        "WANDB_PROJECT": wandb_cfg.get("project"),
        "WANDB_ENTITY": wandb_cfg.get("entity"),
        "WANDB_RUN_NAME": wandb_cfg.get("run_name"),
    }
    return params


def initialise_wandb(params: dict, cfg: dict):
    if not params.get("ENABLE_WANDB", False):
        return None

    wandb_kwargs = {"project": params.get("WANDB_PROJECT", "groundeep-diagnostics")}
    entity = params.get("WANDB_ENTITY")
    if entity:
        wandb_kwargs["entity"] = entity
    run_name = params.get("WANDB_RUN_NAME")
    if run_name:
        wandb_kwargs["name"] = run_name

    wandb_config = {
        "training": cfg.get("training", {}),
        "model": cfg.get("model", {}),
        "dataset": cfg.get("dataset", {}),
        "paths": cfg.get("paths", {}),
    }

    return wandb.init(config=wandb_config, **wandb_kwargs)


def main(config_path: Path = DEFAULT_CONFIG_PATH):
    cfg = load_config(config_path)

    dataset_cfg = cfg.get("dataset", {})
    model_cfg = cfg.get("model", {})

    params = build_params(cfg)

    layer_sizes_img = model_cfg.get("image_layers", [10000, 1500, 500])
    layer_sizes_txt = model_cfg.get("text_layers", [64, 256])
    joint_layer_size = model_cfg.get("joint_hidden", 1000)
    text_posenc_dim = model_cfg.get("text_posenc_dim", 32)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_loader, val_loader, test_loader = create_dataloaders_uniform(
        data_path=dataset_cfg.get("path", "stimuli_dataset_adaptive_auto"),
        data_name=dataset_cfg.get("name", "stimuli_dataset.npz"),
        batch_size=dataset_cfg.get("batch_size", 128),
        num_workers=dataset_cfg.get("num_workers", 1),
        multimodal_flag=dataset_cfg.get("multimodal_flag", True),
    )

    save_dir = Path(params["SAVE_PATH"]).expanduser().resolve()
    save_dir.mkdir(parents=True, exist_ok=True)

    save_path = save_dir / (
        f"{params['SAVE_NAME']}"
        f"_IMG{'-'.join(map(str, layer_sizes_img))}"
        f"_TXT{'-'.join(map(str, layer_sizes_txt))}"
        f"_JOINT{joint_layer_size}.pkl"
    )

    wandb_run = initialise_wandb(params, cfg)

    imdbn = iMDBN(
        layer_sizes_img, layer_sizes_txt, joint_layer_size,
        params, train_loader, val_loader, device,
        text_posenc_dim=text_posenc_dim,
        wandb_run=wandb_run
    )

    # Optionally load a pre-trained image iDBN
    image_pre = cfg.get("paths", {}).get("image_idbn_pretrained")
    if image_pre:
        import pickle
        try:
            with open(image_pre, 'rb') as f:
                loaded = pickle.load(f)
            imdbn.image_idbn = loaded
            print(f"Loaded pre-trained image iDBN from {image_pre}")
        except Exception as e:
            print(f"Warning: failed to load pre-trained image iDBN from {image_pre}: {e}. Falling back to training.")
            print("Training image iDBN...")
            imdbn.image_idbn.train(params["EPOCHS"])
    else:
        print("Training image iDBN...")
        imdbn.image_idbn.train(params["EPOCHS"])

    print("Training text iDBN...")
    imdbn.text_encoder.train(params["EPOCHS TEXT"])

    print("Training joint RBM...")
    imdbn.train_joint(
        epochs=params["EPOCHS JOINT"],
        log_every_pca=params.get("LOG_EVERY_PCA", 10),
        log_every_probe=params.get("LOG_EVERY_PROBE", 10),
        w_rec=params.get("W_REC", 1.0),
        w_sup=params.get("W_SUP", 1.0),
    )

    imdbn.save_model(str(save_path))
    print(f"✅ Saved trained multimodal DBN to {save_path}")

    if wandb_run is not None:
        wandb_run.finish()


if __name__ == "__main__":
    args = parse_args()
    main(args.config)
