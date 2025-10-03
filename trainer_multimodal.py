from gdbn_model import iMDBN
import torch
import os
import wandb
from src.datasets.uniform_dataset import create_dataloaders_uniform


def main():
    # Parametri di training
    params = {
        "LEARNING_RATE": 0.1,
        "WEIGHT_PENALTY": 0.0001,
        "INIT_MOMENTUM": 0.5,
        "FINAL_MOMENTUM": 0.95,
        "LEARNING_RATE_DYNAMIC": True,
        "CD": 1,
        "EPOCHS": 100,
        "EPOCHS TEXT":10,
        "EPOCHS JOINT": 200,
        "SAVE_PATH": "/home/student/Desktop/Groundeep/networks/zipfian/imdbn",
        "SAVE_NAME": "imdbn_trained_zipfian",
        "LOG_EVERY_PCA": 10,
        "ENABLE_WANDB": True,
        "WANDB_PROJECT": "groundeep-diagnostics-multimodal",
    }

    # Architetture
    layer_sizes_img = [10000, 1500, 500]  # immagini (100x100 flatten → 10k visibili → hidden)
    layer_sizes_txt = [32, 500,500]           # labels one-hot (32 possibili classi)
    joint_layer_size = 1000              # dimensione hidden RBM di fusione

    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Dataset multimodale (immagini + labels one-hot)
    train_loader, val_loader, test_loader = create_dataloaders_uniform(
        data_path="/home/student/Desktop/Groundeep/stimuli_dataset_controlled",
        data_name="stimuli_dataset.npz",
        batch_size=128,
        num_workers=1,
        multimodal_flag=True
    )

    # Costruisci path di salvataggio
    save_path = os.path.join(
        params["SAVE_PATH"],
        f"{params['SAVE_NAME']}_IMG{'-'.join(map(str,layer_sizes_img))}_TXT{'-'.join(map(str,layer_sizes_txt))}_JOINT{joint_layer_size}.pkl"
    )
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    wandb_run = None
    if params.get("ENABLE_WANDB", False):
        wandb_run = wandb.init(project=params.get("WANDB_PROJECT", "groundeep-diagnostics"),
                                config=params)

    # Istanzia il modello multimodale
    imdbn = iMDBN(layer_sizes_img, layer_sizes_txt, joint_layer_size,
                  params, train_loader, val_loader, device, wandb_run=wandb_run)

    # 1️⃣ Allena iDBN immagine
    print("Training image iDBN...")
    imdbn.image_idbn.train(params["EPOCHS"])

    # 2️⃣ Allena iDBN testo (labels one-hot)
    print("Training text iDBN...")
    imdbn.text_idbn.train(params["EPOCHS TEXT"])

    # 3️⃣ Allena RBM congiunto
    print("Training joint RBM...")
    imdbn.train_joint(epochs=params["EPOCHS JOINT"],
                      log_every_pca=params.get("LOG_EVERY_PCA", 10))

    # Salva modello completo
    imdbn.save_model(save_path)
    print(f"✅ Saved trained multimodal DBN to {save_path}")

    if wandb_run is not None:
        wandb_run.finish()


if __name__ == "__main__":
    main()
