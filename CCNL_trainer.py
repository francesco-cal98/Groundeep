from gdbn_model import gDBN, iDBN
import torch
import pickle
from src.datasets.uniform_dataset import create_dataloaders_uniform,create_dataloaders_zipfian
import os


def main():
    # Parameters to match MATLAB implementation
    params = {
        "ALGORITHM": "i",
        "LEARNING_RATE": 0.15,
        "WEIGHT_PENALTY": 0.0001,
        "INIT_MOMENTUM": 0.7,
        "FINAL_MOMENTUM": 0.97,  # Final momentum for dynamic momentum increase
        "LEARNING_RATE_DYNAMIC": True,
        "CD": 1,  # Contrastive Divergence  (i.e -1 for Unbiased CD)
        "TRAINING_FILE_PATH":r'/home/student/Desktop/Groundeep/batched_train_data_from_mat.pkl',
        "EPOCHS": 400,
        "SAVE_PATH": "/home/student/Desktop/Groundeep/networks/zipfian/idbn_new_dataset",
        "SAVE_NAME": "dbn_trained_zipfian" #i or g is added on start depending on algorithm
    }
    # List of layer sizes to test
    layer_sizes_list = [ 
        [1000, 500],[1000, 1000], [1000, 1500], [1000, 2000], 
       [1500, 1000], [1500, 1500], [1500, 2000]
    ]

    #layer_sizes_list = [[1500, 500]]
    # Uncomment the following lines to test different configurations
    
    #layer_sizes_list = [[1500,500]]
    #layer_sizes_list = [[1500,1500]]

    # Load preprocessed training dataset
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_loader, val_loader, test_loader = create_dataloaders_zipfian("/home/student/Desktop/Groundeep/circle_dataset_100x100","circle_dataset_100x100_v1.npz", batch_size = 128, num_workers = 1)
    #train_loader, val_loader, test_loader = create_dataloaders_dunja("/home/student/Desktop/Groundeep/batched_train_data_from_mat.pkl", batch_size = 128, num_workers = 1,val_size = 0.02)


    #Train and save DBN for each configuration
    for layer_sizes in layer_sizes_list:
        #print(f"Training DBN with layer sizes: {layer_sizes}")
        layer_name = '_'.join(map(str, layer_sizes))
        
        if params["ALGORITHM"]=="g":
            save_path = f"{params['SAVE_PATH']}_g{params['SAVE_NAME']}_{layer_name}.pkl"
            os.makedirs(os.path.dirname(save_path), exist_ok=True)

            gdbn = gDBN([10000] +layer_sizes, params,train_loader,device,log_dir = "logs-gdbn/uniform")                 
            gdbn.train(epochs=params["EPOCHS"])
            gdbn.save(save_path)
        elif params["ALGORITHM"]=="i":
            save_path = f"{params['SAVE_PATH']}_i{params['SAVE_NAME']}_{layer_name}.pkl"
            os.makedirs(os.path.dirname(save_path), exist_ok=True)

            idbn = iDBN([10000] +layer_sizes, params,train_loader,val_loader,device,log_root="logs-idbn/zipfian")
            idbn.train(epochs=params["EPOCHS"])
            idbn.save_model(save_path)
        print(f"Saved trained DBN to {save_path}")

if __name__ == "__main__":
    main()
