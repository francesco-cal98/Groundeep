from gdbn_model import gDBN, iDBN
import torch
import pickle
from src.datasets.zipfian_dataset import create_dataloaders_zipfian
from src.datasets.uniform_dataset import create_dataloaders_uniform



def main():
    # Parameters to match MATLAB implementation
    params = {
        "ALGORITHM": "i",
        "LEARNING_RATE": 0.015,
        "WEIGHT_PENALTY": 0.0001,
        "INIT_MOMENTUM": 0.7,
        "FINAL_MOMENTUM": 0.97,  # Final momentum for dynamic momentum increase
        "LEARNING_RATE_DYNAMIC": True,
        "TRAINING_FILE_PATH":r'/home/student/Desktop/Groundeep/batched_train_data_from_mat.pkl',
        "EPOCHS": 200,
        "SAVE_PATH": "/home/student/Desktop/Groundeep/networks/uniform/",
         "SAVE_NAME": "dbn_trained_uniform" #i or g is added on start depending on algorithm
    }
        # List of layer sizes to test
    """
        layer_sizes_list = [
         [1000, 1000], [1000, 1500], [1000, 2000], 
         [1500, 500], [1500, 1000], [1500, 1500], [1500, 2000]
     ]
    """
    
    layer_sizes_list = [
         [500, 500], [500, 1000], [500, 1500], [500, 2000], 
         [1000, 500],[1000, 1000], [1000, 1500], [1000, 2000], 
         [1500, 500], [1500, 1000], [1500, 1500], [1500, 2000]
         ]
    
    

    
    # Load preprocessed training dataset
    with open(params["TRAINING_FILE_PATH"], 'rb') as f:
        train_dataset = pickle.load(f)
    
    # Convert NumPy array to PyTorch tensor
    train_data = torch.tensor(train_dataset['data'], dtype=torch.float32)
    train_data = train_data.to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))

    #data_module = ZipfianDataModule(directory="/home/student/Desktop/Groundeep/training_tensors", batch_size=10, num_workers=8)
    #data_module.setup()
    #train_loader = data_module.train_dataloader()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    #train_loader, val_loader, test_loader = create_dataloaders_zipfian("/home/student/Desktop/Groundeep/training_tensors", batch_size = 128, num_workers = 1)
    train_loader, val_loader, test_loader = create_dataloaders_uniform("/home/student/Desktop/Groundeep/training_tensors/uniform/","NumStim_1to40_100x100_TR_uniform.npz", batch_size = 128, num_workers = 1)

    #Train and save DBN for each configuration
    for layer_sizes in layer_sizes_list:
        #print(f"Training DBN with layer sizes: {layer_sizes}")
        layer_name = '_'.join(map(str, layer_sizes))
        
        if params["ALGORITHM"]=="g":
            save_path = f"{params['SAVE_PATH']}_g{params['SAVE_NAME']}_{layer_name}.pkl"
            gdbn = gDBN([10000] +layer_sizes, params,train_loader,device)                 
            gdbn.train(epochs=params["EPOCHS"])
            gdbn.save(save_path)
        elif params["ALGORITHM"]=="i":
            save_path = f"{params['SAVE_PATH']}_i{params['SAVE_NAME']}_{layer_name}.pkl"
            idbn = iDBN([10000] +layer_sizes, params,train_loader,device,log_dir="logs-idbn/uniform")
            idbn.train(epochs=params["EPOCHS"])
            idbn.save(save_path)
        print(f"Saved trained DBN to {save_path}")

if __name__ == "__main__":
    main()
