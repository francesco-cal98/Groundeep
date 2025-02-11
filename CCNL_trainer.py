from gdbn_model import gDBN, iDBN
import torch
import pickle

def main():
    # Parameters to match MATLAB implementation
    params = {
        "ALGORITHM": "i",
        "LEARNING_RATE": 0.15,
        "WEIGHT_PENALTY": 0.0001,
        "INIT_MOMENTUM": 0.7,
        "FINAL_MOMENTUM": 0.97,  # Final momentum for dynamic momentum increase
        "LEARNING_RATE_DYNAMIC": True,
        "TRAINING_FILE_PATH":r'C:\Users\fraca\Desktop\Groundeep\latestcode\batched_train_data_from_mat.pkl',
        "EPOCHS": 200,
        "SAVE_NAME": "dbn_trained" #i or g is added on start depending on algorithm
    }
        # List of layer sizes to test
    # layer_sizes_list = [
    #     [500, 500], [500, 1000], [500, 1500], [500, 2000], 
    #     [1000, 500], [1000, 1000], [1000, 1500], [1000, 2000], 
    #     [1500, 500], [1500, 1000], [1500, 1500], [1500, 2000]
    # ]
    layer_sizes_list = [
        [1000, 500]
    ]

    # Load preprocessed training dataset
    with open(params["TRAINING_FILE_PATH"], 'rb') as f:
        train_dataset = pickle.load(f)
    
    # Convert NumPy array to PyTorch tensor
    train_data = torch.tensor(train_dataset['data'], dtype=torch.float32)
    train_data = train_data.to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))



    #Train and save DBN for each configuration
    for layer_sizes in layer_sizes_list:
        print(f"Training DBN with layer sizes: {layer_sizes}")
        layer_name = '_'.join(map(str, layer_sizes))
        save_path = f"{params['SAVE_NAME']}_{layer_name}.pkl"
        if params["ALGORITHM"]=="g":
            gdbn = gDBN([train_data.shape[2]] +layer_sizes, params)
            gdbn.train(train_data, epochs=params["EPOCHS"])
            gdbn.save("g" + save_path)
        elif params["ALGORITHM"]=="i":
            idbn = iDBN([train_data.shape[2]] +layer_sizes, params)
            idbn.train(train_data, epochs=params["EPOCHS"])
            idbn.save("i" + save_path)
     
        
        print(f"Saved trained DBN to {save_path}")

if __name__ == "__main__":
    main()
