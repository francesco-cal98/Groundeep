import os
import torch
import pickle
import pandas as pd
import gc
import sys
from scipy import io
import numpy as np # Importa numpy per salvare gli array

project_root = os.path.abspath(os.path.join(os.getcwd()))

# Add it to sys.path
sys.path.append(project_root)
module_path = os.path.join(project_root, "src")  # Adjust "src" based on actual location
sys.path.append(module_path)

from CCNL_readout_DBN import forwardrbm,forwardDBN,classifier, beta_extraction # Assicurati che queste funzioni siano disponibili

def main():
    # Set device
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Paths
    path_to_dbn_dir = "/home/student/Desktop/Groundeep/networks/uniform/dataset_10_10"
    path_to_train_dataset = "/home/student/Desktop/Groundeep/behavioral_datasets/binary_de_wind_train.pkl"
    path_to_test_dataset = "/home/student/Desktop/Groundeep/behavioral_datasets/binary_de_wind_test.pkl"
    test_file = "/home/student/Desktop/Groundeep/NumStim_7to28_100x100_TE.mat"
    output_folder = "/home/student/Desktop/Groundeep/outputs/results_excel_new_dataset"
    output_file = "model_coefficients_results_all_uniform.xlsx"
    os.makedirs(output_folder, exist_ok=True)

    # Load datasets
    train_dataset = pickle.load(open(path_to_train_dataset, 'rb'))
    XtrainComp = torch.tensor(train_dataset['data']).to(DEVICE)
    if float(train_dataset['data'].max()) > 1.0 + 1e-6:
        XtrainComp = XtrainComp / 255.0
    YtrainComp = torch.tensor(train_dataset['labels']).to(DEVICE)
    idxs_train = torch.tensor(train_dataset['idxs']).to(DEVICE)

    test_dataset = pickle.load(open(path_to_test_dataset, 'rb'))
    XtestComp = torch.tensor(test_dataset['data']).to(DEVICE)
    if float(test_dataset['data'].max()) > 1.0 + 1e-6:
        XtestComp = XtestComp / 255.0
    YtestComp = torch.tensor(test_dataset['labels']).to(DEVICE)
    idxs_test = torch.tensor(test_dataset['idxs']).to(DEVICE)

    # Load test MAT data
    test_contents = io.loadmat(test_file)
    N_list_test = test_contents['N_list']
    TSA_list_test = test_contents['TSA_list']
    FA_list_test = test_contents['FA_list']

    small_sizes_flag = True # Questa variabile è definita ma non usata

    results = []
    # Loop through each DBN file
    for file in os.listdir(path_to_dbn_dir):
        if file.endswith(".pkl"):
            dbn_path = os.path.join(path_to_dbn_dir, file)
            print(f"Processing {file}...")

            # Load DBN
            with open(dbn_path, 'rb') as f:
                dbn = pickle.load(f)
            
  
            # Forward pass through DBN
            _XtrainComp1 = forwardDBN(dbn, XtrainComp[:, :, 0:10000]).clone()
            _XtrainComp2 = forwardDBN(dbn, XtrainComp[:, :, 10000:20000]).clone()
            _XtrainComp = torch.cat((_XtrainComp1, _XtrainComp2), 2)
            _YtrainComp = YtrainComp.clone()

            _XtestComp1 = forwardDBN(dbn, XtestComp[:, :, 0:10000]).clone()
            _XtestComp2 = forwardDBN(dbn, XtestComp[:, :, 10000:20000]).clone()
            _XtestComp = torch.cat((_XtestComp1, _XtestComp2), 2)
            # Correzione: supposto un errore di battitura, dovrebbe essere YtestComp.clone()
            _YtestComp = YtestComp.clone() 

            # Classifier
            accTR, predTR, acc, choice, weights = classifier(_XtrainComp, _XtestComp, _YtrainComp, _YtestComp)

            # --- INIZIO AGGIUNTA PER SALVARE WEIGHTS ---
            weights_to_save = True
            if isinstance(weights, torch.Tensor):
                weights_to_save = weights.detach().cpu().numpy()
            elif isinstance(weights, np.ndarray):
                weights_to_save = weights
            else:
                print(f"Attenzione: 'weights' non è né un tensore PyTorch né un array NumPy per {file}. Salvataggio saltato.")

            if weights_to_save is not None:
                # Estrai il nome base del file (senza estensione .pkl)
                weights_filename_base = os.path.splitext(file)[0]
                # Costruisci il percorso completo del file di output
                weights_output_path = os.path.join(output_folder, f"{weights_filename_base}_weights.npy")
                # Salva l'array NumPy
                np.save(weights_output_path, weights_to_save)
                print(f"Weights per '{file}' salvati in '{weights_output_path}'")
            # --- FINE AGGIUNTA PER SALVARE WEIGHTS ---

            # Beta extraction
            model_fit, weber, prob_choice_right, _ = beta_extraction(
                choice, idxs_test, N_list_test, TSA_list_test, FA_list_test
            )

            # Save result
            results.append({
                'Network Name': file,
                'Intercept': model_fit,
                'Beta Number': weber[0],
                'Beta Size': weber[1],
                'Beta Spacing': weber[2],
                'Weber Fraction': prob_choice_right, # Verifica che 'prob_choice_right' sia effettivamente la Weber Fraction
                "Accuracy": acc
            })

            # Clean up
            del dbn, _XtrainComp, _XtestComp, _YtrainComp, _YtestComp
            torch.cuda.empty_cache()
            gc.collect()

    # Export all results
    df_results = pd.DataFrame(results)
    output_path = os.path.join(output_folder,output_file)
    df_results.to_excel(output_path, index=False)
    print(f"\nAll results saved to {output_file}")

if __name__ == "__main__":
    main()
