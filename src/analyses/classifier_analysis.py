import os
import torch
import pickle
import pandas as pd
import gc
import sys
from scipy import io
project_root = os.path.abspath(os.path.join(os.getcwd()))

# Add it to sys.path
sys.path.append(project_root)
module_path = os.path.join(project_root, "src")  # Adjust "src" based on actual location
sys.path.append(module_path)

from CCNL_readout_DBN import forwardrbm,forwardDBN,classifier, beta_extraction



# Your helper functions must be imported or defined:
# - forwardDBN
# - classifier
# - beta_extraction
def main():
    # Set device
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Paths
    path_to_dbn_dir = "/home/student/Desktop/Groundeep/networks/uniform"
    path_to_train_dataset = "/home/student/Desktop/Groundeep/pairs_from_mat_train.pkl"
    path_to_test_dataset = "/home/student/Desktop/Groundeep/pairs_from_mat_test.pkl"
    test_file = "/home/student/Desktop/Groundeep/NumStim_7to28_100x100_TE.mat"
    output_file = "model_coefficients_results_all_uniform.xlsx"

    # Load datasets
    train_dataset = pickle.load(open(path_to_train_dataset, 'rb'))
    XtrainComp, YtrainComp, idxs_train = train_dataset['data'].to(DEVICE), train_dataset['labels'].to(DEVICE), train_dataset['idxs'].to(DEVICE)

    test_dataset = pickle.load(open(path_to_test_dataset, 'rb'))
    XtestComp, YtestComp, idxs_test = test_dataset['data'].to(DEVICE), test_dataset['labels'].to(DEVICE), test_dataset['idxs'].to(DEVICE)

    # Load test MAT data
    test_contents = io.loadmat(test_file)
    N_list_test = test_contents['N_list']
    TSA_list_test = test_contents['TSA_list']
    FA_list_test = test_contents['FA_list']

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
            _YtestComp = YtestComp.clone()

            # Classifier
            accTR, predTR, acc, choice = classifier(_XtrainComp, _XtestComp, _YtrainComp, _YtestComp)

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
                'Weber Fraction': prob_choice_right,
                "Accuracy": acc
            })

            # Clean up
            del dbn, _XtrainComp, _XtestComp, _YtrainComp, _YtestComp
            torch.cuda.empty_cache()
            gc.collect()

    # Export all results
    df_results = pd.DataFrame(results)
    df_results.to_excel(output_file, index=False)
    print(f"\nAll results saved to {output_file}")

if __name__ == "__main__":
    main()

