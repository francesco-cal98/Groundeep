import pickle
import torch
import pandas as pd
import gc
from scipy import io

import sys
import os
import gc
# Get the absolute path to the main project folder
project_root = os.path.abspath(os.path.join(os.getcwd()))

# Add it to sys.path
sys.path.append(project_root)
module_path = os.path.join(project_root, "src")  # Adjust "src" based on actual location
sys.path.append(module_path)

from CCNL_readout_DBN import forwardrbm,forwardDBN,classifier, beta_extraction




class NetworkComparison:
    def __init__(self, path_to_dbn1, name_of_dbn1, path_to_dbn2, name_of_dbn2, 
                 path_to_train_dataset, path_to_test_dataset, test_file, output_file):
        self.networks = {
            "network1": {
                "path": f"{path_to_dbn1}/{name_of_dbn1}.pkl",
                "results": []
            },
            "network2": {
                "path": f"{path_to_dbn2}/{name_of_dbn2}.pkl",
                "results": []
            }
        }
        self.path_to_train_dataset = path_to_train_dataset
        self.path_to_test_dataset = path_to_test_dataset
        self.test_file = test_file
        self.output_file = output_file
        self.layer_size = [1000, 500]
        self.train_dataset = None
        self.test_dataset = None
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    def load_data(self):
        self.train_dataset = pickle.load(open(self.path_to_train_dataset, 'rb'))
        self.test_dataset = pickle.load(open(self.path_to_test_dataset, 'rb'))
    
    def process_network(self, network_name):
        network = self.networks[network_name]
        with open(network["path"], 'rb') as f:
            dbn = pickle.load(f)
        
        XtrainComp, YtrainComp = self.train_dataset['data'].to(self.device), self.train_dataset['labels'].to(self.device)
        XtestComp, YtestComp = self.test_dataset['data'].to(self.device), self.test_dataset['labels'].to(self.device)
        
        _XtrainComp1 = forwardDBN(dbn, XtrainComp[:, :, 0:10000]).clone()
        _XtrainComp2 = forwardDBN(dbn, XtrainComp[:, :, 10000:20000]).clone()
        _XtrainComp = torch.cat((_XtrainComp1, _XtrainComp2), 2)
        del _XtrainComp1, _XtrainComp2

        _XtestComp1 = forwardDBN(dbn, XtestComp[:, :, 0:10000]).clone()
        _XtestComp2 = forwardDBN(dbn, XtestComp[:, :, 10000:20000]).clone()
        _XtestComp = torch.cat((_XtestComp1, _XtestComp2), 2)
        del _XtestComp1, _XtestComp2

        accTR, predTR, acc, choice = classifier(_XtrainComp, _XtestComp, YtrainComp, YtestComp)
        
        test_contents = io.loadmat(self.test_file)
        model_fit, weber, prob_choice_right, _ = beta_extraction(
            choice, self.test_dataset['idxs'], 
            test_contents['N_list'], test_contents['TSA_list'], test_contents['FA_list']
        )
        
        network["results"].append({
            'Network': network_name,
            'Layer Size': f"{self.layer_size[0]} {self.layer_size[1]}",
            'Intercept': model_fit,
            'Beta Number': weber[0],
            'Beta Size': weber[1],
            'Beta Spacing': weber[2],
            'Weber Fraction': prob_choice_right,
            "Accuracy": acc
        })
        
        del dbn, XtrainComp, YtrainComp, XtestComp, YtestComp
        del _XtrainComp, _XtestComp
        torch.cuda.empty_cache()
        gc.collect()

    def run_comparison(self):
        self.load_data()
        for network_name in self.networks:
            self.process_network(network_name)
        
        all_results = self.networks["network1"]["results"] + self.networks["network2"]["results"]
        results_df = pd.DataFrame(all_results)
        results_df.to_excel(self.output_file, index=False)
        print(f"Comparison results saved to {self.output_file}")

# Example Usage
comparison = NetworkComparison(
    path_to_dbn1="/home/student/Desktop/Groundeep/networks/zipfian",
    name_of_dbn1="idbn_trained_zipfian_500_500",
    path_to_dbn2="/home/student/Desktop/Groundeep/networks/non_zipfian",
    name_of_dbn2="idbn_trained_500_500",
    path_to_train_dataset='/home/student/Desktop/Groundeep/pairs_from_mat_train.pkl',
    path_to_test_dataset='/home/student/Desktop/Groundeep/pairs_from_mat_test.pkl',
    test_file='/home/student/Desktop/Groundeep/NumStim_1to32_100x100_TE.mat',
    output_file='model_comparison_results.xlsx'
)
comparison.run_comparison()
