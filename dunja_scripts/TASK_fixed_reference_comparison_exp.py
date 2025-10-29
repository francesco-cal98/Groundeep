import pickle
import numpy as np
import pandas as pd
from scipy import io
import scipy
import torch
from datasets_utils import single_stimuli_dataset_modified
import matplotlib.pyplot as plt
import h5py
from CCNL_models import forwardDBN
from CLs import Pseudoinverse_fixed as Pseudoinverse
from CLs import SGD_class_fixed as SGD_classifier
from CLs import Logistic_class_fixed as Logistic_regression
from CLs import Ridge_class_fixed as Ridge_classifier
from CLs import beta_extraction_ref_z, compute_prob_choice

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(DEVICE)

distribution = 'uniform'

MIN_NUM = 1
MAX_NUM = 32
REF_NUMS = [8,14,16,20]

ADJUSTED_LIMITS = 0
alg = "i"

plots = []
results_ref =[]

train_file = f'D:/Dunja/DBN-GPU-MATLAB/datasets/NumStim_{MIN_NUM}to{MAX_NUM}_100x100_TR.mat'
test_file = f'D:/Dunja/DBN-GPU-MATLAB/datasets/NumStim_{MIN_NUM}to{MAX_NUM}_100x100_TE.mat'

output_file = f'D:/Dunja/Sep2024/NEW/results/fixed_ref_results_SEP25.xlsx'

for REF_NUM in REF_NUMS:

        if ADJUSTED_LIMITS:
            LIMITS = np.array([0.49,1,2,4])
            PERCENTAGES = np.array([0, 50, 50, 0])
        else:
            LIMITS = np.array([0.49,2,4])
            PERCENTAGES = np.array([0, 100, 0])

        try:
            train_contents = scipy.io.loadmat(train_file)
            N_list_train = train_contents['N_list']
            TSA_list_train = train_contents['TSA_list']
            FA_list_train = train_contents['FA_list']
        except NotImplementedError:
            with h5py.File(train_file, 'r') as mat_file:
                N_list_train = mat_file['N_list'][()]
                TSA_list_train = mat_file['TSA_list'][()]
                FA_list_train = mat_file['FA_list'][()]

        unique_N = np.unique(N_list_train)
        print(f"Unique numerosity values: {unique_N}")
        print(f'Number of unique numerosities: {len(unique_N)}')

        mean_N = np.mean(N_list_train)
        unique_mean_N = np.mean(unique_N)
        print(f"\nMean of all numerosities: {mean_N}")
        print(f"Mean of unique numerosities: {unique_mean_N}")

        median_N = np.median(N_list_train)
        median_N_unique = np.median(unique_N)
        print(f"\nMedian of all numerosities: {median_N}")
        print(f"Median of unique numerosities: {median_N_unique}") 

        print(f"\nReference numerosity value: {REF_NUM}")

        unique_ratios = unique_N / REF_NUM
        num_ratios_x = np.log2(unique_N / REF_NUM)
        print(f'\nAll possible number ratios: {unique_ratios}')
        print(f'\nAll possible number ratios (log2): {num_ratios_x}')

        ratios_range = (str(np.min(unique_ratios)), str(np.max(unique_ratios)))
        print(f"\nRatios range: {ratios_range}")

        distributed_percentages = {n: 0 for n in unique_N}

        for i in range(len(LIMITS)-1):
            lower, upper = LIMITS[i], LIMITS[i + 1]
            mask = (unique_ratios > lower) & (unique_ratios <= upper)

            valid_idxs = [idx for idx in np.where(mask)[0] if unique_N[idx] != REF_NUM]
            count = len(valid_idxs)

            if count > 0:
                for idx in np.where(mask)[0]:
                    distributed_percentages[unique_N[idx]] = PERCENTAGES[i + 1] / count
    
        distributed_percentages[REF_NUM] = 0

        print(distributed_percentages)

        #Create new single stimuli dataset

        train_dataset = single_stimuli_dataset_modified(train_file, num_samples=15200, ref_num=REF_NUM, num_percentage_dict=distributed_percentages, binarize = True)
        test_dataset = single_stimuli_dataset_modified(test_file, num_samples=15200, ref_num=REF_NUM, num_percentage_dict=distributed_percentages, binarize = True)  

        print("Train dataset data shape:", train_dataset['data'].shape)
        print("Train dataset labels shape:", train_dataset['labels'].shape)
        print("Train dataset idxs shape:", train_dataset['idxs'].shape)

        # ##############################################################
        # # And save the dataset
        # output_path_tr = f'D:/Dunja/Sep2024/DeWind/new/datasets/single_stimuli_dataset_REF_{REF_NUM}_TR.pkl'
        # output_path_te = f'D:/Dunja/Sep2024/DeWind/new/datasets/single_stimuli_dataset_REF_{REF_NUM}_TE.pkl'

        # with open(output_path_tr, 'wb') as f:
        #     pickle.dump(train_dataset, f)

        # with open(output_path_te, 'wb') as f:
        #     pickle.dump(test_dataset, f)


        #############################################################
        # OR load a preexisting single stimuli dataset
        

        # train_dataset = pickle.load(open(f'D:/Dunja/Sep2024/DeWind/new/datasets/single_stimuli_dataset_REF_{REF_NUM}_TR.pkl', 'rb'))
        # test_dataset = pickle.load(open(f'D:/Dunja/Sep2024/DeWind/new/datasets/single_stimuli_dataset_REF_{REF_NUM}_TE.pkl', 'rb'))

        ###############################################################

        layer_sizes =[
            [500, 500], 
            [500, 1000], 
            [500, 1500], 
            [500, 2000], 
            [1000, 500], [1000, 1000], [1000, 1500], 
            [1000, 2000], 
            [1500, 500], 
            [1500, 1500], 
            [1500, 2000],
            [1500, 1000]
            ]
        
        classifiers = [
            # #  Logistic_regression, 
            SGD_classifier, 
            #  Ridge_classifier, 
            # Pseudoinverse
            ]
        
        runs = 1

        dbns_path = f'D:/Dunja/Sep2024//NEW/{alg}dbns_SEP25/{distribution}/{alg}dbn_{distribution}_TE200'

        for run in range(1,runs+1):
            for classifier in classifiers:
                for layer_size in layer_sizes:

                        dbn_path= f"{dbns_path}_R{run}_{layer_size[0]}_{layer_size[1]}.pkl"

                        with open(dbn_path, 'rb') as f:
                            dbn = pickle.load(f)

                        XtrainComp, YtrainComp, idxs_train = train_dataset['data'].to(DEVICE), train_dataset['labels'].to(DEVICE), train_dataset['idxs'].to(DEVICE)
                        _XtrainComp = forwardDBN(dbn,XtrainComp).clone()
                        _YtrainComp = YtrainComp.clone()

                        XtestComp, YtestComp, idxs_test = test_dataset['data'].to(DEVICE), test_dataset['labels'].to(DEVICE), test_dataset['idxs'].to(DEVICE)
                        _XtestComp = forwardDBN(dbn, XtestComp).clone()
                        _YtestComp = YtestComp.clone()

                        print(_XtrainComp.shape, _YtrainComp.shape, _XtestComp.shape, _YtestComp.shape)

                        accTR, predTR, accTE, predTE = classifier(_XtrainComp, _XtestComp, _YtrainComp, _YtestComp)

                        try:
                            test_contents = io.loadmat(test_file)
                            N_list_test = test_contents['N_list']
                            TSA_list_test = test_contents['TSA_list']
                            FA_list_test = test_contents['FA_list']
                        except NotImplementedError:
                            with h5py.File(test_file, 'r') as mat_file:
                                N_list_test = mat_file['N_list'][()]
                                TSA_list_test = mat_file['TSA_list'][()]
                                FA_list_test = mat_file['FA_list'][()]

                        # Run the model for the current layer size
                        model_fit, betas, wf, X, num_ratios, prob_choice_higher, model, numZ, sizeZ, spaceZ, num_list, t_stats, p_vals, standard_errors = beta_extraction_ref_z(
                            predTE, idxs_test, N_list_test, TSA_list_test, FA_list_test, ref_num=REF_NUM 
                        )

                        # Collect results for the current layer size
                        results_ref.append({
                            'reference': REF_NUM,
                            'classifier':classifier.__name__,
                            'Layer Size': f"{layer_size[0]} {layer_size[1]}",
                            'Intercept': model_fit,
                            'Beta Number': betas[0],
                            'Beta Size': betas[1],
                            'Beta Spacing': betas[2],
                            'Weber Fraction': wf,
                            "Accuracy": accTE,
                            'Run': run,
                            't-value Beta Number': t_stats[0],
                            't-value Beta Size': t_stats[1],
                            't-value Beta Spacing': t_stats[2],
                            'p-value Beta Number': p_vals[0],
                            'p-value Beta Size': p_vals[1],
                            'p-value Beta Spacing': p_vals[2]
                        })

#Convert results to a DataFrame
print(results_ref)
results_ref_df = pd.DataFrame(results_ref)
results_ref_df.to_excel(output_file, index=False)

print(f"Results saved to {output_file}")