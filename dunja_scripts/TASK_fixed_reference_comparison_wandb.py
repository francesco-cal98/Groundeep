import pickle
import numpy as np
import pandas as pd
from scipy import io
import scipy
import torch
from datasets_utils import single_stimuli_dataset_modified
import matplotlib.pyplot as plt
import h5py
import wandb
from matplotlib.cm import viridis
import matplotlib.pyplot as plt
import matplotlib.cm as cm

from CCNL_models import forwardDBN
from CLs import Pseudoinverse_fixed as Pseudoinverse
from CLs import SGD_class_fixed as SGD_classifier
from CLs import Logistic_class_fixed as Logistic_regression
from CLs import Ridge_class_fixed as Ridge_classifier
from CLs import beta_extraction_ref_z, compute_prob_choice
def load_mat_file(file_path):
    """Load MATLAB .mat file (supports both v7.3 HDF5 and older formats)."""
    try:
        data = scipy.io.loadmat(file_path)
        N_list = data['N_list']
        TSA_list = data['TSA_list']
        FA_list = data['FA_list']
    except NotImplementedError:
        with h5py.File(file_path, 'r') as f:
            N_list = np.array(f['N_list'])
            TSA_list = np.array(f['TSA_list'])
            FA_list = np.array(f['FA_list'])
    return N_list, TSA_list, FA_list

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(DEVICE)

distribution = 'uniform'
MIN_NUM = 1
MAX_NUM = 32
REF_NUMS = [14, 16]
ADJUSTED_LIMITS = 0
alg = "i"
runs = 1

output_file = f'D:/Dunja/Sep2024/NEW/results/fixed_ref_results_SEP25.xlsx'
# Start a main W&B run
wandb.init(
    project="dbn-reference-comparison",
    config={
        # "distribution": distribution,
        "alg": alg,
        "runs": runs,
        "ref_nums": REF_NUMS
    },
    name=f"reference_comparison_"
)
config = wandb.config

results_ref = []
plots = []

for distribution in ['uniform', 'zipfian']:
    for REF_NUM in REF_NUMS:
        # ---------------- Dataset prep ----------------
        if REF_NUM == 1:
            train_file = f'D:/Dunja/DBN-GPU-MATLAB/datasets/NumStim_7to28_100x100_TR.mat'
            test_file  = f'D:/Dunja/DBN-GPU-MATLAB/datasets/NumStim_7to28_100x100_TE.mat'
            LIMITS = np.array([0.49,2,4])
            PERCENTAGES = np.array([0, 100, 0])
        else:
            train_file = f'D:/Dunja/DBN-GPU-MATLAB/datasets/NumStim_{MIN_NUM}to{MAX_NUM}_100x100_TR.mat'
            test_file  = f'D:/Dunja/DBN-GPU-MATLAB/datasets/NumStim_{MIN_NUM}to{MAX_NUM}_100x100_TE.mat'
            LIMITS = np.array([0.49,2,4])
            PERCENTAGES = np.array([0, 100, 0])

        N_list_train, TSA_list_train, FA_list_train = load_mat_file(train_file)

        # Compute distribution percentages
        unique_N = np.unique(N_list_train)
        unique_ratios = unique_N / REF_NUM
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

        # Create datasets
        train_dataset = single_stimuli_dataset_modified(
            train_file, num_samples=15200, ref_num=REF_NUM,
            num_percentage_dict=distributed_percentages, binarize=True
        )
        test_dataset = single_stimuli_dataset_modified(
            test_file, num_samples=15200, ref_num=REF_NUM,
            num_percentage_dict=distributed_percentages, binarize=True
        )

        # ---------------- Loop over DBNs ----------------
        layer_sizes = [
            # [500, 500], [500, 1000], [500, 1500], [500, 2000],
            # [1000, 500], [1000, 1000], [1000, 1500], [1000, 2000],
            [1500, 500], [1500, 1000], [1500, 1500], [1500, 2000]
        ]
        classifiers = [SGD_classifier, Pseudoinverse]  # add others if needed
        dbns_path = f'D:/Dunja/Sep2024/NEW/{alg}dbns_SEP25/{distribution}/{alg}dbn_{distribution}_TE200'

        for run in range(1, runs+1):
            for classifier in classifiers:
                for layer_size in layer_sizes:

                    dbn_path= f"{dbns_path}_R{run}_{layer_size[0]}_{layer_size[1]}.pkl"
                    with open(dbn_path, 'rb') as f:
                        dbn = pickle.load(f)

                    XtrainComp, YtrainComp, idxs_train = (
                        train_dataset['data'].to(DEVICE),
                        train_dataset['labels'].to(DEVICE),
                        train_dataset['idxs'].to(DEVICE)
                    )
                    _XtrainComp = forwardDBN(dbn,XtrainComp).clone()
                    _YtrainComp = YtrainComp.clone()

                    XtestComp, YtestComp, idxs_test = (
                        test_dataset['data'].to(DEVICE),
                        test_dataset['labels'].to(DEVICE),
                        test_dataset['idxs'].to(DEVICE)
                    )
                    _XtestComp = forwardDBN(dbn, XtestComp).clone()
                    _YtestComp = YtestComp.clone()

                    accTR, predTR, accTE, predTE = classifier(_XtrainComp, _XtestComp, _YtrainComp, _YtestComp)

                    N_list_test,TSA_list_test,FA_list_test = load_mat_file(test_file)
                    

                    model_fit, betas, wf, X, num_ratios, prob_choice_higher, model, numZ, sizeZ, spaceZ, num_list, t_stats, p_vals, standard_errors = beta_extraction_ref_z(
                        predTE, idxs_test, N_list_test, TSA_list_test, FA_list_test, ref_num=REF_NUM
                    )

                    # Save results
                    res = {
                        'Distribution': distribution,
                        'reference': REF_NUM,
                        'classifier': classifier.__name__,
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
                    }
                    results_ref.append(res)

                    numZ_range = np.linspace(numZ.min(), numZ.max(), 100)
                    sizeZ_range = np.linspace(sizeZ.min(), sizeZ.max(), 100)
                    spaceZ_range = np.linspace(spaceZ.min(), spaceZ.max(), 100)

                    # Construct smooth predictor matrix
                    X = np.column_stack((
                        (numZ_range),
                        (np.full_like(numZ_range, 0)),
                        (np.full_like(numZ_range, 0))
                    ))

                    intercept = model['intercept']
                    betas = model['betas']



                    num_ratios_range = np.linspace(np.log2(num_ratios).min(), np.log2(num_ratios).max(), 100)

                    N_list = np.squeeze(N_list_test)
                    nums_unique = np.unique(N_list)
                    nums_mean = np.mean(nums_unique)
                    nums_std = np.std(nums_unique)
                    ref_num = REF_NUM

                    ref_num_z = (ref_num - nums_mean) / nums_std

                    numZ_ratios = numZ/ ref_num_z
                    numZ_ratios_range = np.linspace(numZ.min(), numZ.max(), 100)
                    zero = np.zeros_like(numZ_ratios_range)

                    S = np.column_stack((
                        (numZ_ratios_range),
                        (np.full_like(numZ_ratios_range, 0)),
                        (np.full_like(numZ_ratios_range, 0))
                    ))


                    unique_num_ratios = np.unique(np.log2(num_ratios))
                    avg_prob_choice_higher_ratios = [np.mean(prob_choice_higher[np.log2(num_ratios) == val]) for val in unique_num_ratios]
                    
                    plt.close('all')
                    fig, ax = plt.subplots(figsize=(12, 10))
                    ax.plot(unique_num_ratios, avg_prob_choice_higher_ratios, 'o', markersize=8, label='p(Choose Higher)', color=cm.viridis(0.2))

                    # plt.figure(figsize=(10, 6))
                    # plt.plot(np.log2(num_ratios), prob_choice_higher, 'o', markersize=5, label='Prob Choice Higher', color='#EDB88B')

                    x_vals_ratios = np.linspace(np.log2(num_ratios).min(), np.log2(num_ratios).max(), 100)
                    y_vals_ratios = compute_prob_choice(S, intercept, betas)

                    cl_name = classifier.__name__.split('_')[0]
                    ax.plot(x_vals_ratios, y_vals_ratios, '-', color="#00000083", label=f'Model Fit (CL: {cl_name})')

                    ax.set_xlabel('Numerosity Ratio (log2)',fontsize=38)
                    ax.set_ylabel("p(Choose Higher)", fontsize=38)
                    # ax.set_title("Probability of Choosing 'Higher' as a Function of Numerosity")
                    ax.legend(fontsize=20,title_fontsize=30)
                    ax.tick_params(labelsize=18)
                    ax.grid(True)

                    wandb.log({
                    **res,
                    f"GLM Fit/idbn_{distribution}_TE200_{classifier.__name__}_R{run}_{layer_size[0]}_{layer_size[1]}": wandb.Image(fig)
                    })
                    plt.close(fig)


                    plots.append({
                                    'distribution': distribution,
                                    'reference': REF_NUM,
                                    'classifier': classifier.__name__.split('_')[0],
                                    'layer_size': layer_size,
                                    'unique_num_ratios': unique_num_ratios,
                                    'avg_prob_choice': avg_prob_choice_higher_ratios,
                                    'x_vals': x_vals_ratios,
                                    'y_vals': y_vals_ratios
                                })

results_df = pd.DataFrame(results_ref)
#Log the dataframe as a W&B Table
wandb.log({"Results Table": wandb.Table(dataframe=results_df)})

results_df.to_excel(output_file, index=False)

wandb.save(output_file)
print(f"Results saved to {output_file}")

# Organize plots by layer size (so each figure = one layer size, but inside it classifiers+distributions are compared)
plots_per_layer = {}
for plot in plots:
    layer_key = f"{plot['layer_size'][0]}_{plot['layer_size'][1]}"
    if layer_key not in plots_per_layer:
        plots_per_layer[layer_key] = []
    plots_per_layer[layer_key].append(plot)


loop_over = "distributions"   # or "classifiers" or "references"
fixed_classifier = "SGD"      # e.g., "SGD", "Pseudoinverse", "Ridge", "Logistic"
fixed_distribution = "zipfian"
colors = plt.cm.viridis(np.linspace(0, 1, 3))  # more colors since references can be multiple

if loop_over == "distributions":
    # group by distribution, only keep chosen classifier
    plots_filtered = [p for p in plots if p['classifier'] == fixed_classifier]
    distributions = sorted(set(p['distribution'] for p in plots_filtered), reverse=True)

    for layer_size in sorted(set(tuple(p['layer_size']) for p in plots_filtered)):
        for ref in sorted(set(p['reference'] for p in plots_filtered)):
            fig, ax = plt.subplots(figsize=(12, 10))
            for i, dist in enumerate(distributions):
                for p in plots_filtered:
                    if (tuple(p['layer_size']) == layer_size and 
                        p['distribution'] == dist and 
                        p['reference'] == ref):
                        color = colors[i % len(colors)]
                        ax.plot(p['unique_num_ratios'], p['avg_prob_choice'], 'o', 
                                color=color, label=f"{dist}", markersize=14)
                        ax.plot(p['x_vals'], p['y_vals'], '-', color=color, alpha=0.8,label=f"GLM fit", markersize = 16)
            # ax.set_title(f"GLM Fits | {fixed_classifier}, Layer {layer_size[0]}_{layer_size[1]}, Ref={ref}", fontsize=28)
            ax.set_xlabel("Numerosity Ratios (log2)", fontsize=38)
            ax.set_ylabel("p(Choose Higher)", fontsize=38)
            ax.tick_params(axis='both', which='major', labelsize=18)
            ax.legend(fontsize=20)
            ax.grid(True)
            wandb.log({f"GLM_Fits_{fixed_classifier}_Layer_{layer_size[0]}_{layer_size[1]}_Ref{ref}": wandb.Image(fig)})
            plt.close(fig)

elif loop_over == "classifiers":
    # group by classifier, only keep chosen distribution
    plots_filtered = [p for p in plots if p['distribution'] == fixed_distribution]
    classifiers = sorted(set(p['classifier'] for p in plots_filtered))

    for layer_size in sorted(set(tuple(p['layer_size']) for p in plots_filtered)):
        for ref in sorted(set(p['reference'] for p in plots_filtered)):
            fig, ax = plt.subplots(figsize=(12, 10))
            for i, clf in enumerate(classifiers):
                for p in plots_filtered:
                    if (tuple(p['layer_size']) == layer_size and 
                        p['classifier'] == clf and 
                        p['reference'] == ref):
                        color = colors[i % len(colors)]
                        ax.plot(p['unique_num_ratios'], p['avg_prob_choice'], 'o', 
                                color=color, label=f"{clf} (empirical)", markersize=14)
                        ax.plot(p['x_vals'], p['y_vals'], '-', color=color, alpha=0.8)
            ax.set_title(f"GLM Fits | {fixed_distribution}, Layer {layer_size[0]}_{layer_size[1]}, Ref={ref}", fontsize=28)
            ax.set_xlabel("Numerosity Ratios (log2)", fontsize=24)
            ax.set_ylabel("p(Choose Higher)", fontsize=24)
            ax.tick_params(axis='both', which='major', labelsize=18)
            ax.legend(fontsize=18, title_fontsize=20)
            ax.grid(True)
            wandb.log({f"GLM_Fits_{fixed_distribution}_Layer_{layer_size[0]}_{layer_size[1]}_Ref{ref}": wandb.Image(fig)})
            plt.close(fig)

elif loop_over == "references":
    # focus only on references, across distributions and classifiers
    refs = sorted(set(p['reference'] for p in plots))

    for layer_size in sorted(set(tuple(p['layer_size']) for p in plots)):
        for ref in refs:
            fig, ax = plt.subplots(figsize=(12, 10))
            for i, p in enumerate(plots):
                if tuple(p['layer_size']) == layer_size and p['reference'] == ref:
                    color = colors[i % len(colors)]
                    ax.plot(p['unique_num_ratios'], p['avg_prob_choice'], 'o', 
                            color=color, label=f"{p['classifier']} | {p['distribution']}", markersize=14)
                    ax.plot(p['x_vals'], p['y_vals'], '-', color=color, alpha=0.8)
            ax.set_title(f"GLM Fits | Layer {layer_size[0]}_{layer_size[1]}, Ref={ref}", fontsize=28)
            ax.set_xlabel("Numerosity Ratios (log2)", fontsize=24)
            ax.set_ylabel("p(Choose Higher)", fontsize=24)
            ax.tick_params(axis='both', which='major', labelsize=18)
            ax.legend(fontsize=16, title_fontsize=20)
            ax.grid(True)
            wandb.log({f"GLM_Fits_Layer_{layer_size[0]}_{layer_size[1]}_Ref{ref}": wandb.Image(fig)})
            plt.close(fig)

wandb.finish()
