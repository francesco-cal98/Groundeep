import pickle
import pandas as pd
from scipy import io
import torch
import numpy as np
import wandb
from CCNL_models import forwardDBN
from CLs import Pseudoinverse_pairs as Pseudoinverse
from CLs import SGD_class_pairs as SGD_classifier
from CLs import Logistic_class_pairs as Logistic_regression
from CLs import Ridge_class_pairs as Ridge_classifier
from CLs import beta_extraction, compute_prob_choice
from matplotlib.cm import viridis
import matplotlib.pyplot as plt
import matplotlib.cm as cm
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

dbns_training_distribution =[ 'uniform', 'zipfian']
task = 'comparison_task' # 'comparison_task', 'fixed_reference_comparison', 'estimation_task'

results = []

plots = []


wandb.init(
    project="dbn-comparison",
    config={
        # "distribution": distribution,
        "dbn_alg": "i",
        "runs": 1
    },
    # name=f"comparison_task_{distribution}"
        name=f"comparison_task_all"

)
config = wandb.config

layer_sizes = [
    # [500, 500], [1000, 500], [1500, 500],
    # [500, 1000], [500, 1500], [500, 2000], 
    # [1000, 1000], [1000, 1500], [1000, 2000], 
   [1500, 1000], 
    # [1500, 1500], 
    # [1500, 2000]
]

for distribution in dbns_training_distribution:

    dbns_path = f'D:/Dunja/Sep2024/NEW/{config.dbn_alg}dbns_SEP25/{distribution}/idbn_{distribution}_TE200'
    train_dataset_path = 'D:/Dunja/Sep2024/DeWind/new/datasets/binary_de_wind_train.pkl'
    test_dataset_path  = 'D:/Dunja/Sep2024/DeWind/new/datasets/binary_de_wind_test.pkl'
    test_file   = 'D:/Dunja/Sep2024/DeWind/new/datasets/NumStim_7to28_100x100_TE.mat'

    output_file = f'D:/Dunja/Sep2024/NEW/results/{task}_results_SEP25.xlsx'

    classifiers = [
        SGD_classifier,
        # Pseudoinverse, 
        # Ridge_classifier,Logistic_regression
                    ] 

    for run in range(1, config.runs+1):
        print(f'\nNETWORK {run}')

        for classifier in classifiers:
            print(f"Classifier: {classifier.__name__}\n")

            for layer_size in layer_sizes:
                dbn_path= f"{dbns_path}_R{run}_{layer_size[0]}_{layer_size[1]}.pkl"

                with open(dbn_path, 'rb') as f:
                    dbn = pickle.load(f)
                        
                train_dataset = pickle.load(open(train_dataset_path, 'rb'))
                XtrainComp = torch.tensor(train_dataset['data']).to(DEVICE)
                YtrainComp = torch.tensor(train_dataset['labels']).to(DEVICE)
                idxs_train = torch.tensor(train_dataset['idxs']).to(DEVICE)

                _XtrainComp1 = forwardDBN(dbn,XtrainComp[:, :, 0:10000]).clone()
                _YtrainComp = YtrainComp.clone()
                _XtrainComp2 = forwardDBN(dbn, XtrainComp[:, :, 10000:20000]).clone()
                _XtrainComp = torch.cat((_XtrainComp1, _XtrainComp2), 2)
                del _XtrainComp1, _XtrainComp2

                test_dataset = pickle.load(open(test_dataset_path, 'rb'))
                XtestComp = torch.tensor(test_dataset['data']).to(DEVICE)
                YtestComp = torch.tensor(test_dataset['labels']).to(DEVICE)
                idxs_test = torch.tensor(test_dataset['idxs']).to(DEVICE)
                
                _XtestComp1 = forwardDBN(dbn, XtestComp[:, :, 0:10000]).clone()
                _YtestComp = YtestComp.clone()
                _XtestComp2 = forwardDBN(dbn, XtestComp[:, :, 10000:20000]).clone()
                _XtestComp = torch.cat((_XtestComp1, _XtestComp2), 2)
                del _XtestComp1, _XtestComp2

                accTR, predTR, acc, choice = classifier(_XtrainComp,  _XtestComp, _YtrainComp, _YtestComp)

                test_contents = io.loadmat(test_file)
                N_list_test = test_contents['N_list']
                TSA_list_test = test_contents['TSA_list']
                FA_list_test = test_contents['FA_list']

                model_fit, betas, weber, X, prob_choice_right, model, numRatio, sizeRatio, spaceRatio, t_stats, p_vals, standard_errors = beta_extraction(
                    choice, idxs_test, N_list_test, TSA_list_test, FA_list_test
                )

                res = {
                    'Distribution': distribution,
                    'Classifier': classifier.__name__,
                    'Layer Size': f"{layer_size[0]} {layer_size[1]}",
                    'Intercept': model_fit,
                    'Beta Number': betas[0],
                    'Beta Size': betas[1],
                    'Beta Spacing': betas[2],
                    'Weber Fraction': weber,
                    'Accuracy': acc,
                    'Run': run,
                    't-value Beta Number': t_stats[1],
                    't-value Beta Size': t_stats[2],
                    't-value Beta Spacing': t_stats[3],
                    'p-value Beta Number': p_vals[1],
                    'p-value Beta Size': p_vals[2],
                    'p-value Beta Spacing': p_vals[3]
                }
                results.append(res)

                            # ---- GLM PLOT ----
                numRatio_range = np.linspace(np.log2(numRatio).min(), np.log2(numRatio).max(), 100)

                # smooth predictor matrix (only numerosity varies)
                X_smooth = np.column_stack((
                    numRatio_range,
                    np.full_like(numRatio_range, 0),
                    np.full_like(numRatio_range, 0)
                ))

                intercept = model['intercept']
                betas_fit = model['betas']

                fig, ax = plt.subplots(figsize=(12, 10))

                unique_num_ratios = np.unique(np.log2(numRatio))
                avg_prob_choice = [np.mean(prob_choice_right[np.log2(numRatio) == val]) for val in unique_num_ratios]
                ax.plot(unique_num_ratios, avg_prob_choice, 'o', markersize=8, label='p(Choose Right)', color=cm.viridis(0.2))

                # model fit
                x_vals = np.linspace(np.log2(numRatio).min(), np.log2(numRatio).max(), 100)
                y_vals = compute_prob_choice(X_smooth, intercept, betas_fit)
                ax.plot(x_vals, y_vals, '-', color="#00000083", label='Model Fit')

                ax.set_xlabel('Numerosity Ratio (log2)',fontsize=38)
                ax.set_ylabel("p(Choose Right)", fontsize=38)
                # ax.set_title("Probability of Choosing 'Right' as a Function of Numerosity", fontsize=38)
                ax.legend(fontsize=20,title_fontsize=30)
                ax.tick_params(labelsize=18)
                ax.grid(True)

                # log to wandb
                wandb.log({
                    **res,
                    f"GLM Fit/idbn_{distribution}_TE200_{classifier.__name__}_R{run}_{layer_size[0]}_{layer_size[1]}": wandb.Image(fig)
                })


                plt.close(fig)

                plots.append({
                    'distribution': distribution,
                    'classifier': classifier.__name__.split('_')[0],
                    'layer_size': layer_size,
                    'unique_num_ratios': unique_num_ratios,
                    'avg_prob_choice': avg_prob_choice,
                    'x_vals': x_vals,
                    'y_vals': y_vals
                })


results_df = pd.DataFrame(results)

# Log the dataframe as a W&B Table
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


loop_over = "distributions"   # or "classifiers"
fixed_classifier = "SGD"      # match with plots entries (e.g., "SGD", "Pseudoinverse", "Ridge", "Logistic")
fixed_distribution = "zipfian"
colors = plt.cm.viridis(np.linspace(0, 1, 3))

if loop_over == "distributions":
    # group by distribution, only keep chosen classifier
    plots_filtered = [p for p in plots if p['classifier'] == fixed_classifier]
    distributions = sorted(set(p['distribution'] for p in plots_filtered), reverse=True)

    for layer_size in sorted(set(tuple(p['layer_size']) for p in plots_filtered)):
        fig, ax = plt.subplots(figsize=(12, 10))
        for i, dist in enumerate(distributions):
            for p in plots_filtered:
                if tuple(p['layer_size']) == layer_size and p['distribution'] == dist:
                    color = colors[i % len(colors)]
                    ax.plot(p['unique_num_ratios'], p['avg_prob_choice'], 'o', color=color,
                            label=f"{dist}", markersize=14)
                    ax.plot(p['x_vals'], p['y_vals'], '-', color=color, alpha=0.8, markersize=16, label=f"GLM fit")
        # ax.set_title(f"GLM Fits for {fixed_classifier}, Layer {layer_size[0]}_{layer_size[1]}", fontsize=38)
        ax.set_xlabel("Numerosity Ratios (log2)", fontsize=38)
        ax.set_ylabel("p(Choose Right)", fontsize=38)
        ax.tick_params(axis='both', which='major', labelsize=18)  # tick font size
        ax.legend(fontsize=20,title_fontsize=30)
        ax.grid(True)
        wandb.log({f"GLM_Fits_{fixed_classifier}_Layer_{layer_size[0]}_{layer_size[1]}": wandb.Image(fig)})
        plt.close(fig)

elif loop_over == "classifiers":
    # group by classifier, only keep chosen distribution
    plots_filtered = [p for p in plots if p['distribution'] == fixed_distribution]
    classifiers = sorted(set(p['classifier'] for p in plots_filtered))

    for layer_size in sorted(set(tuple(p['layer_size']) for p in plots_filtered)):
        fig, ax = plt.subplots(figsize=(12, 10))
        for i, clf in enumerate(classifiers):
            for p in plots_filtered:
                if tuple(p['layer_size']) == layer_size and p['classifier'] == clf:
                    color = colors[i % len(colors)]
                    ax.plot(p['unique_num_ratios'], p['avg_prob_choice'], 'o', color=color,
                            label=f"{clf} (empirical)", markersize=16)
                    ax.plot(p['x_vals'], p['y_vals'], '-', color=color, alpha=0.8,markersize=16, label=f"GLM fit")
        ax.set_title(f"GLM Fits for {fixed_distribution}, Layer {layer_size[0]}_{layer_size[1]}", fontsize=38)
        ax.set_xlabel("Numerosity Ratios (log2)",  fontsize=38)
        ax.set_ylabel("p(Choose Right)",  fontsize=38)
        ax.legend(fontsize=20,title_fontsize=30)
        ax.tick_params(labelsize=18)  # tick font size
        ax.grid(True)
        wandb.log({f"GLM_Fits_{fixed_distribution}_Layer_{layer_size[0]}_{layer_size[1]}": wandb.Image(fig)})
        plt.close(fig)


wandb.finish()
