import pickle
import pandas as pd
from scipy import io
import torch
import numpy as np
from CCNL_models import forwardDBN
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import accuracy_score
from scipy.stats import norm,t

def SGD_class_pairs(Xtrain, Xtest, Ytrain, Ytest):
    # Flatten the input features
    Xtrain = Xtrain.view(-1, Xtrain.shape[2]).cpu().detach().numpy() 
    Xtest = Xtest.view(-1, Xtest.shape[2]).cpu().detach().numpy()

    # Ensure Ytrain and Ytest are 2D (if not already)
    Ytrain = Ytrain.reshape(-1, Ytrain.shape[-1]).cpu().detach().numpy() 
    Ytest = Ytest.reshape(-1, Ytest.shape[-1]).cpu().detach().numpy() 

    # If Ytrain and Ytest have more than one column, extract the second column (index 1)
    Ytrain = Ytrain[:, 1].reshape(-1) 
    Ytest = Ytest[:, 1].reshape(-1)

    print(Xtrain.shape, Ytrain.shape, Xtest.shape, Ytest.shape )

    model = SGDClassifier(penalty='l2',max_iter=1000, random_state= 42)
    
    model.fit(Xtrain, Ytrain)

    predicted_train = model.predict(Xtrain)
    accuracy_train = accuracy_score(Ytrain, predicted_train)

    print(f'\naccuracy train: ' , accuracy_train)

    predicted_test = model.predict(Xtest)
    accuracy_test = accuracy_score(Ytest, predicted_test)

    print(f'accuracy test: ', accuracy_test)
    
    return accuracy_train, predicted_train, accuracy_test, predicted_test
def irls_fit(choice, X, guessRate=0.01, max_iter=5000, tol=1e-12):
    """
    Fits the custom probit model using Iteratively Reweighted Least Squares (IRLS).
    """
    response_rate = 1 - guessRate
    n_obs, n_features = X.shape
    beta = np.zeros(n_features + 1)  # Intercept + coefficients
    X_design = np.column_stack((np.ones(n_obs), X))  # Add intercept term

    # Debugging: Initial inputs
    # print('--- Input Debugging ---')
    # print(f'Number of trials: {n_obs}')
    # print(f'Guess Rate: {guessRate}')

    for iteration in range(max_iter):
        # Linear combination
        linear_combination = np.dot(X_design, beta)

        # Predicted probabilities
        prob = response_rate * (norm.cdf(linear_combination) - 0.5) + 0.5
        prob = np.clip(prob, 1e-15, 1 - 1e-15)  # Avoid log(0)

        # Weights for IRLS
        W = (response_rate * norm.pdf(linear_combination)) ** 2 / prob / (1 - prob)
 
        # Weighted residuals
        z = linear_combination + (choice - prob) / (response_rate * norm.pdf(linear_combination))

        # Update beta using weighted least squares
        WX = W[:, np.newaxis] * X_design
        beta_new = np.linalg.solve(np.dot(WX.T, X_design), np.dot(WX.T, z))

        # Check for convergence
        if np.linalg.norm(beta_new - beta) < tol:
            break

        beta = beta_new

    else:
        #raise ValueError("IRLS failed to converge within the maximum number of iterations")
        # Log a warning instead of raising an error
        print("Warning: IRLS did not converge within the maximum number of iterations")

    # Weber fraction
    weber = 1 / (np.sqrt(2) * beta[1])  # Assuming first coefficient is for numerosity

    model = {
        'intercept': beta[0],
        'betas': beta[1:],
        'weber': weber
    }

    # Compute standard errors
    cov_matrix = np.linalg.pinv(X_design.T @ (W[:, np.newaxis] * X_design))  # Pseudo-inverse for stability
    standard_errors = np.sqrt(np.diag(cov_matrix))

    # Compute t-statistics and p-values
    t_stats = beta / standard_errors
    p_values = 2 * t.sf(np.abs(t_stats), df=n_obs - n_features - 1)

    # Debugging: Coefficients
    print('--- Coefficients Debugging ---')
    print(f'Intercept: {beta[0]}')
    print(f'Betas: {beta[1:]}')
    print(f'Weber Fraction: {weber}')
    print(f'T-Statistics: {t_stats}')
    print(f'P-Values: {p_values}')

    return beta[0], beta[1:], weber, prob, model, t_stats, p_values, standard_errors
def num_size_spacing_model(choice, numLeft, numRight, isaLeft, isaRight, faLeft, faRight, guessRate=0.01):
    # Calculate left side features
    tsaLeft = isaLeft * numLeft
    sizeLeft = isaLeft * tsaLeft
    sparLeft = faLeft / numLeft
    spaceLeft = sparLeft * faLeft

    # Calculate right side features
    tsaRight = isaRight * numRight
    sizeRight = isaRight * tsaRight
    sparRight = faRight / numRight
    spaceRight = sparRight * faRight
   
    # Calculate ratios
    numRatio = (numRight / numLeft)
    sizeRatio = (sizeRight / sizeLeft)
    spaceRatio = (spaceRight / spaceLeft)

    # Regression matrix
    X = np.column_stack((np.log2(numRatio), np.log2(sizeRatio), np.log2(spaceRatio)))
    choice = np.array(choice)

    # Fit using IRLS
    intercept, betas, weber, prob, model, t_stats, p_vals, standard_errors = irls_fit(choice, X, guessRate)

    return intercept, betas, weber, X, prob, model,numRatio, sizeRatio, spaceRatio, t_stats, p_vals, standard_errors
def beta_extraction(choice, idxs, N_list, TSA_list, FA_list, guessRate=0.01):
    # Initialize lists for the inputs
    numLeft = []
    numRight = []
    isaLeft = []
    isaRight = []
    faLeft = []
    faRight = []

    # Flatten input arrays and extract pairs of indices
    N_list = np.squeeze(N_list)
    TSA_list = np.squeeze(TSA_list)
    FA_list = np.squeeze(FA_list)
    idxs_flat = idxs.view(-1, 2)

    # Populate lists for left and right parameters based on indices
    for idx_pair in idxs_flat:
        idx_left, idx_right = int(idx_pair[0])-1, int(idx_pair[1])-1

        numLeft.append(N_list[idx_left])
        numRight.append(N_list[idx_right])
        isaLeft.append(TSA_list[idx_left] / N_list[idx_left])
        isaRight.append(TSA_list[idx_right] / N_list[idx_right])
        faLeft.append(FA_list[idx_left])
        faRight.append(FA_list[idx_right])

    # Call the main model function to fit and calculate Weber fraction
    intercept, betas, weber, X, prob_choice_right, model, numRatio, sizeRatio, spaceRatio, t_stats, p_vals, standard_errors = num_size_spacing_model(
        choice, np.array(numLeft), np.array(numRight), np.array(isaLeft),
        np.array(isaRight), np.array(faLeft), np.array(faRight), guessRate
    )

    return intercept, betas, weber, X,prob_choice_right, model, numRatio, sizeRatio, spaceRatio, t_stats, p_vals, standard_errors

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

layer_sizes = [
     [500, 500], [1000, 500], [1500, 500]
]

# DBNs PATHS
distribution = 'zipfian'
dbn_alg = 'i'
dbns_path = f'D:/Dunja/Sep2024/NEW/{dbn_alg}dbns/{distribution}/idbn_{distribution}_TE300'

# DATASETS PATHS
"""
Each dataset contains 15200 pairs of images with labels, and idxs connecting each image to the TEST FILE 
"""
train_dataset_path = 'D:/Dunja/Sep2024/DeWind/new/datasets/pairs_from_mat_TR.pkl' 
test_dataset_path  = 'D:/Dunja/Sep2024/DeWind/new/datasets/pairs_from_mat_TE.pkl'

# TEST FILE with all the visual feature infromation (Numeoristy, Field Area, Convex Hull....)
test_file   = 'D:/Dunja/Sep2024/DeWind/new/datasets/NumStim_7to28_100x100_TE.mat'

output_file = f'comparison_task_results.xlsx'

classifiers = [SGD_class_pairs]

results = []

runs = 1

for run in range(1,runs+1):
    print(f'\nNETWORK {run}')

    for classifier in classifiers:
        print(f"Classifier: {classifier.__name__}\n")

        for layer_size in layer_sizes:
                dbn_path= f"{dbns_path}_R{run}_{layer_size[0]}_{layer_size[1]}.pkl"
                with open(dbn_path, 'rb') as f:
                    dbn = pickle.load(f)

                train_dataset = pickle.load(open(train_dataset_path, 'rb'))
                XtrainComp, YtrainComp, idxs_train = train_dataset['data'].to(DEVICE), train_dataset['labels'].to(DEVICE), train_dataset['idxs'].to(DEVICE)

                _XtrainComp1 = forwardDBN(dbn,XtrainComp[:, :, 0:10000]).clone()
                _YtrainComp = YtrainComp.clone()
                _XtrainComp2 = forwardDBN(dbn, XtrainComp[:, :, 10000:20000]).clone()
                _XtrainComp = torch.cat((_XtrainComp1, _XtrainComp2), 2)
                del _XtrainComp1, _XtrainComp2

                test_dataset = pickle.load(open(test_dataset_path, 'rb'))
                XtestComp, YtestComp, idxs_test = test_dataset['data'].to(DEVICE), test_dataset['labels'].to(DEVICE), test_dataset['idxs'].to(DEVICE)
                
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

                intercept, betas, weber, X, pred_prob_right, model, numRatio, sizeRatio, spaceRatio, t_stats, p_vals, standard_errors = beta_extraction(
                    choice, idxs_test, N_list_test, TSA_list_test, FA_list_test
                )

                results.append({
                    'Classifier': classifier.__name__,
                    'Layer Size': f"{layer_size[0]} {layer_size[1]}",
                    'Accuracy': acc,
                    'Weber Fraction': weber,
                    'Intercept': intercept,
                    'Beta Number': betas[0],
                    'Beta Size': betas[1],
                    'Beta Spacing': betas[2],
                    't-value Beta Side': t_stats[0],
                    't-value Beta Number': t_stats[1],
                    't-value Beta Size': t_stats[2],
                    't-value Beta Spacing': t_stats[3],
                    'p-value Beta Side': p_vals[0],
                    'p-value Beta Number': p_vals[1],
                    'p-value Beta Size': p_vals[2],
                    'p-value Beta Spacing': p_vals[3],
                    'SE Beta Side': standard_errors[0],
                    'SE Beta Number': standard_errors[1],
                    'SE Beta Size': standard_errors[2],
                    'SE Beta Spacing': standard_errors[3]
                })

        print(results) 


results_df = pd.DataFrame(results)
results_df.to_excel(output_file, index=False)

print(f"Results saved to {output_file}")