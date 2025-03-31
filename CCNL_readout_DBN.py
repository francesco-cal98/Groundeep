

import pickle
import numpy as np
import pandas as pd
from scipy import io
from scipy import io
import scipy
from scipy.stats import norm
import torch

import numpy as np
import statsmodels.api as sm
from sklearn.metrics import accuracy_score
import numpy as np
from sklearn.metrics import accuracy_score
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def forwardrbm(self, v):
    p_h = torch.sigmoid(torch.matmul(v, self.W) + self.hid_bias)
    h = (p_h > torch.rand_like(p_h)).float()  # Stochastic activation
    return p_h, h
def forwardDBN(self, X):
    for rbm in self.layers:
        # Ensure tensors are created on the correct device
        _X = torch.zeros([X.shape[0], X.shape[1], rbm.num_hidden], device=DEVICE)
        Xtorch = torch.zeros(X.shape[1], rbm.num_hidden, device=DEVICE)  # Intermediate tensor

        # Process each sample in the batch
        batch_indices = list(range(X.shape[0]))
        for n in batch_indices:
            Xtorch = torch.Tensor(X[n, :, :]).to(DEVICE)  # Get a single sample
            _X[n, :, :] = forwardrbm(rbm, Xtorch.clone())[0].clone()  # Store the transformed sample
        #end

        # Free up memory used by the intermediate tensor
        del Xtorch
        X = _X.clone()  # Update X with the transformed batch
        del _X  # Free memory used by the temporary batch tensor
    #end
    
    return X

def classifier(Xtrain, Xtest, Ytrain, Ytest):
      # Reshaping the data
    Xtrain = Xtrain.view(-1, Xtrain.shape[2]).detach().cpu().numpy()
    Xtest = Xtest.view(-1, Xtest.shape[2]).detach().cpu().numpy()
    
    Ytrain = Ytrain.view(-1, Ytrain.shape[-1]).detach().cpu().numpy()
    Ytest = Ytest.view(-1, Ytest.shape[-1]).detach().cpu().numpy()
    print(Xtrain.shape, Ytrain.shape, Xtest.shape, Ytest.shape )
    # Add bias term (biases are analogous to adding ones in the perceptron implementation)
    Xtrain = np.hstack([Xtrain, np.ones((Xtrain.shape[0], 1))])
    Xtest = np.hstack([Xtest, np.ones((Xtest.shape[0], 1))])
    
    # Train weights using pseudo-inverse
    #weights = np.linalg.lstsq(Xtrain, Xtest, rcond=None)
    weights =np.linalg.pinv(Xtrain) @ Ytrain
    # Predictions for training and testing data
    pred_train = Xtrain @ weights
    pred_test = Xtest @ weights
    
    # Convert predictions to class labels
    predicted_train = np.argmax(pred_train, axis=1)
    predicted_test = np.argmax(pred_test, axis=1)
    Ytrain_labels = np.argmax(Ytrain, axis=1)
    Ytest_labels = np.argmax(Ytest, axis=1)

    # Calculate accuracy
    accuracy_train = accuracy_score(Ytrain_labels, predicted_train)
    accuracy_test = accuracy_score(Ytest_labels, predicted_test)

    print('Pseudo-Inverse Perceptron:')
    print('Train Accuracy:', accuracy_train)
    print('Test Accuracy:', accuracy_test)

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
        raise ValueError("IRLS failed to converge within the maximum number of iterations")

    # Weber fraction
    weber = 1 / (np.sqrt(2) * beta[1])  # Assuming first coefficient is for numerosity

    # Debugging: Coefficients
    print('--- Coefficients Debugging ---')
    print(f'Intercept: {beta[0]}')
    print(f'Betas: {beta[1:]}')
    print(f'Weber Fraction: {weber}') #beta number , beta size, beta spacing

    return beta[0], beta[1:], weber

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

    # Debugging: Left side features
    # print('--- Left Side Features ---')
    # print(f'Total Surface Area (Left): {np.sum(tsaLeft)}')
    # print(f'Size (Left): {np.sum(sizeLeft)}')
    # print(f'Sparsity (Left): {np.sum(sparLeft)}')
    # print(f'Spacing (Left): {np.sum(spaceLeft)}')

    # Debugging: Right side features
    # print('--- Right Side Features ---')
    # print(f'Total Surface Area (Right): {np.sum(tsaRight)}')
    # print(f'Size (Right): {np.sum(sizeRight)}')
    # print(f'Sparsity (Right): {np.sum(sparRight)}')
    # print(f'Spacing (Right): {np.sum(spaceRight)}')

    # Calculate ratios
    numRatio = (numRight / numLeft)
    sizeRatio = (sizeRight / sizeLeft)
    spaceRatio = (spaceRight / spaceLeft)

    # Debugging: Ratios
    # print('--- Ratios ---')
    # print(f'Numerosity Ratio: {np.mean(numRatio)}')
    # print(f'Size Ratio: {np.mean(sizeRatio)}')
    # print(f'Spacing Ratio: {np.mean(spaceRatio)}')

    # Regression matrix
    X = np.column_stack((np.log2(numRatio), np.log2(sizeRatio), np.log2(spaceRatio)))
    choice = np.array(choice)

    # Fit using IRLS
    intercept, betas, weber = irls_fit(choice, X, guessRate)

    return intercept, betas, weber, X

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

    # Debugging: Inputs for left and right parameters
    # print('--- Input Debugging ---')
    # print(f'Number of trials: {len(choice)}')
    # print(f'Choice vector: {np.sum(choice == 0)} left, {np.sum(choice == 1)} right')

    # Populate lists for left and right parameters based on indices
    for idx_pair in idxs_flat:
        idx_left, idx_right = int(idx_pair[0]), int(idx_pair[1])

        numLeft.append(N_list[idx_left])
        numRight.append(N_list[idx_right])
        isaLeft.append(TSA_list[idx_left] / N_list[idx_left])
        isaRight.append(TSA_list[idx_right] / N_list[idx_right])
        faLeft.append(FA_list[idx_left])
        faRight.append(FA_list[idx_right])

    # Call the main model function to fit and calculate Weber fraction
    model_fit, weber, prob_choice_right, X = num_size_spacing_model(
        choice, np.array(numLeft), np.array(numRight), np.array(isaLeft),
        np.array(isaRight), np.array(faLeft), np.array(faRight), guessRate
    )

    return model_fit, weber, prob_choice_right, X

"""
layer_sizes = [
    [500, 500], [500, 1000], [500, 1500], [500, 2000], 
    [1000, 500], [1000, 1000], [1000, 1500], [1000, 2000], 
    [1500, 500], [1500, 1000], [1500, 1500], [1500, 2000]
]
path_to_dbn="/home/student/Desktop/Groundeep"
name_of_dbn="idbn_trained_zipfian_1000_500.pkl"
path_to_train_dataset='/home/student/Desktop/Groundeep/pairs_from_mat_train.pkl'
path_to_test_dataset= '/home/student/Desktop/Groundeep/test_datasetx.pkl'
test_file = '/home/student/Desktop/Groundeep/NumStim_1to32_100x100_TE.mat'
output_file = 'model_coefficients_results_5.xlsx'
# Prepare the results for saving into an Excel file
results = []


train_dataset = pickle.load(open(path_to_train_dataset, 'rb'))
XtrainComp, YtrainComp, idxs_train = train_dataset['data'].to(DEVICE), train_dataset['labels'].to(DEVICE), train_dataset['idxs'].to(DEVICE)
test_dataset = pickle.load(open(path_to_test_dataset, 'rb'))
XtestComp, YtestComp, idxs_test = test_dataset['data'].to(DEVICE), test_dataset['labels'].to(DEVICE), test_dataset['idxs'].to(DEVICE)
for layer_size in layer_sizes:
    from scipy import io

    input_path= f"{path_to_dbn}/{name_of_dbn}_{layer_size[0]}_{layer_size[1]}.pkl"
    with open(input_path, 'rb') as f:
        dbn = pickle.load(f)

    _XtrainComp1 = forwardDBN(dbn,XtrainComp[:, :, 0:10000]).clone()
    _YtrainComp = YtrainComp.clone()
    _XtrainComp2 = forwardDBN(dbn, XtrainComp[:, :, 10000:20000]).clone()
    _XtrainComp = torch.cat((_XtrainComp1, _XtrainComp2), 2)
    del _XtrainComp1, _XtrainComp2

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

    # Run the model for the current layer size
    model_fit, weber, prob_choice_right, X = beta_extraction(
        choice, idxs_test, N_list_test, TSA_list_test, FA_list_test
    )
    # Collect results for the current layer size
    results.append({
        'Layer Size': f"{layer_size[0]} {layer_size[1]}",
        'Intercept': model_fit,
        'Beta Number': weber[0],
        'Beta Size': weber[1],
        'Beta Spacing': weber[2],
        'Weber Fraction': prob_choice_right,
        "Accuracy": acc
    })
    print(results)

#Convert results to a DataFrame
results_df = pd.DataFrame(results)

# Save the DataFrame to an Excel file

results_df.to_excel(output_file, index=False)

print(f"Results saved to {output_file}")





"""
