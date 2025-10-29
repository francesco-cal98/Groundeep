import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression, LinearRegression, Ridge, Lasso, ElasticNet, PoissonRegressor, RidgeClassifier, SGDClassifier, SGDRegressor
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from scipy.stats import norm,t, zscore

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class NumerosityDataset(torch.utils.data.Dataset):
    def __init__(self, X, Y):
        self.X = X
        self.Y = Y

    def __len__(self):
        return self.X.shape[0]  

    def __getitem__(self, idx):
        return self.X[idx], self.Y[idx] 
class Softmax(nn.Module):
    def __init__(self, n_inputs, n_outputs):
        super(Softmax, self).__init__()
        self.linear = nn.Linear(n_inputs, n_outputs)
        # self.dropout = nn.Dropout(p=0.1)

    def forward(self, x):
        # x = self.dropout(x)
        return self.linear(x)

# Regeressions for numerosity naming 
def Logistic_regression_multiclass(Xtrain, Xtest, Ytrain, Ytest, labels='int', scale = False, last_layer_size = 1000, MAX_NUM = 15): 
    # Xtrain = Xtrain.view(-1, Xtrain.shape[2]).cpu().detach().numpy()  
    # Xtest = Xtest.view(-1, Xtest.shape[2]).cpu().detach().numpy() 
    # Ytrain = Ytrain.reshape(-1, Ytrain.shape[-1]).cpu().detach().numpy()  
    # Ytest = Ytest.reshape(-1, Ytest.shape[-1]).cpu().detach().numpy()  

    # Ytrain = Ytrain[:, 1].reshape(-1)  
    # Ytest = Ytest[:, 1].reshape(-1)  

    if Xtrain.ndim > 2:
        Xtrain = Xtrain.view(-1, Xtrain.shape[-1]).detach().cpu().numpy()
        Xtest  = Xtest.view(-1,  Xtest.shape[-1]).detach().cpu().numpy()
    Ytrain = Ytrain.detach().cpu().numpy().ravel()
    Ytest  = Ytest.detach().cpu().numpy().ravel()

    model = LogisticRegression(max_iter=100, multi_class='multinomial', solver='saga', penalty='l2')

    _Ytrain = Ytrain
    _Ytest = Ytest

    if scale:
        scaler = StandardScaler()
        _Ytrain = scaler.fit_transform(_Ytrain.reshape(-1, 1)).ravel()
        _Ytest = scaler.transform(_Ytest.reshape(-1, 1)).ravel()

    model.fit(Xtrain, _Ytrain)

    predicted_train = model.predict(Xtrain)
    predicted_test = model.predict(Xtest)
    print(predicted_test[:5])

    if scale:
        predicted_train_classes = (predicted_train * scaler.scale_ + scaler.mean_ + 0.5).astype(int)
        predicted_test_classes = (predicted_test * scaler.scale_ + scaler.mean_ + 0.5).astype(int)
    else:
        predicted_train_classes = predicted_train
        predicted_test_classes = predicted_test

    accuracy_train = accuracy_score(Ytrain, predicted_train_classes)
    accuracy_test = accuracy_score(Ytest, predicted_test_classes)

    prob_train = model.predict_proba(Xtrain)
    prob_test = model.predict_proba(Xtest)

    print("Train probabilities (first few examples):")
    print(np.round(prob_train, 5)[:2]) 
    print("Test probabilities (first few examples):")
    print(np.round(prob_test, 5)[:2]) 

    probTR = np.argmax(prob_train, axis=1)
    probTE = np.argmax(prob_test, axis=1)

    return accuracy_train, predicted_train_classes, predicted_train, probTR, accuracy_test, predicted_test_classes, predicted_test, probTE
def Ridge_regression(Xtrain, Xtest, Ytrain, Ytest, labels='int', scale = False, last_layer_size = 1000, MAX_NUM = 15): 
    if Xtrain.ndim > 2:
        Xtrain = Xtrain.view(-1, Xtrain.shape[-1]).detach().cpu().numpy()
        Xtest = Xtest.view(-1, Xtest.shape[-1]).detach().cpu().numpy()
    
    Ytrain = Ytrain.detach().cpu().numpy().ravel()
    Ytest = Ytest.detach().cpu().numpy().ravel()

    model = Ridge(alpha=53)

    if labels == 'log':
        _Ytrain = np.log(Ytrain)
        _Ytest = np.log(Ytest)
    elif labels == 'int':
        _Ytrain = Ytrain
        _Ytest = Ytest

    zero_shift_tr = 0
    zero_shift_te = 0

    if scale:
        scaler = StandardScaler()
        _Ytrain = scaler.fit_transform(_Ytrain.reshape(-1, 1)).ravel()
        _Ytest = scaler.transform(_Ytest.reshape(-1, 1)).ravel()

    model.fit(Xtrain, _Ytrain)
 
    predicted_train = model.predict(Xtrain)
    predicted_test = model.predict(Xtest)

    # Inverse transformation if needed
    if labels == 'log':
        if scale:
            predicted_train_classes = (np.exp(predicted_train * scaler.scale_ + scaler.mean_ + zero_shift_tr) + 0.5).astype(int)
            predicted_test_classes = (np.exp(predicted_test * scaler.scale_ + scaler.mean_ + zero_shift_te) + 0.5).astype(int)
        else:
            predicted_train_classes = (np.exp(predicted_train + zero_shift_tr) + 0.5).astype(int)
            predicted_test_classes = (np.exp(predicted_test + zero_shift_te) + 0.5).astype(int)
    elif labels == 'int':
        if scale:
            predicted_train_classes = (predicted_train * scaler.scale_ + scaler.mean_ + zero_shift_tr + 0.5).astype(int)
            predicted_test_classes = (predicted_test * scaler.scale_ + scaler.mean_ + zero_shift_te + 0.5).astype(int)
        else:
            predicted_train_classes = (predicted_train + zero_shift_tr + 0.5).astype(int)
            predicted_test_classes = (predicted_test + zero_shift_te + 0.5).astype(int)

    accuracy_train = accuracy_score(Ytrain, predicted_train_classes)
    accuracy_test = accuracy_score(Ytest, predicted_test_classes)

    probTR = False
    probTE = False

    return accuracy_train, predicted_train_classes, predicted_train, probTR, accuracy_test, predicted_test_classes, predicted_test, probTE

def SGD_regression(Xtrain, Xtest, Ytrain, Ytest, labels='int', scale = False, last_layer_size = 1000, MAX_NUM = 15): 
    if Xtrain.ndim > 2:
        Xtrain = Xtrain.view(-1, Xtrain.shape[-1]).detach().cpu().numpy()
        Xtest = Xtest.view(-1, Xtest.shape[-1]).detach().cpu().numpy()
    
    Ytrain = Ytrain.detach().cpu().numpy().ravel()
    Ytest = Ytest.detach().cpu().numpy().ravel()

    model = SGDRegressor(loss='squared_error', penalty='l2', random_state= 42)

    if labels == 'log':
        _Ytrain = np.log(Ytrain)
        _Ytest = np.log(Ytest)
    elif labels == 'int':
        _Ytrain = Ytrain
        _Ytest = Ytest

    zero_shift_tr = 0
    zero_shift_te = 0

    if scale:
        scaler = StandardScaler()
        _Ytrain = scaler.fit_transform(_Ytrain.reshape(-1, 1)).ravel()
        _Ytest = scaler.transform(_Ytest.reshape(-1, 1)).ravel()

    model.fit(Xtrain, _Ytrain)
 
    predicted_train = model.predict(Xtrain)
    predicted_test = model.predict(Xtest)

    # Inverse transformation if needed
    if labels == 'log':
        if scale:
            predicted_train_classes = (np.exp(predicted_train * scaler.scale_ + scaler.mean_ + zero_shift_tr) + 0.5).astype(int)
            predicted_test_classes = (np.exp(predicted_test * scaler.scale_ + scaler.mean_ + zero_shift_te) + 0.5).astype(int)
        else:
            predicted_train_classes = (np.exp(predicted_train + zero_shift_tr) + 0.5).astype(int)
            predicted_test_classes = (np.exp(predicted_test + zero_shift_te) + 0.5).astype(int)
    elif labels == 'int':
        if scale:
            predicted_train_classes = (predicted_train * scaler.scale_ + scaler.mean_ + zero_shift_tr + 0.5).astype(int)
            predicted_test_classes = (predicted_test * scaler.scale_ + scaler.mean_ + zero_shift_te + 0.5).astype(int)
        else:
            predicted_train_classes = (predicted_train + zero_shift_tr + 0.5).astype(int)
            predicted_test_classes = (predicted_test + zero_shift_te + 0.5).astype(int)

    accuracy_train = accuracy_score(Ytrain, predicted_train_classes)
    accuracy_test = accuracy_score(Ytest, predicted_test_classes)

    probTR = False
    probTE = False

    return accuracy_train, predicted_train_classes, predicted_train, probTR, accuracy_test, predicted_test_classes, predicted_test, probTE

def Linear_regression(Xtrain, Xtest, Ytrain, Ytest, labels = 'int', scale = False, last_layer_size = 1000, MAX_NUM = 15): 
    if Xtrain.ndim > 2:
        Xtrain = Xtrain.view(-1, Xtrain.shape[-1]).detach().cpu().numpy()
        Xtest = Xtest.view(-1, Xtest.shape[-1]).detach().cpu().numpy()
    
    Ytrain = Ytrain.detach().cpu().numpy().ravel()
    Ytest = Ytest.detach().cpu().numpy().ravel()

    model = LinearRegression()

    if labels == 'log':
        _Ytrain = np.log(Ytrain)
        _Ytest = np.log(Ytest)
    elif labels == 'int':
        _Ytrain = Ytrain
        _Ytest = Ytest

    zero_shift_tr = 0
    zero_shift_te = 0

    if scale:
        scaler = StandardScaler()
        _Ytrain = scaler.fit_transform(_Ytrain.reshape(-1, 1)).ravel()
        _Ytest = scaler.transform(_Ytest.reshape(-1, 1)).ravel()

    model.fit(Xtrain, _Ytrain)
 
    predicted_train = model.predict(Xtrain)
    predicted_test = model.predict(Xtest)

    # Inverse transformation if needed
    if labels == 'log':
        if scale:
            predicted_train_classes = (np.exp(predicted_train * scaler.scale_ + scaler.mean_ + zero_shift_tr) + 0.5).astype(int)
            predicted_test_classes = (np.exp(predicted_test * scaler.scale_ + scaler.mean_ + zero_shift_te) + 0.5).astype(int)
        else:
            predicted_train_classes = (np.exp(predicted_train + zero_shift_tr) + 0.5).astype(int)
            predicted_test_classes = (np.exp(predicted_test + zero_shift_te) + 0.5).astype(int)
    elif labels == 'int':
        if scale:
            predicted_train_classes = (predicted_train * scaler.scale_ + scaler.mean_ + zero_shift_tr + 0.5).astype(int)
            predicted_test_classes = (predicted_test * scaler.scale_ + scaler.mean_ + zero_shift_te + 0.5).astype(int)
        else:
            predicted_train_classes = (predicted_train + zero_shift_tr + 0.5).astype(int)
            predicted_test_classes = (predicted_test + zero_shift_te + 0.5).astype(int)

    accuracy_train = accuracy_score(Ytrain, predicted_train_classes)
    accuracy_test = accuracy_score(Ytest, predicted_test_classes)

    probTR = False
    probTE = False

    return accuracy_train, predicted_train_classes, predicted_train, probTR, accuracy_test, predicted_test_classes, predicted_test, probTE
def Poisson_regression(Xtrain, Xtest, Ytrain, Ytest, labels='int', scale=False, last_layer_size = 1000, MAX_NUM = 15): 
    if Xtrain.ndim > 2:
        Xtrain = Xtrain.view(-1, Xtrain.shape[-1]).detach().cpu().numpy()
        Xtest = Xtest.view(-1, Xtest.shape[-1]).detach().cpu().numpy()
    
    Ytrain = Ytrain.detach().cpu().numpy().ravel()
    Ytest = Ytest.detach().cpu().numpy().ravel()

    # Train the model
    model = PoissonRegressor()
    model.fit(Xtrain, Ytrain)

    predicted_train = model.predict(Xtrain)
    predicted_test = model.predict(Xtest)

    predicted_train_classes = (predicted_train).astype(int)
    predicted_test_classes = (predicted_test).astype(int)

    accuracy_train = accuracy_score(Ytrain, predicted_train_classes)
    accuracy_test = accuracy_score(Ytest, predicted_test_classes)

    probTR = False
    probTE = False

    return accuracy_train, predicted_train_classes, predicted_train, probTR, accuracy_test, predicted_test_classes, predicted_test, probTE
def Lasso_regression(Xtrain, Xtest, Ytrain, Ytest, labels = 'int', scale = 0, last_layer_size = 1000, MAX_NUM = 15): 
    if Xtrain.ndim > 2:
        Xtrain = Xtrain.view(-1, Xtrain.shape[-1]).detach().cpu().numpy()
        Xtest = Xtest.view(-1, Xtest.shape[-1]).detach().cpu().numpy()
    
    Ytrain = Ytrain.detach().cpu().numpy().ravel()
    Ytest = Ytest.detach().cpu().numpy().ravel()

    # Train the model
    model = Lasso(alpha=0.005)
    
    if labels == 'log':
        _Ytrain = np.log(Ytrain)
        _Ytest = np.log(Ytest)
    elif labels == 'int':
        _Ytrain = Ytrain
        _Ytest = Ytest

    zero_shift_tr = 0
    zero_shift_te = 0

    if scale:
        scaler = StandardScaler()
        _Ytrain = scaler.fit_transform(_Ytrain.reshape(-1, 1)).ravel()
        _Ytest = scaler.transform(_Ytest.reshape(-1, 1)).ravel()

    model.fit(Xtrain, _Ytrain)
 
    predicted_train = model.predict(Xtrain)
    predicted_test = model.predict(Xtest)

    # Inverse transformation if needed
    if labels == 'log':
        if scale:
            predicted_train_classes = (np.exp(predicted_train * scaler.scale_ + scaler.mean_ + zero_shift_tr) + 0.5).astype(int)
            predicted_test_classes = (np.exp(predicted_test * scaler.scale_ + scaler.mean_ + zero_shift_te) + 0.5).astype(int)
        else:
            predicted_train_classes = (np.exp(predicted_train + zero_shift_tr) + 0.5).astype(int)
            predicted_test_classes = (np.exp(predicted_test + zero_shift_te) + 0.5).astype(int)
    elif labels == 'int':
        if scale:
            predicted_train_classes = (predicted_train * scaler.scale_ + scaler.mean_ + zero_shift_tr + 0.5).astype(int)
            predicted_test_classes = (predicted_test * scaler.scale_ + scaler.mean_ + zero_shift_te + 0.5).astype(int)
        else:
            predicted_train_classes = (predicted_train + zero_shift_tr + 0.5).astype(int)
            predicted_test_classes = (predicted_test + zero_shift_te + 0.5).astype(int)

    accuracy_train = accuracy_score(Ytrain, predicted_train_classes)
    accuracy_test = accuracy_score(Ytest, predicted_test_classes)

    probTR = False
    probTE = False

    return accuracy_train, predicted_train_classes, predicted_train, probTR, accuracy_test, predicted_test_classes, predicted_test, probTE
def Elastic_net_regression(Xtrain, Xtest, Ytrain, Ytest, labels = 'int', scale = 0, last_layer_size = 1000, MAX_NUM = 15): 
    if Xtrain.ndim > 2:
        Xtrain = Xtrain.view(-1, Xtrain.shape[-1]).detach().cpu().numpy()
        Xtest = Xtest.view(-1, Xtest.shape[-1]).detach().cpu().numpy()
    
    Ytrain = Ytrain.detach().cpu().numpy().ravel()
    Ytest = Ytest.detach().cpu().numpy().ravel()

    # Train the model
    model = ElasticNet(alpha=0.005, l1_ratio=0.05)
    
    if labels == 'log':
        _Ytrain = np.log(Ytrain)
        _Ytest = np.log(Ytest)
    elif labels == 'int':
        _Ytrain = Ytrain
        _Ytest = Ytest

    zero_shift_tr = 0
    zero_shift_te = 0

    if scale:
        scaler = StandardScaler()
        _Ytrain = scaler.fit_transform(_Ytrain.reshape(-1, 1)).ravel()
        _Ytest = scaler.transform(_Ytest.reshape(-1, 1)).ravel()

    model.fit(Xtrain, _Ytrain)
 
    predicted_train = model.predict(Xtrain)
    predicted_test = model.predict(Xtest)

    # Inverse transformation if needed
    if labels == 'log':
        if scale:
            predicted_train_classes = (np.exp(predicted_train * scaler.scale_ + scaler.mean_ + zero_shift_tr) + 0.5).astype(int)
            predicted_test_classes = (np.exp(predicted_test * scaler.scale_ + scaler.mean_ + zero_shift_te) + 0.5).astype(int)
        else:
            predicted_train_classes = (np.exp(predicted_train + zero_shift_tr) + 0.5).astype(int)
            predicted_test_classes = (np.exp(predicted_test + zero_shift_te) + 0.5).astype(int)
    elif labels == 'int':
        if scale:
            predicted_train_classes = (predicted_train * scaler.scale_ + scaler.mean_ + zero_shift_tr + 0.5).astype(int)
            predicted_test_classes = (predicted_test * scaler.scale_ + scaler.mean_ + zero_shift_te + 0.5).astype(int)
        else:
            predicted_train_classes = (predicted_train + zero_shift_tr + 0.5).astype(int)
            predicted_test_classes = (predicted_test + zero_shift_te + 0.5).astype(int)

    accuracy_train = accuracy_score(Ytrain, predicted_train_classes)
    accuracy_test = accuracy_score(Ytest, predicted_test_classes)

    probTR = False
    probTE = False

    return accuracy_train, predicted_train_classes, predicted_train, probTR, accuracy_test, predicted_test_classes, predicted_test, probTE

# Classifiers for the pairwise comparison task
def Pseudoinverse_pairs(Xtrain, Xtest, Ytrain, Ytest):
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
def Ridge_class_pairs(Xtrain, Xtest, Ytrain, Ytest):
    Xtrain = Xtrain.view(-1, Xtrain.shape[2]).detach().cpu().numpy()
    Xtest = Xtest.view(-1, Xtest.shape[2]).detach().cpu().numpy()
    
    Ytrain = Ytrain.view(-1, Ytrain.shape[-1]).detach().cpu().numpy()
    Ytest = Ytest.view(-1, Ytest.shape[-1]).detach().cpu().numpy()
    print(Xtrain.shape, Ytrain.shape, Xtest.shape, Ytest.shape )

    Xtrain = np.hstack([Xtrain, np.ones((Xtrain.shape[0], 1))])
    Xtest = np.hstack([Xtest, np.ones((Xtest.shape[0], 1))])
    
    model = RidgeClassifier(solver='svd')
    
    model.fit(Xtrain, Ytrain)

    predicted_train = model.predict(Xtrain)
    accuracy_train = accuracy_score(Ytrain, predicted_train)

    print('\naccuracy train: ' , accuracy_train)

    predicted_test = model.predict(Xtest)
    accuracy_test = accuracy_score(Ytest, predicted_test)

    print(f'accuracy test: ', accuracy_test) 

    predicted_train = np.argmax(predicted_train, axis=1)
    predicted_test = np.argmax(predicted_test, axis=1)
    Ytrain_labels = np.argmax(Ytrain, axis=1)
    Ytest_labels = np.argmax(Ytest, axis=1)

    accuracy_train = accuracy_score(Ytrain_labels, predicted_train)
    accuracy_test = accuracy_score(Ytest_labels, predicted_test)


    return accuracy_train, predicted_train, accuracy_test, predicted_test
def SGD_class_pairs(Xtrain, Xtest, Ytrain, Ytest):
    # Flatten the input features
    Xtrain = Xtrain.view(-1, Xtrain.shape[2]).cpu().detach().numpy()  # Detach and convert to NumPy
    Xtest = Xtest.view(-1, Xtest.shape[2]).cpu().detach().numpy()  # Detach and convert to NumPy
    
    # Ensure Ytrain and Ytest are 2D (if not already)
    Ytrain = Ytrain.reshape(-1, Ytrain.shape[-1]).cpu().detach().numpy()  # Detach and convert to NumPy
    Ytest = Ytest.reshape(-1, Ytest.shape[-1]).cpu().detach().numpy()  # Detach and convert to NumPy

    # If Ytrain and Ytest have more than one column, extract the second column (index 1)
    Ytrain = Ytrain[:, 1].reshape(-1)  # Flatten to 1D after extracting second column
    Ytest = Ytest[:, 1].reshape(-1)  # Flatten to 1D after extracting second column


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
def Logistic_class_pairs(Xtrain, Xtest, Ytrain, Ytest):
   # Flatten the input features
    Xtrain = Xtrain.view(-1, Xtrain.shape[2]).cpu().detach().numpy()  # Detach and convert to NumPy
    Xtest = Xtest.view(-1, Xtest.shape[2]).cpu().detach().numpy()  # Detach and convert to NumPy
    
    # Ensure Ytrain and Ytest are 2D (if not already)
    Ytrain = Ytrain.reshape(-1, Ytrain.shape[-1]).cpu().detach().numpy()  # Detach and convert to NumPy
    Ytest = Ytest.reshape(-1, Ytest.shape[-1]).cpu().detach().numpy()  # Detach and convert to NumPy

    # If Ytrain and Ytest have more than one column, extract the second column (index 1)
    Ytrain = Ytrain[:, 1].reshape(-1)  # Flatten to 1D after extracting second column
    Ytest = Ytest[:, 1].reshape(-1)  # Flatten to 1D after extracting second column

    
    in_features = Xtrain.shape[1]
    model = LogisticRegression(penalty='l1', solver='saga', max_iter=1000)                 
    
    model.fit(Xtrain, Ytrain)

    predicted_train = model.predict(Xtrain)
    accuracy_train = accuracy_score(Ytrain, predicted_train)

    print('\naccuracy train: ' , accuracy_train)

    predicted_test = model.predict(Xtest)
    accuracy_test = accuracy_score(Ytest, predicted_test)

    print(f'accuracy test: ', accuracy_test) 

    return accuracy_train, predicted_train, accuracy_test, predicted_test

# Classifiers for the fixed reference comparison task
def Pseudoinverse_fixed(Xtrain, Xtest, Ytrain, Ytest):
      # Reshaping the data
    if Xtrain.ndim > 2:
        Xtrain = Xtrain.view(-1, Xtrain.shape[-1]).detach().cpu().numpy()
        Xtest  = Xtest.view(-1,  Xtest.shape[-1]).detach().cpu().numpy()
    Ytrain = Ytrain.detach().cpu().numpy().ravel()
    Ytest  = Ytest.detach().cpu().numpy().ravel()

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
    predicted_train = (pred_train > 0.5).astype(int)
    predicted_test  = (pred_test  > 0.5).astype(int)
    print(predicted_train[:10])

    accuracy_train = accuracy_score(Ytrain, predicted_train)
    accuracy_test  = accuracy_score(Ytest,  predicted_test)


    print('Pseudo-Inverse Perceptron:')
    print('Train Accuracy:', accuracy_train)
    print('Test Accuracy:', accuracy_test)

    return accuracy_train, predicted_train, accuracy_test, predicted_test
def Ridge_class_fixed(Xtrain, Xtest, Ytrain, Ytest):
      # Reshaping the data
    if Xtrain.ndim > 2:
        Xtrain = Xtrain.view(-1, Xtrain.shape[-1]).detach().cpu().numpy()
        Xtest  = Xtest.view(-1,  Xtest.shape[-1]).detach().cpu().numpy()
    Ytrain = Ytrain.detach().cpu().numpy().ravel()
    Ytest  = Ytest.detach().cpu().numpy().ravel()

    Xtrain = np.hstack([Xtrain, np.ones((Xtrain.shape[0], 1))])
    Xtest = np.hstack([Xtest, np.ones((Xtest.shape[0], 1))])
    
    model = RidgeClassifier(solver='svd')
    
    model.fit(Xtrain, Ytrain)

    predicted_train = model.predict(Xtrain)
    accuracy_train = accuracy_score(Ytrain, predicted_train)

    print(f'\naccuracy train: {accuracy_train}')

    predicted_test = model.predict(Xtest)
    accuracy_test = accuracy_score(Ytest, predicted_test)

    print(f'accuracy test: {accuracy_test}') 

    return accuracy_train, predicted_train, accuracy_test, predicted_test
def Logistic_class_fixed(Xtrain, Xtest, Ytrain, Ytest):
      # Reshaping the data
    if Xtrain.ndim > 2:
        Xtrain = Xtrain.view(-1, Xtrain.shape[-1]).detach().cpu().numpy()
        Xtest  = Xtest.view(-1,  Xtest.shape[-1]).detach().cpu().numpy()
    Ytrain = Ytrain.detach().cpu().numpy().ravel()
    Ytest  = Ytest.detach().cpu().numpy().ravel()

    Xtrain = np.hstack([Xtrain, np.ones((Xtrain.shape[0], 1))])
    Xtest = np.hstack([Xtest, np.ones((Xtest.shape[0], 1))])
    
    model = LogisticRegression(penalty='l1', solver='saga', max_iter=1000)  
    
    model.fit(Xtrain, Ytrain)

    predicted_train = model.predict(Xtrain)
    accuracy_train = accuracy_score(Ytrain, predicted_train)

    print(f'\naccuracy train: {accuracy_train}')

    predicted_test = model.predict(Xtest)
    accuracy_test = accuracy_score(Ytest, predicted_test)

    print(f'accuracy test: {accuracy_test}') 

    return accuracy_train, predicted_train, accuracy_test, predicted_test
def SGD_class_fixed(Xtrain, Xtest, Ytrain, Ytest):
      # Reshaping the data
    if Xtrain.ndim > 2:
        Xtrain = Xtrain.view(-1, Xtrain.shape[-1]).detach().cpu().numpy()
        Xtest  = Xtest.view(-1,  Xtest.shape[-1]).detach().cpu().numpy()
    Ytrain = Ytrain.detach().cpu().numpy().ravel()
    Ytest  = Ytest.detach().cpu().numpy().ravel()

    Xtrain = np.hstack([Xtrain, np.ones((Xtrain.shape[0], 1))])
    Xtest = np.hstack([Xtest, np.ones((Xtest.shape[0], 1))])
    
    # model = SGDClassifier(loss='perceptron',penalty='l1',max_iter=1000, random_state= 42)
    model = SGDClassifier(penalty='l2',max_iter=1000,random_state= 42)
    
    model.fit(Xtrain, Ytrain)

    predicted_train = model.predict(Xtrain)
    accuracy_train = accuracy_score(Ytrain, predicted_train)

    print(f'\naccuracy train: {accuracy_train}')

    predicted_test = model.predict(Xtest)
    accuracy_test = accuracy_score(Ytest, predicted_test)

    print(f'accuracy test: {accuracy_test}') 

    return accuracy_train, predicted_train, accuracy_test, predicted_test

# GLM helper functions
def irls_fit_b(choice, X, guessRate=0.01, max_iter=5000, tol=1e-12):
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

    # Compute standard errors
    cov_matrix = np.linalg.pinv(X_design.T @ (W[:, np.newaxis] * X_design))
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

    return beta[0], beta[1:], weber, t_stats[1:], p_values[1:]
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

    # Debugging: Inputs for left and right parameters
    # print('--- Input Debugging ---')
    # print(f'Number of trials: {len(choice)}')
    # print(f'Choice vector: {np.sum(choice == 0)} left, {np.sum(choice == 1)} right')

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
def beta_extraction_ref_z(choice, idxs, N_list, TSA_list, FA_list, guessRate=0.01, ref_num=None):
    N_list = np.squeeze(N_list)
    TSA_list = np.squeeze(TSA_list)
    FA_list = np.squeeze(FA_list)

    idxs=idxs.ravel()

    num = []
    isa = []
    fa = []

    for idx in idxs:
        idx = int(idx)
        num.append(N_list[idx])
        isa.append(TSA_list[idx]/ N_list[idx])
        fa.append(FA_list[idx])
   

    model_fit, betas, wf, X, prob_choice_higher, model, numZ, sizeZ, spaceZ, t_stats, p_vals,standard_errors = num_size_spacing_model_ref_z(
        choice, np.array(num), np.array(isa), np.array(fa), guessRate
    )

    num_ratios = np.array(num)/ref_num

    return model_fit, betas, wf, X, num_ratios, prob_choice_higher, model, numZ, sizeZ, spaceZ, num, t_stats, p_vals,standard_errors
def num_size_spacing_model_ref_z(choice, num, isa, fa, guessRate=0.01):
    tsa = isa* num
    size = isa * tsa
    spar = fa / num
    space = spar* fa

    numZ,sizeZ, spaceZ = zscores_unique(num, size, space)

    X = np.column_stack((numZ, sizeZ, spaceZ))
    choice = np.array(choice)

    # Fit using IRLS
    intercept, betas, weber, prob_choice_higher, model, t_stats, p_vals, standard_errors = irls_fit(choice, X, guessRate)

    return intercept, betas, weber, X, prob_choice_higher, model, numZ, sizeZ, spaceZ, t_stats, p_vals, standard_errors
def zscores_unique(num, size, space):
    num_unique = (np.unique(num))
    size_unique = (np.unique(size))
    space_unique = (np.unique(space))

    numZ_map = zscore(np.log2(num_unique))
    sizeZ_map = zscore(np.log2(size_unique))
    spaceZ_map = zscore(np.log2(space_unique))

    num_z_unique = {value: z for value, z in zip(num_unique, numZ_map)}
    size_z_unique = {value: z for value, z in zip(size_unique, sizeZ_map)}
    space_z_unique = {value: z for value, z in zip(space_unique, spaceZ_map)}

    num_z = np.array([num_z_unique[val] for val in num])
    size_z = np.array([size_z_unique[val] for val in size])
    space_z = np.array([space_z_unique[val] for val in space])

    return num_z, size_z, space_z
def compute_prob_choice(X, intercept, betas, guessRate=0.01):
    linear_combination = intercept + np.dot(X, betas)
    prob_choice = guessRate + (1 - guessRate) * norm.cdf(linear_combination)
    return prob_choice

def Softmax_multiclass(_XtrainComp, _XtestComp, _YtrainComp, _YtestComp, labels = 'int', scale = 0, last_layer_size = 1000, MAX_NUM = 15):
    with torch.no_grad(): # Normalizing
        _XtrainComp = (_XtrainComp - _XtrainComp.mean(dim=0)) / (_XtrainComp.std(dim=0) + 1e-8)
        _XtestComp = (_XtestComp - _XtestComp.mean(dim=0)) / (_XtestComp.std(dim=0) + 1e-8)

    Xtrain_flat = _XtrainComp.view(-1, _XtrainComp.shape[2])
    Xtest_flat = _XtestComp.view(-1, _XtestComp.shape[2])
    Ytrain_flat = _YtrainComp.to(dtype=torch.int64, device=DEVICE).view(-1) - 1
    Ytest_flat = _YtestComp.to(dtype=torch.int64, device=DEVICE).view(-1) - 1

    train_dataset = NumerosityDataset(Xtrain_flat.to('cpu'), Ytrain_flat.to('cpu'))
    test_dataset = NumerosityDataset(Xtest_flat.to('cpu'), Ytest_flat.to('cpu'))

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=100, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=100, shuffle=True)

    for batch_idx, (X_batch, Y_batch) in enumerate(train_loader):
        print(f"Batch {batch_idx}: X shape {X_batch.shape}, Y shape {Y_batch.shape}")
        break

    model_softmax = Softmax(last_layer_size, MAX_NUM).to(DEVICE)
    print(model_softmax)
    optimizer = optim.SGD(model_softmax.parameters(), lr=0.01, weight_decay=0.001)
    criterion = nn.CrossEntropyLoss()

    batch_size = 100

    train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)

    epochs = 200
    Loss = []
    acc = []

    # Early stopping parameters
    patience = 10  # Number of epochs to wait for improvement
    best_accuracy = 0.0
    patience_counter = 0

    for epoch in range(epochs):
        model_softmax.train()
        for data, labels in train_loader:
            optimizer.zero_grad()  # Clears previous gradients

            data = data.view(-1, last_layer_size).to(DEVICE) 
            labels = labels.view(-1).long().squeeze().to(DEVICE)
            data = data.detach()

            outputs = model_softmax(data)
            loss = criterion(outputs, labels)

            Loss.append(loss.detach().cpu().item())
            loss.backward(retain_graph=False) 
            optimizer.step()

        model_softmax.eval()
        correct = 0
        total = 0
        all_pred_probs = []
        all_pred_classes = []
        all_labels = []
        with torch.no_grad():
            for data, labels in val_loader:
                data = data.view(-1, last_layer_size).to(DEVICE)
                labels = labels.view(-1).long().to(DEVICE)

                outputs = model_softmax(data)
                all_pred_probs.append(outputs)

                preds = outputs.argmax(dim=1)
                correct += (preds== labels).sum().item()

                pred_classes = preds + 1
                class_labels = labels + 1

                all_pred_classes.append(pred_classes)
                all_labels.append(class_labels)

                total += labels.size(0)

        accuracy = 100 * correct / total
        acc.append(accuracy)

        if accuracy > best_accuracy:
            best_accuracy = accuracy
            patience_counter = 0  # Reset patience counter
        else:
            patience_counter += 1

        # Stop training if patience is exceeded
        if patience_counter >= patience:
            print(f"Early stopping at epoch {epoch}. Best accuracy: {best_accuracy:.2f}%")
            break

        if epoch % 10 == 0:
            print(f'Epoch: {epoch}, Loss: {loss.item():.4f}, Accuracy: {accuracy:.2f}%')

    # # Plot Loss
    # plt.plot(Loss)
    # plt.xlabel("Epochs")
    # plt.ylabel("Loss")
    # plt.show()

    # # Plot Accuracy
    # plt.plot(acc)
    # plt.xlabel("Epochs")
    # plt.ylabel("Accuracy (%)")
    # plt.show()

    all_pred_prob_tensor = torch.cat(all_pred_probs, dim=0)
    probabilities = torch.softmax(all_pred_prob_tensor, dim=1)

    all_pred_classes_tensor = torch.cat(all_pred_classes, dim=0)
    all_pred_classes = all_pred_classes_tensor.cpu().detach().numpy()

    all_labels_tensor = torch.cat(all_labels, dim=0)
    all_labels = all_labels_tensor.cpu().detach().numpy()

    accuracy_train = 0
    predicted_train_classes = all_pred_classes
    predicted_train = 0
    probTR = probabilities
    accuracy_test = best_accuracy/100
    predicted_test_classes = all_pred_classes
    predicted_test = 0
    probTE = probabilities


    return accuracy_train, predicted_train_classes, predicted_train, probTR, accuracy_test, predicted_test_classes, predicted_test, probTE