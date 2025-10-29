import math
import pickle
from tqdm import tqdm
import torch
import torch.nn as nn
import numpy as np
 
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def forwardrbm(self, v):
    v = v.float()
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

# Utility function for sigmoid activation
def sigmoid(x):
    return 1 / (1 + torch.exp(-x))


# RBM Class
class RBM(nn.Module):
    def __init__(self, num_visible, num_hidden, learning_rate, weight_decay, momentum, dynamic_lr=False, final_momentum=0.97):
        super(RBM, self).__init__()
        self.num_visible = num_visible
        self.num_hidden = num_hidden
        self.lr = learning_rate
        self.weight_decay = weight_decay
        self.momentum = momentum
        self.dynamic_lr = dynamic_lr
        self.final_momentum = final_momentum

        # Weights and biases
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.W = nn.Parameter(
            torch.randn(num_visible, num_hidden, device=device) / math.sqrt(num_visible)
        )
        self.hid_bias = nn.Parameter(torch.zeros(num_hidden, device=device))
        self.vis_bias = nn.Parameter(torch.zeros(num_visible, device=device))

        # Gradients momentum
        self.W_momentum = torch.zeros_like(self.W)
        self.hid_bias_momentum = torch.zeros_like(self.hid_bias)
        self.vis_bias_momentum = torch.zeros_like(self.vis_bias)

    def forward(self, v):
        return sigmoid(torch.matmul(v, self.W) + self.hid_bias)

    def backward(self, h):
        return sigmoid(torch.matmul(h, self.W.T) + self.vis_bias) 

    def train_epoch(self, data, epoch, max_epochs, CD=1):
        total_loss = 0.0
        lr = self.lr / (1 + 0.01 * epoch) if self.dynamic_lr else self.lr
        batch_size = data.size(1)
        momentum = self.momentum if epoch <= 5 else self.final_momentum
        for batch_idx in range(data.size(0)):
            batch_data = data[batch_idx]

            with torch.no_grad():
                # Positive phase
                pos_hid_probs = self.forward(batch_data)
                pos_hid_states = (pos_hid_probs > torch.rand_like(pos_hid_probs)).float()
                pos_assoc = torch.matmul(batch_data.T, pos_hid_probs)

                # Negative phase
                neg_data = batch_data
                for _ in range(CD):
                    neg_vis_probs = self.backward(pos_hid_states)
                    neg_data = (neg_vis_probs > torch.rand_like(neg_vis_probs)).float()
                    neg_hid_probs = self.forward(neg_data)
                    pos_hid_states = (neg_hid_probs > torch.rand_like(neg_hid_probs)).float()
                neg_assoc = torch.matmul(neg_data.T, neg_hid_probs)

                # Weight and bias updates
                self.W_momentum.mul_(momentum).add_(lr * ((pos_assoc - neg_assoc) / batch_size - self.weight_decay * self.W))
                self.hid_bias_momentum.mul_(momentum).add_(lr * (pos_hid_probs.sum(0) - neg_hid_probs.sum(0)) / batch_size)
                self.vis_bias_momentum.mul_(momentum).add_(lr * (batch_data.sum(0) - neg_data.sum(0)) / batch_size)

                self.W.add_(self.W_momentum)
                self.hid_bias.add_(self.hid_bias_momentum)
                self.vis_bias.add_(self.vis_bias_momentum)

                batch_loss = torch.sum((batch_data - neg_vis_probs) ** 2) / batch_size
                total_loss += batch_loss.item()

                torch.cuda.empty_cache()

        return total_loss / data.size(0)


# DBN Class
class gDBN:
    def __init__(self, layer_sizes, params):
        self.layers = []
        self.params = params
        for i in range(len(layer_sizes) - 1):
            rbm = RBM(
                num_visible=layer_sizes[i],
                num_hidden=layer_sizes[i + 1],
                learning_rate=params['LEARNING_RATE'],
                weight_decay=params['WEIGHT_PENALTY'],
                momentum=params['INIT_MOMENTUM'],
                dynamic_lr=params['LEARNING_RATE_DYNAMIC'],
                final_momentum=params["FINAL_MOMENTUM"] 
            )
            self.layers.append(rbm)

    def train(self, data, epochs):
        for layer_idx, rbm in enumerate(self.layers):
            print(f'Training RBM Layer {layer_idx + 1}')
            layer_data = data
            for epoch in tqdm(range(epochs)):
                loss = rbm.train_epoch(layer_data, epoch, epochs, CD=1)
                if epoch % 10 == 0:
                    print(f'Epoch {epoch}, Loss: {loss:.4f}')
                torch.cuda.empty_cache()
            data = rbm.forward(data)

    def save(self, path):
        with open(path, 'wb') as f:
            pickle.dump(self, f)

# gDBN with pretraining
class pgDBN:
    def __init__(self, layer_sizes, params):
        self.layers = []
        self.params = params
        for i in range(len(layer_sizes) - 1):
            rbm = RBM(
                num_visible=layer_sizes[i],
                num_hidden=layer_sizes[i + 1],
                learning_rate=params['LEARNING_RATE'],
                weight_decay=params['WEIGHT_PENALTY'],
                momentum=params['INIT_MOMENTUM'],
                dynamic_lr=params['LEARNING_RATE_DYNAMIC'],
                final_momentum=params["FINAL_MOMENTUM"]
            )
            self.layers.append(rbm)

    def train(self, data, epochs, pretrain_data, pretrain_epochs):
        for layer_idx, rbm in enumerate(self.layers):
            print(f'Training RBM Layer {layer_idx + 1}')
            if layer_idx ==  0:
                pretrain_layer_data = pretrain_data
                for epoch in tqdm(range(pretrain_epochs)):
                    loss = rbm.train_epoch(pretrain_layer_data, epoch, epochs, CD=1)
                    if epoch % 10 == 0:
                        print(f'Pretraing Epoch {epoch}, Loss: {loss:.4f}')
                    torch.cuda.empty_cache()
                # pretrain_data = rbm.forward(pretrain_data)

            layer_data = data
            for epoch in tqdm(range(epochs)):
                loss = rbm.train_epoch(layer_data, epoch, epochs, CD=1)
                if epoch % 10 == 0:
                    print(f'Epoch {epoch}, Loss: {loss:.4f}')
                torch.cuda.empty_cache()
            data = rbm.forward(data)

    def save(self, path):
        with open(path, 'wb') as f:
            pickle.dump(self, f)


# iDBN with pretraining
class piDBN:
    def __init__(self, layer_sizes, params):
        self.layers = []
        self.params = params
        for i in range(len(layer_sizes) - 1):
            rbm = RBM(
                num_visible=layer_sizes[i],
                num_hidden=layer_sizes[i + 1],
                learning_rate=params['LEARNING_RATE'],
                weight_decay=params['WEIGHT_PENALTY'],
                momentum=params['INIT_MOMENTUM'],
                dynamic_lr=params['LEARNING_RATE_DYNAMIC'],
                final_momentum=params["FINAL_MOMENTUM"]
            )
            self.layers.append(rbm)

    def train(self, data, epochs, pretrain_data, pretrain_epochs, reinitialize_W = 0, threshold = 0.01): 
            for layer_idx, rbm in enumerate(self.layers):
                if layer_idx ==  0:
                    print(f'\nPretraining RBM Layer {layer_idx + 1}')
                    pretrain_layer_data = pretrain_data
                    for epoch in tqdm(range(pretrain_epochs)):
                        loss = rbm.train_epoch(pretrain_layer_data, epoch, epochs, CD=1)
                        if epoch % 10 == 0:
                            print(f'\nPretraing Epoch {epoch}, Loss: {loss:.4f}')
                        torch.cuda.empty_cache()
                    pretrain_data = rbm.forward(pretrain_data)
                    if reinitialize_W:
                        weights = rbm.W
                        mask = (weights.abs() > threshold)
                        # Keep weights above threshold, reinitialize those below
                        reint_weights = torch.randn_like(weights) * 0.01
                        updated_weights = weights * mask + reint_weights * (~mask) 
                        rbm.W = torch.nn.Parameter(updated_weights)
                    

            for epoch in tqdm(range(epochs)):
                temp_data = data.clone()  # Reset to original data for each epoch
                for i, rbm_layer in enumerate(self.layers):
                    print(f"Before Layer {i}: temp_data shape = {temp_data.shape}, W shape = {rbm_layer.W.shape}")
                    
                    # Train the current RBM layer
                    loss = rbm_layer.train_epoch(temp_data, epoch, epochs, CD=1)
                    
                    # Generate output for the next layer
                    temp_data = rbm_layer.forward(temp_data)
                    print(f"After Layer {i}: temp_data shape = {temp_data.shape}")
                    
                if epoch % 10 == 0:
                    print(f'Epoch {epoch}, Loss: {loss:.4f}')
                torch.cuda.empty_cache()

    def save(self, path):
        with open(path, 'wb') as f:
            pickle.dump(self, f)

# iDBN Class
class iDBN:
    def __init__(self, layer_sizes, params):
        self.layers = []
        self.params = params
        for i in range(len(layer_sizes) - 1):
            rbm = RBM(
                num_visible=layer_sizes[i],
                num_hidden=layer_sizes[i + 1],
                learning_rate=params['LEARNING_RATE'],
                weight_decay=params['WEIGHT_PENALTY'],
                momentum=params['INIT_MOMENTUM'],
                dynamic_lr=params['LEARNING_RATE_DYNAMIC'],
                final_momentum=params["FINAL_MOMENTUM"]
            )
            self.layers.append(rbm)

    def train(self, data, epochs): 
        # for layer_idx, rbm in enumerate(self.layers):
        #     print(f'Training iDBN Layer {layer_idx + 1}')
            for epoch in tqdm(range(epochs)):
                temp_data = data.clone()  # Reset to original data for each epoch
                for i, rbm_layer in enumerate(self.layers):
                    print(f"Before Layer {i}: temp_data shape = {temp_data.shape}, W shape = {rbm_layer.W.shape}")
                    
                    # Train the current RBM layer
                    loss = rbm_layer.train_epoch(temp_data, epoch, epochs, CD=1)
                    
                    # Generate output for the next layer
                    temp_data = rbm_layer.forward(temp_data)
                    print(f"After Layer {i}: temp_data shape = {temp_data.shape}")
                    
                if epoch % 10 == 0:
                    print(f'Epoch {epoch}, Loss: {loss:.4f}')
                torch.cuda.empty_cache() 



    def save(self, path):
        with open(path, 'wb') as f:
            pickle.dump(self, f)

# Example usage
# params = {
#     "LEARNING_RATE": 0.15,
#     "WEIGHT_PENALTY": 0.0001,
#     "INIT_MOMENTUM": 0.7,
#     "FINAL_MOMENTUM": 0.97,  # Final momentum for dynamic momentum increase
#     "LEARNING_RATE_DYNAMIC": True,
# }

# # Load preprocessed training dataset
# with open('D:/asya/code/Sep2024/DeWind/python files/batched_train_data_from_mat.pkl', 'rb') as f:
#     train_dataset = pickle.load(f)

# # Convert NumPy array to PyTorch tensor
# train_data = torch.tensor(train_dataset['data'], dtype=torch.float32)
# train_data = train_data.to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))

# # List of layer sizes to test
# layer_sizes_list = [
#     [500, 500], [500, 1000], [500, 1500], [500, 2000],
#     [1000, 500], [1000, 1000], [1000, 1500], [1000, 2000],
#     [1500, 500], [1500, 1000], [1500, 1500], [1500, 2000]
# ]

# idbn = iDBN([train_data.shape[2]] + layer_sizes_list[0], params)
# idbn.train(train_data, epochs=200)
# idbn.save("idbn_trained")
