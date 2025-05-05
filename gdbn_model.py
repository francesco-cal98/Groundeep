import math
import torch
import torch.nn as nn
import pickle
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
import random
import os
from datetime import datetime




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
        batch_size = data.size(0)
        momentum = self.momentum if epoch <= 5 else self.final_momentum
       
        with torch.no_grad():
            
            # Positive phase
            pos_hid_probs = self.forward(data)
            pos_hid_states = (pos_hid_probs > torch.rand_like(pos_hid_probs)).float()
            pos_assoc = torch.matmul(data.T, pos_hid_probs)

            # Negative phase
            neg_data = data
            for _ in range(CD):
                neg_vis_probs = self.backward(pos_hid_states)
                neg_data = (neg_vis_probs > torch.rand_like(neg_vis_probs)).float()
                neg_hid_probs = self.forward(neg_data)
                pos_hid_states = (neg_hid_probs > torch.rand_like(neg_hid_probs)).float()
            neg_assoc = torch.matmul(neg_data.T, neg_hid_probs)

            # Weight and bias updates
            self.W_momentum.mul_(momentum).add_(lr * ((pos_assoc - neg_assoc) / batch_size - self.weight_decay * self.W))
            self.hid_bias_momentum.mul_(momentum).add_(lr * (pos_hid_probs.sum(0) - neg_hid_probs.sum(0)) / batch_size)
            self.vis_bias_momentum.mul_(momentum).add_(lr * (data.sum(0) - neg_data.sum(0)) / batch_size)

            self.W.add_(self.W_momentum)
            self.hid_bias.add_(self.hid_bias_momentum)
            self.vis_bias.add_(self.vis_bias_momentum)

            batch_loss = torch.sum((data - neg_vis_probs) ** 2) / batch_size

            torch.cuda.empty_cache()

        return batch_loss


class gDBN:
    def __init__(self, layer_sizes, params, dataloader, device, log_dir="logs-gdbn"):
        self.layers = []
        self.layer_sizes = layer_sizes
        self.params = params
        self.dataloader = dataloader
        self.device = device
        self.log_dir = log_dir

        # === Build architecture string from layer sizes ===
        arch_str = '-'.join(str(size) for size in layer_sizes)

        # === Create unique log directory with architecture and timestamp ===
        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        run_name = f"run_{arch_str}_{timestamp}"
        full_log_dir = os.path.join(self.log_dir, run_name)
        self.writer = SummaryWriter(full_log_dir)

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

    def train(self, epochs):
        writer = SummaryWriter(log_dir=self.log_dir)
        all_epoch_losses = {}

        # Select sample images for logging
        sample_images, _ = next(iter(self.dataloader))
        sample_images = sample_images[:5].to(self.device)

        for layer_idx, rbm in enumerate(self.layers):
            layer_name = f"RBM_Layer_{layer_idx + 1}"
            print(f'Training {layer_name}')
            epoch_losses = []

            for epoch in tqdm(range(epochs), desc=f"Training {layer_name}"):
                total_loss = 0
                num_batches = 0

                for batch in self.dataloader:
                    train_data, _ = batch
                    train_data = train_data.to(self.device)

                    # If this is NOT the first layer, pass through previous RBMs
                    if layer_idx > 0:
                        with torch.no_grad():  # Prevent gradient tracking
                            for prev_rbm in self.layers[:layer_idx]:  
                                train_data = prev_rbm.forward(train_data)

                    # Train the current RBM
                    batch_loss = rbm.train_epoch(train_data, epoch, epochs, CD=1)
                    total_loss += batch_loss
                    num_batches += 1

                    torch.cuda.empty_cache()

                avg_epoch_loss = total_loss / num_batches
                epoch_losses.append(avg_epoch_loss)
                writer.add_scalar(f"Loss/{layer_name}", avg_epoch_loss, epoch)

            all_epoch_losses[layer_name] = epoch_losses

        writer.close()
        return all_epoch_losses


    def save(self, path):
        self.dataloader = None
        self.writer = None
        with open(path, 'wb') as f:
            pickle.dump(self, f)


# iDBN Class
class iDBN:
    def __init__(self, layer_sizes, params, dataloader, device, log_dir="logs-idbn"):
        self.layers = []
        self.params = params
        self.dataloader = dataloader
        self.device = device

        # === Build architecture string from layer sizes ===
        arch_str = '-'.join(str(size) for size in layer_sizes)

        # === Create unique log directory with architecture and timestamp ===
        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        run_name = f"run_{arch_str}_{timestamp}"
        full_log_dir = os.path.join(log_dir, run_name)
        self.writer = SummaryWriter(full_log_dir)

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

    def train(self, epochs):
            epoch_losses = []

            # Select 5 random images to log reconstruction
            sample_images,_ = next(iter(self.dataloader))  # Get first batch
            sample_images = sample_images[:5].to(self.device)  # Take first 5 images

            for epoch in tqdm(range(epochs)):
                for batch in self.dataloader:
                    train_data,labels = batch 
                    temp_data = train_data.to(self.device)
                    # Train the current RBM layer
                    for i, rbm_layer in enumerate(self.layers):
                        #print(f'Training iDBN Layer {i + 1}')
                        #print(f"Before Layer {i}: temp_data shape = {temp_data.shape}, W shape = {rbm_layer.W.shape}")

                        batch_loss = rbm_layer.train_epoch(temp_data, epoch, epochs, CD=1)
                        
                        # Generate output for the next layer
                        temp_data = rbm_layer.forward(temp_data)
                        #print(f"After Layer {i}: temp_data shape = {temp_data.shape}")
                        epoch_losses.append(batch_loss)

                mean_loss = sum(epoch_losses) / len(epoch_losses) if epoch_losses else 0  # Compute mean loss

                self.writer.add_scalar("Loss/train", mean_loss, epoch)
                # Log reconstructed images every 5 epochs

                if epoch % 5 == 0:
                    with torch.no_grad():
                        reconstructed_images = self.reconstruct(sample_images)
                    self.log_images(sample_images.view(-1,1,100,100), reconstructed_images.view(-1,1,100,100), epoch)
            
                if epoch % 10 == 0:
                    print(f'Epoch {epoch}, Loss: {mean_loss:.4f}')
                torch.cuda.empty_cache()

    def log_images(self, original, reconstructed, epoch):
        """Logs original and reconstructed images to TensorBoard."""
        images = torch.cat((original, reconstructed), dim=0)  # Stack originals and reconstructions
        self.writer.add_images("Reconstruction", images, epoch)


    def reconstruct(self, data):
        with torch.no_grad():
            # Forward pass through the DBN
            temp_data = data
            for rbm in self.layers:
                temp_data = rbm.forward(temp_data)

            # Backward pass through the DBN (reverse order)
            for rbm in reversed(self.layers):
                temp_data = rbm.backward(temp_data)

        return temp_data


    def save(self, path):
        self.dataloader = None
        self.writer = None
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
