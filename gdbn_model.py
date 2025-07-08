import math
import torch
import torch.nn as nn
import pickle
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
import random
import os
import torch.nn.functional as F
from datetime import datetime
import copy


import torch
import torch.nn.functional as F

class UnbiasedRBMSamplerTorch:
    def __init__(self, w, b, c, device='cpu'):
        self.device = device
        self.w = w.to(device)
        self.b = b.to(device)
        self.c = c.to(device)
        self.m, self.n = self.w.shape

    def clip_sigmoid(self, x):
        return torch.sigmoid(torch.clamp(x, -10.0, 10.0))

    def sample_bernoulli(self, prob):
        return (torch.rand(prob.shape, device=self.device) <= prob).float()

    def sample_v_given_h(self, h):
        vmean = self.clip_sigmoid(F.linear(h, self.w.t(), self.b))
        return self.sample_bernoulli(vmean)

    def sample_h_given_v(self, v):
        hmean = self.clip_sigmoid(F.linear(v, self.w, self.c))
        return self.sample_bernoulli(hmean)

    def max_coup(self, vc0, hc0, v1, h1, max_try=10):
        v2mean = self.clip_sigmoid(F.linear(h1, self.w.t(), self.b))
        v2 = self.sample_bernoulli(v2mean)
        h2 = self.sample_h_given_v(v2)

        if torch.norm(v1 - vc0) == 0 and torch.norm(h1 - hc0) == 0:
            return v2.clone(), h2.clone(), v2, h2, 0

        vc1mean = self.clip_sigmoid(F.linear(hc0, self.w.t(), self.b))
        logpxi1 = torch.sum(torch.log(v2mean + 1e-8) * v2 + torch.log(1 - v2mean + 1e-8) * (1 - v2))
        logpeta0 = torch.sum(torch.log(vc1mean + 1e-8) * v2 + torch.log(1 - vc1mean + 1e-8) * (1 - v2))
        u = torch.distributions.Exponential(1.0).sample()

        if u >= (logpxi1 - logpeta0):
            return v2.clone(), h2.clone(), v2, h2, 0

        v2_final = None
        vc1_final = None

        for i in range(max_try):
            uv = torch.rand(self.m, device=self.device)

            if v2_final is None:
                v2_try = (uv <= v2mean).float()
                logpv2 = torch.sum(torch.log(v2mean + 1e-8) * v2_try + torch.log(1 - v2mean + 1e-8) * (1 - v2_try))
                logqv2 = torch.sum(torch.log(vc1mean + 1e-8) * v2_try + torch.log(1 - vc1mean + 1e-8) * (1 - v2_try))
                u1 = torch.distributions.Exponential(1.0).sample()
                if i == max_try - 1 or u1 < (logpv2 - logqv2):
                    v2_final = v2_try

            if vc1_final is None:
                vc1_try = (uv <= vc1mean).float()
                logpvc1 = torch.sum(torch.log(v2mean + 1e-8) * vc1_try + torch.log(1 - v2mean + 1e-8) * (1 - vc1_try))
                logqvc1 = torch.sum(torch.log(vc1mean + 1e-8) * vc1_try + torch.log(1 - vc1mean + 1e-8) * (1 - vc1_try))
                u2 = torch.distributions.Exponential(1.0).sample()
                if i == max_try - 1 or u2 < (logqvc1 - logpvc1):
                    vc1_final = vc1_try

            if v2_final is not None and vc1_final is not None:
                break

        h2mean = self.clip_sigmoid(F.linear(v2_final, self.w, self.c))
        hc1mean = self.clip_sigmoid(F.linear(vc1_final, self.w, self.c))
        uh = torch.rand(self.n, device=self.device)
        h2 = (uh <= h2mean).float()
        hc1 = (uh <= hc1mean).float()

        return vc1_final, hc1, v2_final, h2, i

    def sample(self, v0, min_steps=1, max_steps=100):
        vc = v0
        hc = self.sample_h_given_v(vc)
        v = self.sample_v_given_h(hc)
        h = self.sample_h_given_v(v)

        discarded = 0
        vhist = [v]
        vchist = []

        for i in range(max_steps):
            vc, hc, v, h, disc = self.max_coup(vc, hc, v, h, max_try=10)
            discarded += disc
            vhist.append(v)
            vchist.append(vc)
            if i >= min_steps - 1 and torch.norm(v - vc) == 0 and torch.norm(h - hc) == 0:
                break

        return torch.stack(vhist), torch.stack(vchist), discarded

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
    
    def train_epoch_ucd(self, data, epoch):
        lr = self.lr / (1 + 0.01 * epoch) if self.dynamic_lr else self.lr
        batch_size = data.size(0)
        momentum = self.momentum if epoch <= 5 else self.final_momentum

        # Positive phase
        pos_hid_probs = self.forward(data)  # shape: [batch_size, num_hidden]
        pos_assoc = torch.matmul(data.T, pos_hid_probs)  # shape: [num_visible, num_hidden]

        # Sample binary hidden states from positive probabilities
        pos_hid_states = (pos_hid_probs > torch.rand_like(pos_hid_probs)).float()

        # Negative phase: use unbiased sampler with batch input
        # This should return neg_visible: [batch_size, num_visible]
        neg_visible = self.unbiased_sampler.sample(data)  # You must implement this batch version!
        neg_hid_probs = self.forward(neg_visible)

        neg_assoc = torch.matmul(neg_visible.T, neg_hid_probs)

        # Momentum updates
        self.W_momentum.mul_(momentum).add_(
            lr * ((pos_assoc - neg_assoc) / batch_size - self.weight_decay * self.W)
        )
        self.hid_bias_momentum.mul_(momentum).add_(
            lr * (pos_hid_probs.sum(0) - neg_hid_probs.sum(0)) / batch_size
        )
        self.vis_bias_momentum.mul_(momentum).add_(
            lr * (data.sum(0) - neg_visible.sum(0)) / batch_size
        )

        self.W.add_(self.W_momentum)
        self.hid_bias.add_(self.hid_bias_momentum)
        self.vis_bias.add_(self.vis_bias_momentum)

        batch_loss = torch.sum((data - neg_visible) ** 2) / batch_size

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



import os
import copy
import pickle
from tqdm import tqdm
import torch
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter

class iDBN:
    def __init__(self, layer_sizes, params, dataloader, val_loader, device, log_root="logs-idbn"):
        self.layers = []
        self.params = params
        self.dataloader = dataloader
        self.device = device
        val_batch, _ = next(iter(val_loader))  # Get one batch from validation loader
        self.validation_images = val_batch[:5].to(self.device)  # Take 5 images

        self.arch_str = '-'.join(str(size) for size in layer_sizes)
        self.arch_dir = os.path.join(log_root, f"architecture_{self.arch_str}")
        os.makedirs(self.arch_dir, exist_ok=True)

        self.cd_k = self.params.get("CD", 1)  # Default: CD-k sampler; -1 for unbiased sampler

        for i in range(len(layer_sizes) - 1):
            rbm = RBM(
                num_visible=layer_sizes[i],
                num_hidden=layer_sizes[i + 1],
                learning_rate=params['LEARNING_RATE'],
                weight_decay=params['WEIGHT_PENALTY'],
                momentum=params['INIT_MOMENTUM'],
                dynamic_lr=params['LEARNING_RATE_DYNAMIC'],
                final_momentum=params["FINAL_MOMENTUM"],
            )
            self.layers.append(rbm)

    def train(self, epochs, run_id=0):
        assert self.validation_images is not None, "Validation images must be set before training."
        writer = SummaryWriter(log_dir=os.path.join(self.arch_dir, f"run_{run_id}"))
        epoch_losses = []

        for epoch in tqdm(range(epochs)):
            for train_data, _ in self.dataloader:
                temp_data = train_data.to(self.device)
                for rbm_layer in self.layers:
                    batch_loss = rbm_layer.train_epoch(temp_data, epoch, epochs)
                    temp_data = rbm_layer.forward(temp_data)
                    epoch_losses.append(batch_loss)

            mean_loss = sum(epoch_losses) / len(epoch_losses) if epoch_losses else 0
            writer.add_scalar("Loss/train", mean_loss, epoch)

            if epoch % 5 == 0:
                with torch.no_grad():
                    reconstructed = self.reconstruct(self.validation_images)
                    self.log_images(writer, self.validation_images.view(-1,1,100,100), reconstructed.view(-1,1,100,100), epoch)
                    mse = F.mse_loss(self.validation_images, reconstructed)
                    writer.add_scalar("Reconstruction/MSE", mse.item(), epoch)

            if epoch % 10 == 0:
                print(f'[Run {run_id}] Epoch {epoch}, Loss: {mean_loss:.4f}, MSE: {mse.item():.4f}')
            torch.cuda.empty_cache()

        writer.close()

    def train_multiple_runs(self, epochs=50, runs=10, save_path="networks"):
        for run_id in range(runs):
            print(f"\n=== Training Run {run_id+1}/{runs} for Architecture {self.arch_str} ===")
            self.reinitialize_layers()
            self.train(epochs=epochs, run_id=run_id)
            output_path = os.path.join(save_path, f"{self.arch_dir}/model_run_{run_id}.pkl")
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            self.save_model(output_path)

    def reinitialize_layers(self):
        new_layers = []
        layer_sizes = [layer.num_visible for layer in self.layers] + [self.layers[-1].num_hidden]
        for i in range(len(layer_sizes) - 1):
            rbm = RBM(
                num_visible=layer_sizes[i],
                num_hidden=layer_sizes[i + 1],
                learning_rate=self.params['LEARNING_RATE'],
                weight_decay=self.params['WEIGHT_PENALTY'],
                momentum=self.params['INIT_MOMENTUM'],
                dynamic_lr=self.params['LEARNING_RATE_DYNAMIC'],
                final_momentum=self.params["FINAL_MOMENTUM"],
                device=self.device
            )
            new_layers.append(rbm)
        self.layers = new_layers

    def log_images(self, writer, original, reconstructed, epoch):
        images = torch.cat((original, reconstructed), dim=0)
        writer.add_images("Reconstruction", images, epoch)

    def reconstruct(self, data):
        with torch.no_grad():
            temp_data = data
            for rbm in self.layers:
                temp_data = rbm.forward(temp_data)
            for rbm in reversed(self.layers):
                temp_data = rbm.backward(temp_data)
        return temp_data

    def save_model(self, path):
        model_copy = copy.deepcopy(self)
        model_copy.dataloader = None
        with open(path, 'wb') as f:
            pickle.dump(model_copy, f)
