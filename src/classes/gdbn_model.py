import math
import torch
import torch.nn as nn
import pickle
import numpy as np 
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
import random
import os
import torch.nn.functional as F
from datetime import datetime
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from io import BytesIO
from PIL import Image
import wandb
from src.utils.wandb_utils import (
    plot_2d_embedding_and_correlations,
    plot_3d_embedding_and_correlations,
)
import copy
import umap

import torchvision.utils as vutils

import torch
import torch.nn.functional as F
import sys

current_dir = os.getcwd()
project_root = os.path.abspath(os.path.join(current_dir, '..', '..'))
sys.path.append(project_root)
sys.path.append(current_dir) # Add current_dir if it contains other necessary modules


# importa il tuo modulo utils
from src.utils.probe_utils import log_linear_probe, log_joint_linear_probe  # <— qui l'orchestratore



# Utility function for sigmoid activation
def sigmoid(x):
    return 1 / (1 + torch.exp(-x))

# RBM Class
class RBM(nn.Module):
    def __init__(self, num_visible, num_hidden, learning_rate, weight_decay, momentum, dynamic_lr=False, final_momentum=0.97,sparsity = False,sparsity_factor = 0.05):
        super(RBM, self).__init__()
        self.num_visible = num_visible
        self.num_hidden = num_hidden
        self.lr = learning_rate
        self.weight_decay = weight_decay
        self.momentum = momentum
        self.dynamic_lr = dynamic_lr
        self.final_momentum = final_momentum
        self.sparsity = sparsity
        self.sparsity_factor = sparsity_factor

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

            # ========================================================
            # Positive phase
            # ========================================================
            pos_hid_probs = self.forward(data)
            pos_hid_states = (pos_hid_probs > torch.rand_like(pos_hid_probs)).float()
            pos_assoc = torch.matmul(data.T, pos_hid_probs)

            # ========================================================
            # Negative phase (CD-k)
            # ========================================================
            neg_data = data
            for _ in range(CD):
                neg_vis_probs = self.backward(pos_hid_states)
                neg_data = (neg_vis_probs > torch.rand_like(neg_vis_probs)).float()
                neg_hid_probs = self.forward(neg_data)
                pos_hid_states = (neg_hid_probs > torch.rand_like(neg_hid_probs)).float()
            neg_assoc = torch.matmul(neg_data.T, neg_hid_probs)

            # ========================================================
            # Update weights
            # ========================================================
            self.W_momentum.mul_(momentum).add_(
                lr * ((pos_assoc - neg_assoc) / batch_size - self.weight_decay * self.W)
            )
            self.W.add_(self.W_momentum)

            # ========================================================
            # Update hidden biases (standard RBM update)
            # ========================================================
            self.hid_bias_momentum.mul_(momentum).add_(
                lr * (pos_hid_probs.sum(0) - neg_hid_probs.sum(0)) / batch_size
            )

            # Extra sparsity penalty (not through momentum)
            if self.sparsity:
                Q = pos_hid_probs.sum(0) / batch_size
                avg_Q = Q.mean()
                if avg_Q > self.sparsity_factor:
                    self.hid_bias_momentum.add_(-lr * (Q - self.sparsity_factor))

            self.hid_bias.add_(self.hid_bias_momentum)

            # ========================================================
            # Update visible biases
            # ========================================================
            self.vis_bias_momentum.mul_(momentum).add_(
                lr * (data.sum(0) - neg_data.sum(0)) / batch_size
            )
            self.vis_bias.add_(self.vis_bias_momentum)

            # Reconstruction error
            batch_loss = torch.sum((data - neg_vis_probs) ** 2) / batch_size

            torch.cuda.empty_cache()

        return batch_loss
    
    def conditional_gibbs(self, v_known, known_mask, n_steps=20, sample_h=False):
        """
        v_known: tensor (B, num_visible) - contains observed values where known_mask==1
        known_mask: same shape, values 1.0 = known/locked, 0.0 = to be sampled
        n_steps: number of alternating Gibbs steps
        sample_h: if True sample binary hidden states, else use hidden probabilities
        Returns:
        v_probs: final visible probabilities (B, num_visible)
        """
        device = self.W.device
        v = v_known.clone().to(device)
        km = known_mask.to(device)

        # initialize unknown entries randomly in (0,1)
        v = v * km + (1.0 - km) * torch.rand_like(v)

        with torch.no_grad():
            for _ in range(n_steps):
                # hidden probabilities
                h_probs = sigmoid(torch.matmul(v, self.W) + self.hid_bias)  # (B,H)
                if sample_h:
                    h_states = (h_probs > torch.rand_like(h_probs)).float()
                    h_in = h_states
                else:
                    h_in = h_probs

                # visible probabilities
                v_probs = sigmoid(torch.matmul(h_in, self.W.T) + self.vis_bias)  # (B,V)
                # clamp known entries to their observed values
                v = v_probs * (1.0 - km) + v_known * km

        # return probabilities (useful downstream)
        return v

    
        

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
    def __init__(self, layer_sizes, params, dataloader, val_loader, device,
                 log_root="logs-idbn", text_flag: bool = False, wandb_run=None):
        self.layers = []
        self.params = params
        self.dataloader = dataloader
        self.device = device
        self.text_flag = text_flag
        self.wandb_run = wandb_run
        self.val_loader = val_loader

        if val_loader is not None:
            self.val_batch, val_labels = next(iter(val_loader))
            self.validation_images = (self.val_batch[:5].to(self.device) if not text_flag else val_labels[:5].to(self.device))

            indices = val_loader.dataset.indices
            base_dataset = val_loader.dataset.dataset

            # Cache original labels (could be one-hot for multimodal use-cases)
            self.val_labels_raw = val_labels

            # Convert to numeric labels using the underlying dataset to ensure consistency
            numeric_labels = torch.tensor(
                [base_dataset.labels[i] for i in indices],
                dtype=torch.float32,
            )

            self.labels = numeric_labels
            self.cumArea_vals = [base_dataset.cumArea_list[i] for i in indices]
            self.convex_hull = [base_dataset.CH_list[i] for i in indices]
            density_source = getattr(base_dataset, "density_list", None)
            self.density_vals = [density_source[i] for i in indices] if density_source is not None else None

            # When we operate in multimodal mode keep a copy of one-hot labels for downstream needs
            if isinstance(val_labels, torch.Tensor) and val_labels.dim() > 1:
                self.labels_one_hot = val_labels.detach().clone()
            else:
                self.labels_one_hot = torch.nn.functional.one_hot(
                    (numeric_labels.long() - 1).clamp(min=0), num_classes=32
                ).float()

            self.features = {
                "Cumulative Area": torch.tensor(self.cumArea_vals, dtype=torch.float32),
                "Convex Hull": torch.tensor(self.convex_hull, dtype=torch.float32),
                "Labels": numeric_labels,
            }
            if self.density_vals is not None:
                self.features["Density"] = torch.tensor(self.density_vals, dtype=torch.float32)
        else:
            self.validation_images = None
            self.features = None
            self.density_vals = None

        self.arch_str = '-'.join(str(size) for size in layer_sizes)
        self.arch_dir = os.path.join(log_root, f"architecture_{self.arch_str}")
        os.makedirs(self.arch_dir, exist_ok=True)

        self._features_numpy_cache = None

        self.cd_k = int(self.params.get("CD", 1))
        self.sparsity = self.params.get("SPARSITY", False)
        self.sparsity_factor = self.params.get("SPARSITY_FACTOR", 0.1)

        # Costruzione RBM (come prima)
        for i in range(len(layer_sizes) - 1):
            rbm_params = {
                'num_visible': layer_sizes[i],
                'num_hidden': layer_sizes[i + 1],
                'learning_rate': params['LEARNING_RATE'],
                'weight_decay': params['WEIGHT_PENALTY'],
                'momentum': params['INIT_MOMENTUM'],
                'dynamic_lr': params['LEARNING_RATE_DYNAMIC'],
                'final_momentum': params["FINAL_MOMENTUM"],
            }
            if i == len(layer_sizes) - 2:
                rbm_params['sparsity'] = self.sparsity
                rbm_params['sparsity_factor'] = self.sparsity_factor

            rbm = RBM(**rbm_params)
            rbm.to(self.device)
            self.layers.append(rbm)

    def train(self, epochs, run_id=0, log_every_pca=10, log_every_probe=10):
        writer = SummaryWriter(log_dir=os.path.join(self.arch_dir, f"run_{run_id}"))
        for epoch in tqdm(range(epochs)):
            epoch_losses = []
            for train_data, labels in self.dataloader:
                temp_data = (train_data.to(self.device) if not self.text_flag else labels.to(self.device))
                temp_data = temp_data.view(temp_data.size(0), -1).float()

                for rbm_layer in self.layers:
                    batch_loss = rbm_layer.train_epoch(temp_data, epoch, epochs, CD=self.cd_k)
                    temp_data = rbm_layer.forward(temp_data)
                    epoch_losses.append(batch_loss.item() if isinstance(batch_loss, torch.Tensor) else float(batch_loss))

            mean_loss = sum(epoch_losses) / len(epoch_losses) if epoch_losses else 0.0
            writer.add_scalar("Loss/train", mean_loss, epoch)

            if epoch % 5 == 0 and (not self.text_flag) and (self.validation_images is not None):
                with torch.no_grad():
                    reconstructed = self.reconstruct(self.validation_images)
                    try:
                        imgs_orig = self.validation_images.view(-1, 1, int(math.sqrt(self.validation_images.size(1))), int(math.sqrt(self.validation_images.size(1))))
                        imgs_rec = reconstructed.view(-1, 1, imgs_orig.size(2), imgs_orig.size(3))
                    except Exception:
                        imgs_orig = self.validation_images
                        imgs_rec = reconstructed
                    self.log_images(writer, imgs_orig, imgs_rec, epoch)
                    try:
                        mse = F.mse_loss(imgs_orig, imgs_rec)
                        writer.add_scalar("Reconstruction/MSE", mse.item(), epoch)
                    except Exception:
                        pass

            # PCA logging per layer (first & top, when available)
            if self.wandb_run and self.features and epoch % log_every_pca == 0:
                for layer_idx in self._layers_to_monitor():
                    layer_tag = self._layer_tag(layer_idx)
                    self._log_layer_embeddings(layer_idx, layer_tag)

            # 🔔 Linear probe logging via utils (per monitored layer)
            if self.wandb_run and self.features and self.val_loader is not None and epoch % log_every_probe == 0:
                for layer_idx in self._layers_to_monitor():
                    layer_tag = self._layer_tag(layer_idx)
                    self._log_linear_probe_for_layer(
                        epoch,
                        layer_idx,
                        layer_tag,
                        n_bins=5,
                        test_size=0.2,
                        steps=1000,
                        lr=1e-2,
                        patience=20,
                        min_delta=0.0,
                    )

            torch.cuda.empty_cache()

        writer.close()

    def reinitialize_layers(self):
        layer_sizes = [layer.num_visible for layer in self.layers] + [self.layers[-1].num_hidden]
        new_layers = []
        for i in range(len(layer_sizes) - 1):
            rbm = RBM(
                num_visible=layer_sizes[i],
                num_hidden=layer_sizes[i + 1],
                learning_rate=self.params['LEARNING_RATE'],
                weight_decay=self.params['WEIGHT_PENALTY'],
                momentum=self.params['INIT_MOMENTUM'],
                dynamic_lr=self.params['LEARNING_RATE_DYNAMIC'],
                final_momentum=self.params["FINAL_MOMENTUM"],
            )
            rbm.to(self.device)
            new_layers.append(rbm)
        self.layers = new_layers

    @torch.no_grad()
    def represent(self, x: torch.Tensor, upto_layer: int | None = None) -> torch.Tensor:
        v = x.view(x.size(0), -1).float().to(self.device)

        if upto_layer is None:
            target_layers = len(self.layers)
        else:
            if upto_layer <= 0:
                return v
            target_layers = min(len(self.layers), upto_layer)

        for idx in range(target_layers):
            v = self.layers[idx].forward(v)
        return v

    def _layers_to_monitor(self) -> list[int]:
        layers = {len(self.layers)}
        if not self.text_flag and len(self.layers) > 1:
            layers.add(1)
        return sorted(layers)

    def _layer_tag(self, layer_idx: int) -> str:
        return f"layer{layer_idx}"

    def _get_features_numpy(self) -> dict:
        if self.features is None:
            return {}
        if self._features_numpy_cache is None:
            self._features_numpy_cache = {
                key: (value.detach().cpu().numpy() if isinstance(value, torch.Tensor) else np.asarray(value))
                for key, value in self.features.items()
            }
        return self._features_numpy_cache

    def _validation_source_batch(self) -> torch.Tensor | None:
        if self.text_flag:
            return getattr(self, "val_labels_raw", None)
        return getattr(self, "val_batch", None)

    def _log_layer_embeddings(self, upto_layer: int, layer_tag: str) -> None:
        if not self.wandb_run or self.features is None:
            return

        source_batch = self._validation_source_batch()
        if source_batch is None:
            return

        try:
            with torch.no_grad():
                embeddings = self.represent(source_batch, upto_layer=upto_layer).detach().cpu()
            if embeddings.shape[0] <= 1 or embeddings.shape[1] <= 1:
                return

            emb_np = embeddings.numpy()
            feature_arrays = self._get_features_numpy()
            arch_label = f"{self.arch_str}_{layer_tag}"

            pca2 = PCA(n_components=2)
            emb_2d_pca = pca2.fit_transform(emb_np)
            plot_2d_embedding_and_correlations(
                emb_2d=emb_2d_pca,
                features=feature_arrays,
                arch_name=arch_label,
                dist_name="validation",
                method_name="pca",
                wandb_run=self.wandb_run,
            )

            if embeddings.shape[1] >= 3 and embeddings.shape[0] >= 3:
                pca3 = PCA(n_components=3)
                emb_3d_pca = pca3.fit_transform(emb_np)
                plot_3d_embedding_and_correlations(
                    emb_3d=emb_3d_pca,
                    features=feature_arrays,
                    arch_name=arch_label,
                    dist_name="validation",
                    method_name="pca",
                    wandb_run=self.wandb_run,
                )
        except Exception as e:
            if self.wandb_run:
                self.wandb_run.log({f"warn/pca_logging_error_{layer_tag}": str(e)})

    def _log_linear_probe_for_layer(
        self,
        epoch: int,
        layer_idx: int,
        layer_tag: str,
        n_bins: int = 5,
        test_size: float = 0.2,
        steps: int = 1000,
        lr: float = 1e-2,
        patience: int = 20,
        min_delta: float = 0.0,
    ) -> None:
        try:
            log_linear_probe(
                self,
                epoch=epoch,
                n_bins=n_bins,
                test_size=test_size,
                steps=steps,
                lr=lr,
                patience=patience,
                min_delta=min_delta,
                upto_layer=layer_idx,
                layer_tag=layer_tag,
            )
        except Exception as e:
            if self.wandb_run:
                self.wandb_run.log({f"warn/probe_logging_error_{layer_tag}": str(e)})

    @torch.no_grad()
    def reconstruct(self, x: torch.Tensor) -> torch.Tensor:
        v = x.view(x.size(0), -1).float().to(self.device)
        h_stack = []
        cur = v
        for rbm in self.layers:
            cur = rbm.forward(cur)
            h_stack.append(cur)
        cur = h_stack[-1]
        for rbm in reversed(self.layers):
            cur = rbm.backward(cur)
        return cur.view(v.size(0), -1)

    def log_images(self, writer, original, reconstructed, epoch):
        images = torch.cat((original, reconstructed), dim=0)
        writer.add_images("Reconstruction", images, epoch)

    def save_model(self, path):
        model_copy = copy.deepcopy(self)
        model_copy.dataloader = None
        with open(path, 'wb') as f:
            pickle.dump(model_copy, f)

    def load_model(self, path):
        with open(path, 'rb') as f:
            loaded = pickle.load(f)
        self.__dict__.update(loaded.__dict__)

# -----------------------
# Multimodal iMDBN (joint RBM + supervised head + cross-rec)
# -----------------------

class TextRBMEncoder:
    def __init__(self, num_visible, num_hidden, params, dataloader, val_loader, device,
                 positional_dim=0, log_root="logs-text-rbm", wandb_run=None):
        self.device = device
        self.params = params
        self.dataloader = dataloader
        self.val_loader = val_loader
        self.wandb_run = wandb_run
        self.num_labels = int(params.get("NUM_LABELS", 32))
        self.positional_dim = int(positional_dim)
        self.gaussian_sigma = float(params.get("LABEL_SMOOTH_SIGMA", 0.0))
        self.include_scalar = bool(params.get("INCLUDE_LABEL_SCALAR", False))
        self.fourier_k = int(params.get("LABEL_FOURIER_K", 0))

        # Compute feature dim dynamically: one-hot + posenc + gaussian + scalar/Fourier
        feature_dim = self.num_labels
        if self.positional_dim > 0:
            feature_dim += self.positional_dim
        if self.gaussian_sigma > 0.0:
            feature_dim += self.num_labels
        if self.include_scalar:
            feature_dim += 1 + (2 * self.fourier_k)
        self.feature_dim = feature_dim

        # Precompute positional table if needed
        self.positional_table = None
        if self.positional_dim > 0:
            self.positional_table = self._build_positional_table(self.num_labels, self.positional_dim)

        # Precompute Gaussian basis over labels if needed
        self.gauss_basis = None
        if self.gaussian_sigma > 0.0:
            idx = torch.arange(self.num_labels, dtype=torch.float32).unsqueeze(1)
            centers = torch.arange(self.num_labels, dtype=torch.float32).unsqueeze(0)
            d2 = (idx - centers) ** 2  # [L,L]
            basis = torch.exp(-d2 / (2.0 * (self.gaussian_sigma ** 2)))
            self.gauss_basis = (basis / (basis.sum(dim=1, keepdim=True) + 1e-8)).to(self.device)

        text_lr = params.get('TEXT_LEARNING_RATE', params['LEARNING_RATE'])
        self.rbm = RBM(
            num_visible=self.feature_dim,
            num_hidden=num_hidden,
            learning_rate=text_lr,
            weight_decay=params['WEIGHT_PENALTY'],
            momentum=params['INIT_MOMENTUM'],
            dynamic_lr=params['LEARNING_RATE_DYNAMIC'],
            final_momentum=params['FINAL_MOMENTUM'],
        )
        self.rbm.to(self.device)
        self.layers = [self.rbm]

        self.arch_str = f"{self.feature_dim}-{num_hidden}"
        self.arch_dir = os.path.join(log_root, f"text_rbm_{self.arch_str}")
        os.makedirs(self.arch_dir, exist_ok=True)

        if val_loader is not None:
            _, val_labels = next(iter(val_loader))
            self.val_samples = val_labels[:5].to(self.device)
        else:
            self.val_samples = None

    def train(self, epochs, run_id=0):
        writer = SummaryWriter(log_dir=os.path.join(self.arch_dir, f"run_{run_id}"))
        cd_k = int(self.params.get("JOINT_CD", self.params.get("CD", 1)))
        for epoch in range(epochs):
            epoch_losses = []
            for _, labels in self.dataloader:
                batch = self._make_features(labels)
                batch_loss = self.rbm.train_epoch(batch, epoch, epochs, CD=cd_k)
                value = batch_loss.item() if isinstance(batch_loss, torch.Tensor) else float(batch_loss)
                epoch_losses.append(value)
            if epoch_losses:
                mean_loss = sum(epoch_losses) / len(epoch_losses)
                writer.add_scalar("Loss/text_rbm", mean_loss, epoch)
                if self.wandb_run:
                    self.wandb_run.log({"text_rbm/loss": mean_loss, "epoch": epoch})
            torch.cuda.empty_cache()
        writer.close()

        self._log_validation_pca()

    @torch.no_grad()
    def represent(self, labels):
        feat = self._make_features(labels)
        return self.rbm.forward(feat)

    @torch.no_grad()
    def decode(self, hidden):
        hidden = hidden.to(self.device).float()
        decoded = self.rbm.backward(hidden)
        if decoded.size(1) > self.num_labels:
            return decoded[:, :self.num_labels]
        return decoded
    
    def _make_features(self, labels):
        labels = labels.to(self.device).float()
        idx = torch.argmax(labels, dim=1)

        feats = [labels]
        if self.positional_dim > 0 and self.positional_table is not None:
            pos = self.positional_table.to(self.device).index_select(0, idx)
            feats.append(pos)
        if self.gaussian_sigma > 0.0 and self.gauss_basis is not None:
            gauss = self.gauss_basis.index_select(0, idx)
            feats.append(gauss)
        if self.include_scalar:
            x = idx.float() / max(1.0, float(self.num_labels - 1))
            x = x.unsqueeze(1)
            scalars = [x]
            if self.fourier_k > 0:
                two_pi = 2.0 * math.pi
                ks = torch.arange(1, self.fourier_k + 1, device=self.device, dtype=torch.float32).unsqueeze(0)
                ang = two_pi * x @ ks
                scalars.append(torch.sin(ang))
                scalars.append(torch.cos(ang))
            feats.append(torch.cat(scalars, dim=1))
        return torch.cat(feats, dim=1)

    @staticmethod
    def _build_positional_table(num_labels: int, dim: int) -> torch.Tensor:
        if dim <= 0:
            return None
        position = torch.arange(num_labels, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, dim, 2, dtype=torch.float32) * (-math.log(10000.0) / dim))
        table = torch.zeros(num_labels, dim, dtype=torch.float32)
        table[:, 0::2] = torch.sin(position * div_term)
        table[:, 1::2] = torch.cos(position * div_term)
        table = (table + 1.0) / 2.0  # map to [0,1]
        return table

    def _log_validation_pca(self):
        if self.wandb_run is None or self.val_loader is None:
            return

        embeddings = []
        labels_numeric = []
        with torch.no_grad():
            for _, labs in self.val_loader:
                labs = labs.to(self.device).float()
                emb = self.represent(labs)
                embeddings.append(emb.detach().cpu())
                labels_numeric.append(torch.argmax(labs, dim=1).cpu())

        if not embeddings:
            return

        emb = torch.cat(embeddings, dim=0)
        if emb.size(0) < 2 or emb.size(1) < 2:
            return

        emb_np = emb.numpy()
        labels_np = torch.cat(labels_numeric, dim=0).numpy()

        try:
            pca2 = PCA(n_components=2)
            emb2 = pca2.fit_transform(emb_np)
            fig, ax = plt.subplots(figsize=(6, 5))
            scatter = ax.scatter(emb2[:, 0], emb2[:, 1], c=labels_np, cmap='tab20', s=12, alpha=0.7)
            ax.set_title('Text RBM PCA (validation)')
            ax.set_xlabel('PC1')
            ax.set_ylabel('PC2')
            plt.colorbar(scatter, ax=ax, label='Label')
            fig.tight_layout()
            self.wandb_run.log({'text_rbm/pca2d': wandb.Image(fig)})
            plt.close(fig)
        except Exception as exc:
            self.wandb_run.log({'warn/text_rbm_pca_error': str(exc)})


class iMDBN(nn.Module):
    def __init__(self, layer_sizes_img, layer_sizes_txt, joint_layer_size, params,
                 dataloader, val_loader, device, text_posenc_dim=0,
                 log_root="logs-imdbn",num_labels =32, embedding_dim=64,
                 wandb_run=None):
        super(iMDBN, self).__init__()
        self.params = params
        self.device = device
        self.dataloader = dataloader
        self.val_loader = val_loader
        self.wandb_run = wandb_run

        self.num_labels = num_labels
        self.embedding_dim = embedding_dim



        # Validation batch (images + labels)
        val_batch, val_labels = next(iter(val_loader))
        self.validation_images = val_batch[:5].to(self.device)
        self.validation_labels = val_labels[:5].to(self.device)
        self.val_batch = (val_batch, val_labels)

        self.arch_str = f"IMG{'-'.join(map(str,layer_sizes_img))}_TXT{'-'.join(map(str,layer_sizes_txt))}_JOINT{joint_layer_size}"
        self.arch_dir = os.path.join(log_root, f"architecture_{self.arch_str}")
        os.makedirs(self.arch_dir, exist_ok=True)

        params["NUM_LABELS"] = self.num_labels

        # Unimodal encoders (image: iDBN, text: single RBM)
        self.image_idbn = iDBN(layer_sizes=layer_sizes_img, params=params,
                               dataloader=self.dataloader, val_loader=val_loader,
                               device=device, log_root=log_root, text_flag=False,
                               wandb_run=wandb_run)

        text_visible = layer_sizes_txt[0]
        text_hidden = layer_sizes_txt[-1]
        self.text_encoder = TextRBMEncoder(
            num_visible=text_visible,
            num_hidden=text_hidden,
            params=params,
            dataloader=self.dataloader,
            val_loader=val_loader,
            device=device,
            log_root=log_root,
            positional_dim=text_posenc_dim,
            wandb_run=wandb_run,
        )

        # Joint RBM on concatenated top-level reps
        joint_rbm_visible_size = layer_sizes_img[-1] + layer_sizes_txt[-1]
        self.joint_rbm = RBM(
            num_visible=joint_rbm_visible_size,
            num_hidden=joint_layer_size,
            learning_rate=params['LEARNING_RATE'],
            weight_decay=params['WEIGHT_PENALTY'],
            momentum=params['INIT_MOMENTUM'],
            dynamic_lr=params['LEARNING_RATE_DYNAMIC'],
            final_momentum=params["FINAL_MOMENTUM"],
        )
        self.joint_rbm.to(self.device)
        self.joint_cd = int(params.get("JOINT_CD", params.get("CD", 1)))
        self.cross_gibbs_steps = int(params.get("CROSS_GIBBS_STEPS", 10))

        # supervised head from joint hidden -> predict N (regression)
        self.supervised_head = nn.Linear(joint_layer_size, 1).to(self.device)
        self.sup_optimizer = torch.optim.Adam(self.supervised_head.parameters(), lr=1e-3)

        # Cache dataset-driven features for downstream analyses (mirrors iDBN behaviour)
        indices = val_loader.dataset.indices
        base_dataset = val_loader.dataset.dataset

        numeric_labels = torch.tensor(
            [base_dataset.labels[i] for i in indices], dtype=torch.float32
        )
        self.val_labels_numeric = numeric_labels

        self.features = {
            "Cumulative Area": torch.tensor(
                [base_dataset.cumArea_list[i] for i in indices], dtype=torch.float32
            ),
            "Convex Hull": torch.tensor(
                [base_dataset.CH_list[i] for i in indices], dtype=torch.float32
            ),
            "Labels": numeric_labels,
        }

        density_source = getattr(base_dataset, "density_list", None)
        if density_source is not None:
            self.features["Density"] = torch.tensor(
                [density_source[i] for i in indices], dtype=torch.float32
            )

    @torch.no_grad()
    def represent(self, batch):
        """Return the joint hidden representation given (images, one-hot labels)."""
        if isinstance(batch, (tuple, list)):
            img_data, label_data = batch
        else:
            raise ValueError("Expected batch to be a tuple (images, labels) for multimodal represent().")

        img_data = img_data.to(self.device).view(img_data.size(0), -1).float()
        label_data = label_data.to(self.device).float()

        img_rep = self.image_idbn.represent(img_data)
        txt_rep = self.text_encoder.represent(label_data)

        joint_input = torch.cat((img_rep, txt_rep), dim=1)
        return self.joint_rbm.forward(joint_input)

    # -----------------
    def forward(self, img_data, labels):
        # returns joint hidden probabilities (no grad)
        with torch.no_grad():
            img_rep = self.image_idbn.represent(img_data.to(self.device))
            txt_rep = self.text_encoder.represent(labels.to(self.device))
            joint_input = torch.cat((img_rep, txt_rep), dim=1)
            joint_h = self.joint_rbm.forward(joint_input)
        return joint_h

    # -----------------
    def _cross_reconstruct(self, z_img, z_txt, gibbs_steps=None):
        """
        Return:
        recon_img_from_txt, recon_txt_from_img
        z_img: (B, Dz_img)  top-level image rep (probabilities)
        z_txt: (B, Dz_txt)  top-level text rep (probabilities)
        """
        with torch.no_grad():
            if gibbs_steps is None:
                gibbs_steps = self.cross_gibbs_steps
            B = z_img.size(0)
            Dz_img = z_img.size(1)
            Dz_txt = z_txt.size(1)
            Vdim = Dz_img + Dz_txt

            # ensure same device & dtype
            z_img = z_img.to(self.device)
            z_txt = z_txt.to(self.device)

            # normalize / clamp if useful (recommended)
            # z_img = F.layer_norm(z_img, z_img.shape[1:])
            # z_txt = F.layer_norm(z_txt, z_txt.shape[1:])

            # --- IMAGE -> TEXT (clamp image top) ---
            v_known = torch.zeros((B, Vdim), device=self.device)
            known_mask = torch.zeros_like(v_known, device=self.device)
            v_known[:, :Dz_img] = z_img
            known_mask[:, :Dz_img] = 1.0

            v_final_from_img = self.joint_rbm.conditional_gibbs(v_known, known_mask, n_steps=gibbs_steps, sample_h=False)
            v_img_top_from_img = v_final_from_img[:, :Dz_img]
            v_txt_top_from_img = v_final_from_img[:, Dz_img:]

            # decode text-top -> text space
            recon_txt_from_img = self.text_encoder.decode(v_txt_top_from_img)

            # --- TEXT -> IMAGE (clamp text top) ---
            v_known = torch.zeros((B, Vdim), device=self.device)
            known_mask = torch.zeros_like(v_known, device=self.device)
            v_known[:, Dz_img:] = z_txt
            known_mask[:, Dz_img:] = 1.0

            v_final_from_txt = self.joint_rbm.conditional_gibbs(v_known, known_mask, n_steps=gibbs_steps, sample_h=False)
            v_img_top_from_txt = v_final_from_txt[:, :Dz_img]
            v_txt_top_from_txt = v_final_from_txt[:, Dz_img:]

            # decode image-top -> image space
            recon_img_from_txt = v_img_top_from_txt
            for rbm in reversed(self.image_idbn.layers):
                recon_img_from_txt = rbm.backward(recon_img_from_txt)

            return recon_img_from_txt, recon_txt_from_img


    # -----------------
    def train_joint(self, epochs, run_id=0, w_rec=1.0, w_sup=1.0, log_every_pca=10, log_every_probe=10):
        """
        Train joint RBM (via CD inside RBM.train_epoch) and also train a supervised
        head that predicts N from joint hidden probabilities.
        - w_rec: weight for cross-reconstruction loss (BCE for images, CE for one-hot)
        - w_sup: weight for supervised MSE loss
        """
        writer = SummaryWriter(log_dir=os.path.join(self.arch_dir, f"run_{run_id}"))
        cd_k = int(self.params.get("CD", 1))

        for epoch in tqdm(range(epochs)):
            epoch_losses = []
            sup_losses = []
            rec_losses = []
            # Train-set cross-reconstruction accumulators (log once per epoch)
            tr_total = 0
            tr_correct1 = 0
            tr_correct3 = 0
            tr_ce_sum = 0.0
            tr_mse_sum = 0.0
            last_npix = None
            # Auxiliary clamped-CD settings (unsupervised strengthening of conditionals)
            aux_every_k = int(self.params.get("JOINT_AUX_EVERY_K", 2))  # run aux every k batches
            aux_cd_k = int(self.params.get("JOINT_AUX_CD", 1))
            aux_cond_steps = int(self.params.get("JOINT_AUX_COND_STEPS", 25))
            aux_lr_scale = float(self.params.get("JOINT_AUX_LR_SCALE", 0.2))

            for b_idx, (img_data, labels) in enumerate(self.dataloader):
                img_data = img_data.to(self.device).view(img_data.size(0), -1).float()
                labels_one_hot = labels.to(self.device).float()
                numeric_targets = torch.argmax(labels_one_hot, dim=1, keepdim=True).float() + 1.0
                #label_indices = labels.to(self.device).long()  # assume 0..num_labels-1
                #labels_dense = self.label_embeddings(label_indices)  # (B, embedding_dim)

                # get top-level reps (freeze unimodals)
                with torch.no_grad():
                    z_img = self.image_idbn.represent(img_data)
                    z_txt = self.text_encoder.represent(labels_one_hot)

                joint_input = torch.cat((z_img, z_txt), dim=1).to(self.device)

                # Removed modality masking: train joint RBM only on full (z_img, z_txt)


                # 1) Train joint RBM via CD (updates internal weights)
                batch_loss_cd = self.joint_rbm.train_epoch(joint_input, epoch, epochs, CD=cd_k)
                epoch_losses.append(float(batch_loss_cd.item() if isinstance(batch_loss_cd, torch.Tensor) else batch_loss_cd))

                # 2) Cross reconstruction losses (differentiable parts: supervised head; we use recon outputs as targets)
                # Compute reconstructions from single modality (no grad for RBM ops here)
                recon_img_from_txt, recon_txt_from_img = self._cross_reconstruct(z_img, z_txt)

                # image reconstruction loss: BCE between recon_img_from_txt (prob) and img_data (0/1)
                # ensure shapes
                recon_img_from_txt = recon_img_from_txt.view(img_data.size(0), -1)
                L_rec_img = F.binary_cross_entropy(recon_img_from_txt.clamp(1e-6, 1-1e-6), img_data.clamp(0.0, 1.0), reduction='mean')

                # text reconstruction loss: BCE on one-hot targets (independent Bernoulli per class)
                recon_txt_from_img = recon_txt_from_img.view(labels_one_hot.size(0), -1)
                L_rec_txt = F.binary_cross_entropy(
                    recon_txt_from_img.clamp(1e-6, 1 - 1e-6),
                    labels_one_hot,
                    reduction='mean'
                )

                L_rec = (L_rec_img + L_rec_txt)
                rec_losses.append(float((w_rec * L_rec).item()))

                if w_sup > 0:
                    joint_h = self.joint_rbm.forward(joint_input)
                    pred_N = self.supervised_head(joint_h)
                    L_sup = F.mse_loss(pred_N, numeric_targets, reduction='mean') * w_sup
                    sup_losses.append(float(L_sup.item()))

                    self.sup_optimizer.zero_grad()
                    L_sup.backward()
                    self.sup_optimizer.step()
                else:
                    sup_losses.append(0.0)

                # Accumulate train cross-reconstruction metrics
                with torch.no_grad():
                    # text metrics (BCE vs one-hot)
                    gt_idx = labels_one_hot.argmax(dim=1)
                    pred_idx = recon_txt_from_img.argmax(dim=1)
                    topk_idx = recon_txt_from_img.topk(k=min(3, recon_txt_from_img.size(1)), dim=1).indices
                    bsz = img_data.size(0)
                    tr_total += bsz
                    tr_correct1 += (pred_idx == gt_idx).sum().item()
                    tr_correct3 += (topk_idx == gt_idx.unsqueeze(1)).any(dim=1).sum().item()
                    tr_ce_sum += float(F.binary_cross_entropy(
                        recon_txt_from_img.clamp(1e-6, 1 - 1e-6),
                        labels_one_hot,
                        reduction='sum'
                    ).item())

                    # image mse (sum over pixels)
                    recon_flat = recon_img_from_txt.view(img_data.size(0), -1)
                    target_flat = img_data.view(img_data.size(0), -1)
                    last_npix = target_flat.size(1)
                    tr_mse_sum += float(F.mse_loss(recon_flat, target_flat, reduction='sum').item())

                # 3) Auxiliary clamped-CD (unsupervised): strengthen p(z_txt|z_img) and p(z_img|z_txt)
                do_aux = (aux_every_k > 0) and (b_idx % aux_every_k == 0)
                if do_aux:
                    B = z_img.size(0)
                    Dz_img = z_img.size(1)
                    Dz_txt = z_txt.size(1)
                    Vdim = Dz_img + Dz_txt

                    # Alternate which side to clamp (even batches: image-clamped; odd: text-clamped)
                    clamp_image = (b_idx % 2 == 0)

                    v_known = torch.zeros((B, Vdim), device=self.device)
                    known_mask = torch.zeros_like(v_known)
                    if clamp_image:
                        v_known[:, :Dz_img] = z_img
                        known_mask[:, :Dz_img] = 1.0
                    else:
                        v_known[:, Dz_img:] = z_txt
                        known_mask[:, Dz_img:] = 1.0

                    # Fill unknown modality with conditional Gibbs (no grad)
                    with torch.no_grad():
                        v_fill = self.joint_rbm.conditional_gibbs(
                            v_known, known_mask, n_steps=aux_cond_steps, sample_h=False
                        )

                    # Temporarily scale LR for a lighter aux update
                    old_lr = float(self.joint_rbm.lr)
                    try:
                        self.joint_rbm.lr = max(1e-8, old_lr * aux_lr_scale)
                        aux_loss = self.joint_rbm.train_epoch(v_fill, epoch, epochs, CD=aux_cd_k)
                        epoch_losses.append(float(aux_loss.item() if isinstance(aux_loss, torch.Tensor) else aux_loss))
                    finally:
                        self.joint_rbm.lr = old_lr

            # epoch logging
            mean_cd = sum(epoch_losses) / len(epoch_losses) if epoch_losses else 0.0
            mean_rec = sum(rec_losses) / len(rec_losses) if rec_losses else 0.0
            mean_sup = sum(sup_losses) / len(sup_losses) if sup_losses else 0.0
            writer.add_scalar("JointRBM/CD_loss", mean_cd, epoch)
            writer.add_scalar("JointRBM/rec_loss", mean_rec, epoch)
            writer.add_scalar("JointRBM/sup_loss", mean_sup, epoch)
            if self.wandb_run:
                self.wandb_run.log({
                    "joint/cd_loss": mean_cd,
                    "joint/reconstruction_loss": mean_rec,
                    "joint/supervised_loss": mean_sup,
                    "epoch": epoch
                })
            # Epoch-level train cross-reconstruction metrics
            if tr_total > 0:
                tr_top1 = tr_correct1 / tr_total
                tr_top3 = tr_correct3 / tr_total
                tr_ce_mean = tr_ce_sum / tr_total
                # normalize MSE per-pixel per-sample
                denom = max(1, tr_total * max(1, (last_npix or 1)))
                tr_mse_mean = tr_mse_sum / denom
                writer.add_scalar("CrossModalityTrain/TextTop1", tr_top1, epoch)
                writer.add_scalar("CrossModalityTrain/TextTop3", tr_top3, epoch)
                writer.add_scalar("CrossModalityTrain/TextBCE", tr_ce_mean, epoch)
                writer.add_scalar("CrossModalityTrain/ImageMSE", tr_mse_mean, epoch)
                if self.wandb_run:
                    self.wandb_run.log({
                        "cross_modality_train/text_top1": tr_top1,
                        "cross_modality_train/text_top3": tr_top3,
                        "cross_modality_train/text_bce": tr_ce_mean,
                        "cross_modality_train/image_mse": tr_mse_mean,
                        "cross_modality_train/n": int(tr_total),
                        "epoch": epoch,
                    })
            #self.log_joint_cosine_similarity(writer, epoch)
            self.log_joint_diagnostics(writer, epoch, num_batches=5)
            if epoch % log_every_probe == 0:
                log_joint_linear_probe(
                    self,
                    epoch=epoch,
                    n_bins=5,
                    test_size=0.2,
                    steps=1000,
                    lr=1e-2,
                    rng_seed=42,
                    patience=20,
                    min_delta=0.0,
                    save_csv=True,
                    metric_prefix="joint"
                )

            # PCA logging for multimodal joint representations
            if self.wandb_run and self.features and epoch % log_every_pca == 0:
                try:
                    with torch.no_grad():
                        embeddings = self.represent(self.val_batch).detach().cpu()
                    if embeddings.shape[0] > 1 and embeddings.shape[1] > 2:
                        n_neighbors_umap = min(embeddings.shape[0] - 1, 15)
                        umap.UMAP(n_components=2, random_state=42, n_neighbors=n_neighbors_umap)
                        pca2 = PCA(n_components=2)
                        emb_2d_pca = pca2.fit_transform(embeddings.numpy())
                        plot_2d_embedding_and_correlations(
                            emb_2d=emb_2d_pca,
                            features=self.features,
                            arch_name=self.arch_str,
                            dist_name="validation",
                            method_name="pca",
                            wandb_run=self.wandb_run,
                        )
                        if embeddings.shape[1] >= 3 and embeddings.shape[0] >= 3:
                            pca3 = PCA(n_components=3)
                            emb_3d_pca = pca3.fit_transform(embeddings.numpy())
                            plot_3d_embedding_and_correlations(
                                emb_3d=emb_3d_pca,
                                features=self.features,
                                arch_name=self.arch_str,
                                dist_name="validation",
                                method_name="pca",
                                wandb_run=self.wandb_run,
                            )
                except Exception as e:
                    if self.wandb_run:
                        self.wandb_run.log({"warn/imdbn_pca_logging_error": str(e)})



            if epoch % 10 == 0:
                print(f"[Joint RBM] Epoch {epoch}, CD_loss={mean_cd:.4f}, rec={mean_rec:.4f}, sup={mean_sup:.4f}")
            if epoch % 5 == 0:
                self.log_cross_modality(writer, epoch, num_samples=5)
                self.log_cross_modality_full(writer, epoch)


            torch.cuda.empty_cache()

        writer.close()

    # -----------------
    def save_model(self, path):
        model_copy = copy.deepcopy(self)
        model_copy.dataloader = None
        if hasattr(model_copy, 'text_encoder'):
            model_copy.text_encoder.dataloader = None
            model_copy.text_encoder.val_loader = None
        with open(path, 'wb') as f:
            pickle.dump(model_copy, f)
        print(f"Model saved to {path}")

    def load_model(self, path):
        with open(path, 'rb') as f:
            loaded = pickle.load(f)
        self.__dict__.update(loaded.__dict__)
        print(f"Model loaded from {path}")

    def log_cross_modality(self, writer, epoch, num_samples=5):
        img_data = self.validation_images[:num_samples]
        lbl_data = self.validation_labels[:num_samples]

        with torch.no_grad():
            # top-level reps
            z_img = self.image_idbn.represent(img_data)
            z_txt = self.text_encoder.represent(lbl_data)
            recon_img_from_txt, recon_txt_from_img = self._cross_reconstruct(z_img, z_txt)

        # ---- immagini ----
        recon_img_from_txt_norm = recon_img_from_txt.clamp(0, 1)
        try:
            _, C, H, W = self.validation_images.shape
        except Exception:
            side = int(recon_img_from_txt_norm.size(1) ** 0.5)
            C, H, W = 1, side, side
        img_grid = recon_img_from_txt_norm.view(-1, C, H, W)
        grid_tensor = vutils.make_grid(img_grid.cpu(), nrow=min(num_samples, img_grid.size(0)))
        writer.add_image('CrossModality/Image_from_Label', grid_tensor, epoch)
        
        # ---- labels ----
        recon_probs = recon_txt_from_img.clamp(0, 1)

        pred_indices = recon_probs.argmax(dim=1)  # predizione numerica
        gt_indices = lbl_data.argmax(dim=1)
        top1_acc = (pred_indices == gt_indices).float().mean().item()
        writer.add_scalar("CrossModality/TextTop1", top1_acc, epoch)

        topk = min(3, recon_probs.size(1))
        topk_indices = recon_probs.topk(topk, dim=1).indices
        topk_acc = (topk_indices == gt_indices.unsqueeze(1)).any(dim=1).float().mean().item()
        writer.add_scalar("CrossModality/TextTop3", topk_acc, epoch)

        gt_onehot = F.one_hot(gt_indices, num_classes=recon_probs.size(1)).float()
        text_ce = F.binary_cross_entropy(recon_probs.clamp(1e-6, 1 - 1e-6), gt_onehot)
        writer.add_scalar("CrossModality/TextCrossEntropy", text_ce.item(), epoch)

        img_mse = F.mse_loss(recon_img_from_txt.view(img_data.size(0), -1), img_data.view(img_data.size(0), -1))
        writer.add_scalar("CrossModality/ImageMSE", img_mse.item(), epoch)

        table_rows = []
        for i in range(min(num_samples, recon_probs.size(0))):
            gt = int(gt_indices[i].item())
            pred = int(pred_indices[i].item())
            table_rows.append((i, pred, gt))

        table_str = " | Sample | Prediction | GroundTruth |\n"
        table_str += "|--------|------------|-------------|\n"
        for sample_idx, pred, gt in table_rows:
            table_str += f"| {sample_idx} | {pred} | {gt} |\n"

        writer.add_text("CrossModality/Predictions", table_str, epoch)

        if self.wandb_run:
            grid_np = grid_tensor.detach().cpu()
            self.wandb_run.log({
                "cross_modality/image_from_text": wandb.Image(grid_np.permute(1, 2, 0).numpy(), caption=f"Epoch {epoch}"),
                "cross_modality/text_top1": top1_acc,
                "cross_modality/text_top3": topk_acc,
                "cross_modality/text_ce": text_ce.item(),
                "cross_modality/image_mse": img_mse.item(),
                "epoch": epoch
            })
            wandb_table = wandb.Table(columns=["sample", "prediction", "ground_truth"],
                                      data=[[int(s), int(p), int(g)] for s, p, g in table_rows])
            self.wandb_run.log({"cross_modality/text_predictions": wandb_table, "epoch": epoch})

            cm_plot = wandb.plot.confusion_matrix(
                y_true=gt_indices.cpu().numpy(),
                preds=pred_indices.cpu().numpy(),
                class_names=[str(i + 1) for i in range(self.num_labels)]
            )
            self.wandb_run.log({"cross_modality/text_confusion": cm_plot, "epoch": epoch})

    def log_cross_modality_full(self, writer, epoch):
        if self.val_loader is None:
            return
        total = 0
        correct1 = 0
        correct3 = 0
        ce_sum = 0.0
        mse_sum = 0.0
        with torch.no_grad():
            for img_data, lbl_data in self.val_loader:
                imgs = img_data.to(self.device).view(img_data.size(0), -1).float()
                labs = lbl_data.to(self.device).float()

                z_img = self.image_idbn.represent(imgs)
                z_txt = self.text_encoder.represent(labs)
                recon_img_from_txt, recon_txt_from_img = self._cross_reconstruct(z_img, z_txt)

                # Text metrics
                gt_idx = labs.argmax(dim=1)
                pred_idx = recon_txt_from_img.argmax(dim=1)
                topk_idx = recon_txt_from_img.topk(k=min(3, recon_txt_from_img.size(1)), dim=1).indices
                gt_onehot = F.one_hot(gt_idx, num_classes=recon_txt_from_img.size(1)).float()
                ce = F.binary_cross_entropy(
                    recon_txt_from_img.clamp(1e-6, 1 - 1e-6),
                    gt_onehot,
                    reduction='sum'
                )

                # Image MSE
                mse = F.mse_loss(recon_img_from_txt.view_as(imgs), imgs, reduction='sum')

                bsz = imgs.size(0)
                total += bsz
                correct1 += (pred_idx == gt_idx).sum().item()
                correct3 += (topk_idx == gt_idx.unsqueeze(1)).any(dim=1).sum().item()
                ce_sum += float(ce.item())
                mse_sum += float(mse.item())

        if total > 0:
            top1 = correct1 / total
            top3 = correct3 / total
            ce_mean = ce_sum / total
            # Normalize image MSE per pixel per sample for better scale
            npix = None
            try:
                npix = int(self.validation_images.view(self.validation_images.size(0), -1).size(1))
            except Exception:
                npix = imgs.size(1)
            denom = max(1, total * max(1, npix))
            mse_mean = mse_sum / denom
            writer.add_scalar("CrossModalityFull/TextTop1", top1, epoch)
            writer.add_scalar("CrossModalityFull/TextTop3", top3, epoch)
            writer.add_scalar("CrossModalityFull/TextCrossEntropy", ce_mean, epoch)
            writer.add_scalar("CrossModalityFull/ImageMSE", mse_mean, epoch)
            if self.wandb_run:
                self.wandb_run.log({
                    "cross_modality_full/text_top1": top1,
                    "cross_modality_full/text_top3": top3,
                    "cross_modality_full/text_ce": ce_mean,
                    "cross_modality_full/image_mse": mse_mean,
                    "cross_modality_full/n": int(total),
                    "epoch": epoch,
                })



    def log_joint_cosine_similarity(self, writer, epoch, num_batches=5):
        """
        Calcola e logga la cosine similarity tra rappresentazioni immagine e testo
        nel joint RBM hidden space.
        """
        cos_sims = []

        with torch.no_grad():
            for i, (img_data, labels) in enumerate(self.dataloader):
                if i >= num_batches:
                    break

                img_data = img_data.to(self.device).view(img_data.size(0), -1).float()
                labels = labels.to(self.device).float()

                # Top-level unimodal reps
                z_img = F.normalize(self.image_idbn.represent(img_data), dim=1)
                z_txt = F.normalize(self.text_encoder.represent(labels), dim=1)

                cos_dim = min(z_img.size(1), z_txt.size(1))
                if cos_dim > 0:
                    sim = F.cosine_similarity(
                        z_img[:, :cos_dim], z_txt[:, :cos_dim], dim=1
                    )
                    cos_sims.append(sim.mean().item())

                joint_h_img = self.joint_rbm.forward(torch.cat([z_img, torch.zeros_like(z_txt)], dim=1))
                joint_h_txt = self.joint_rbm.forward(torch.cat([torch.zeros_like(z_img), z_txt], dim=1))
                sim_joint = F.cosine_similarity(joint_h_img, joint_h_txt, dim=1)
                cos_sims.append(sim_joint.mean().item())

        mean_cos = sum(cos_sims) / len(cos_sims) if cos_sims else 0.0
        writer.add_scalar("JointRBM/cosine_similarity_img_txt", mean_cos, epoch)
        print(f"[Epoch {epoch}] Cosine similarity (img vs txt in joint space): {mean_cos:.4f}")

    # (removed stray import) F is already imported at module level

    def log_joint_diagnostics(self, writer, epoch, num_batches=5):
        """
        Calcola e logga:
        - cosine(z_img, z_txt) (prima del joint RBM)
        - cosine(joint_h_img, joint_h_txt) (dopo joint RBM forward)
        - distribuzioni / istogrammi e norme
        Chiamalo da train_joint per avere diagnostica ogni epoca.
        """
        cos_z_list = []
        cos_joint_list = []
        z_img_means, z_txt_means = [], []
        joint_img_norms, joint_txt_norms = [], []

        with torch.no_grad():
            for i, (img_data, labels) in enumerate(self.dataloader):
                if i >= num_batches:
                    break

                # Prepara batch
                imgs = img_data.to(self.device).view(img_data.size(0), -1).float()
                labs = labels.to(self.device).float()

                # Top-level unimodal reps
                z_img = F.normalize(self.image_idbn.represent(imgs), dim=1)   # (B, D_img)
                z_txt = F.normalize(self.text_encoder.represent(labs), dim=1)   # (B, D_txt)

                cos_dim = min(z_img.size(1), z_txt.size(1))
                if cos_dim > 0:
                    cos_z = F.cosine_similarity(z_img[:, :cos_dim], z_txt[:, :cos_dim], dim=1)  # (B,)
                    cos_z_list.append(cos_z.cpu())

                zeros_txt = torch.zeros_like(z_txt).to(self.device)
                zeros_img = torch.zeros_like(z_img).to(self.device)

                joint_h_img = self.joint_rbm.forward(torch.cat([z_img, zeros_txt], dim=1))  # (B, H)
                joint_h_txt = self.joint_rbm.forward(torch.cat([zeros_img, z_txt], dim=1))

                joint_h_img_n = F.normalize(joint_h_img, dim=1)
                joint_h_txt_n = F.normalize(joint_h_txt, dim=1)

                cos_joint = F.cosine_similarity(joint_h_img_n, joint_h_txt_n, dim=1)  # (B,)
                cos_joint_list.append(cos_joint.cpu())

                # stats
                z_img_means.append(z_img.mean().item())
                z_txt_means.append(z_txt.mean().item())
                joint_img_norms.append(joint_h_img.norm(dim=1).mean().item())
                joint_txt_norms.append(joint_h_txt.norm(dim=1).mean().item())

        # aggregate
        if cos_z_list:
            cos_z_all = torch.cat(cos_z_list)
            writer.add_scalar("Diag/cosine_z_mean", float(cos_z_all.mean()), epoch)
            writer.add_histogram("Diag/cosine_z_hist", cos_z_all.numpy(), epoch)

        if cos_joint_list:
            cos_joint_all = torch.cat(cos_joint_list)
            writer.add_scalar("Diag/cosine_joint_mean", float(cos_joint_all.mean()), epoch)
            writer.add_histogram("Diag/cosine_joint_hist", cos_joint_all.numpy(), epoch)

        # norms / means
        if z_img_means:
            writer.add_scalar("Diag/z_img_mean", float(sum(z_img_means)/len(z_img_means)), epoch)
        if z_txt_means:
            writer.add_scalar("Diag/z_txt_mean", float(sum(z_txt_means)/len(z_txt_means)), epoch)
        if joint_img_norms:
            writer.add_scalar("Diag/joint_img_norm", float(sum(joint_img_norms)/len(joint_img_norms)), epoch)
        if joint_txt_norms:
            writer.add_scalar("Diag/joint_txt_norm", float(sum(joint_txt_norms)/len(joint_txt_norms)), epoch)

        # Print quick summary to console
        print(f"[Diag] epoch {epoch} | cos_z_mean {cos_z_all.mean():.4f} | cos_joint_mean {cos_joint_all.mean():.4f}")

def plot_bar(values):
    fig, ax = plt.subplots()
    ax.bar(range(len(values)), values)
    ax.set_ylim(0, 1)
    return fig
