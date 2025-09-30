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
from src.utils.wandb_utils import plot_2d_embedding_and_correlations
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
from src.utils.probe_utils import log_linear_probe  # <— qui l'orchestratore



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
            self.labels = val_labels
            self.cumArea_vals = [val_loader.dataset.dataset.cumArea_list[i] for i in indices]
            self.convex_hull = [val_loader.dataset.dataset.CH_list[i] for i in indices]

            self.features = {
                "Cumulative Area": torch.tensor(self.cumArea_vals, dtype=torch.float),
                "Convex Hull": torch.tensor(self.convex_hull, dtype=torch.float),
                "Labels": self.labels,
            }
        else:
            self.validation_images = None
            self.features = None

        self.arch_str = '-'.join(str(size) for size in layer_sizes)
        self.arch_dir = os.path.join(log_root, f"architecture_{self.arch_str}")
        os.makedirs(self.arch_dir, exist_ok=True)

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

            # PCA logging (se usi già plot_2d_embedding_and_correlations)
            if self.wandb_run and self.features and epoch % log_every_pca == 0:
                try:
                    with torch.no_grad():
                        final_embedding = self.represent(self.val_batch).detach().cpu()
                    if final_embedding.shape[0] > 1 and final_embedding.shape[1] > 2:
                        n_neighbors_umap = min(final_embedding.shape[0] - 1, 15)
                        Umap = umap.UMAP(n_components=2, random_state=42, n_neighbors=n_neighbors_umap)
                        pca = PCA(n_components=2)
                        emb_2d_pca = pca.fit_transform(final_embedding)
                        plot_2d_embedding_and_correlations(
                            emb_2d=emb_2d_pca,
                            features=self.features,
                            arch_name=self.arch_str,
                            dist_name="validation",
                            method_name="pca",
                            wandb_run=self.wandb_run,
                        )
                except Exception as e:
                    if self.wandb_run:
                        self.wandb_run.log({"warn/pca_logging_error": str(e)})

            # 🔔 Linear probe logging via utils
            if self.wandb_run and self.features and self.val_loader is not None and epoch % log_every_probe == 0:
                try:
                    log_linear_probe(self, epoch=epoch, n_bins=5, test_size=0.2, steps=1000, lr=1e-2, patience=20, min_delta=0.0)
                except Exception as e:
                    if self.wandb_run:
                        self.wandb_run.log({"warn/probe_logging_error": str(e)})

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
    def represent(self, x: torch.Tensor) -> torch.Tensor:
        v = x.view(x.size(0), -1).float().to(self.device)
        for rbm in self.layers:
            v = rbm.forward(v)
        return v

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
class iMDBN(nn.Module):
    def __init__(self, layer_sizes_img, layer_sizes_txt, joint_layer_size, params,
                 dataloader, val_loader, device, log_root="logs-imdbn",num_labels =32, embedding_dim=64):
        super(iMDBN, self).__init__()
        self.params = params
        self.device = device
        self.dataloader = dataloader

        self.num_labels = num_labels
        self.embedding_dim = embedding_dim

        # Learnable embeddings per label (dense vector per numero di dots)
        self.label_embeddings = nn.Embedding(num_labels, embedding_dim).to(device)


        # Validation batch (images + labels)
        val_batch, val_labels = next(iter(val_loader))
        self.validation_images = val_batch[:5].to(self.device)
        self.validation_labels = val_labels[:5].to(self.device)

        self.arch_str = f"IMG{'-'.join(map(str,layer_sizes_img))}_TXT{'-'.join(map(str,layer_sizes_txt))}_JOINT{joint_layer_size}"
        self.arch_dir = os.path.join(log_root, f"architecture_{self.arch_str}")
        os.makedirs(self.arch_dir, exist_ok=True)

        # Unimodal iDBNs (these manage their own RBMs)
        self.image_idbn = iDBN(layer_sizes=layer_sizes_img, params=params,
                               dataloader=self.dataloader, val_loader=val_loader,
                               device=device, log_root=log_root, text_flag=False)

        self.text_idbn = iDBN(layer_sizes=[self.num_labels] + layer_sizes_txt[1:], params=params,
                              dataloader=self.dataloader, val_loader=val_loader,
                              device=device, log_root=log_root, text_flag=True)

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

        # supervised head from joint hidden -> predict N (regression)
        self.supervised_head = nn.Linear(joint_layer_size, 1).to(self.device)
        self.sup_optimizer = torch.optim.Adam(self.supervised_head.parameters(), lr=1e-3)

    # -----------------
    def forward(self, img_data, labels):
        # returns joint hidden probabilities (no grad)
        with torch.no_grad():
            img_rep = self.image_idbn.represent(img_data.to(self.device))
            txt_rep = self.text_idbn.represent(labels.to(self.device))
            joint_input = torch.cat((img_rep, txt_rep), dim=1)
            joint_h = self.joint_rbm.forward(joint_input)
        return joint_h

    # -----------------
    def _cross_reconstruct(self, z_img, z_txt, gibbs_steps=20):
        """
        Return:
        recon_img_from_txt, recon_txt_from_img
        z_img: (B, Dz_img)  top-level image rep (probabilities)
        z_txt: (B, Dz_txt)  top-level text rep (probabilities)
        """
        with torch.no_grad():
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
            recon_txt_from_img = v_txt_top_from_img
            for rbm in reversed(self.text_idbn.layers):
                recon_txt_from_img = rbm.backward(recon_txt_from_img)

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
    def train_joint(self, epochs, run_id=0, w_rec=1.0, w_sup=1.0):
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
            for img_data, labels in self.dataloader:
                img_data = img_data.to(self.device).view(img_data.size(0), -1).float()
                labels = labels.to(self.device).float()  # one-hot or similar
                #label_indices = labels.to(self.device).long()  # assume 0..num_labels-1
                #labels_dense = self.label_embeddings(label_indices)  # (B, embedding_dim)
                z_txt = self.text_idbn.represent(labels)

                # get top-level reps (freeze unimodals)
                with torch.no_grad():
                    z_img = self.image_idbn.represent(img_data)
                    z_txt = self.text_idbn.represent(labels)

                joint_input = torch.cat((z_img, z_txt), dim=1).to(self.device)

                mask_p = self.params.get("JOINT_MASK_P", 0.3)

                if random.random() < mask_p:
                    joint_input = joint_input.clone()
                    joint_input[:, z_img:] = 0.0  # mask text
                elif random.random() < mask_p:
                    joint_input = joint_input.clone()
                    joint_input[:, :z_img] = 0.0  # mask image
                else:
                    joint_input = joint_input


                # 1) Train joint RBM via CD (updates internal weights)
                batch_loss_cd = self.joint_rbm.train_epoch(joint_input, epoch, epochs, CD=cd_k)
                epoch_losses.append(float(batch_loss_cd.item() if isinstance(batch_loss_cd, torch.Tensor) else batch_loss_cd))

                # 2) Cross reconstruction losses (differentiable parts: supervised head; we use recon outputs as targets)
                # Compute reconstructions from single modality (no grad for RBM ops here)
                recon_img_from_txt, recon_txt_from_img = self._cross_reconstruct(z_img, z_txt,gibbs_steps=10)

                # image reconstruction loss: BCE between recon_img_from_txt (prob) and img_data (0/1)
                # ensure shapes
                recon_img_from_txt = recon_img_from_txt.view(img_data.size(0), -1)
                L_rec_img = F.binary_cross_entropy(recon_img_from_txt.clamp(1e-6, 1-1e-6), img_data.clamp(0.0, 1.0), reduction='mean')

                # text reconstruction loss: if labels are one-hot, use MSE or BCE
                recon_txt_from_img = recon_txt_from_img.view(labels.size(0), -1)
                L_rec_txt = F.mse_loss(recon_txt_from_img, labels, reduction='mean')

                L_rec = (L_rec_img + L_rec_txt) * w_rec
                rec_losses.append(float(L_rec.item()))

                # 3) supervised head: predict N from joint hidden (we compute joint hidden probs now, allow grads on head)
                joint_h = self.joint_rbm.forward(joint_input)  # probabilities (no grad inside RBM but returns tensor)
                # enable grad for head training
                pred_N = self.supervised_head(joint_h)  # (B,1)
                # we need target numeric N: assume labels include N as integer in last column? 
                # **IMPORTANT**: your dataloader must provide numeric N as separate tensor or encode it in labels.
                # For now, we assume labels include N in a tensor attr (you must adapt to your dataloader).
                # Here I try to read labels[:, -1] if present (fall back to zeros)
                try:
                    N_target = labels[:, -1].unsqueeze(1)  # if dataloader packs N there
                except Exception:
                    N_target = torch.zeros_like(pred_N).to(self.device)

                L_sup = F.mse_loss(pred_N, N_target, reduction='mean') * w_sup
                sup_losses.append(float(L_sup.item()))

                # Backprop supervised head + (optionally) small finetune on text_idbn/ image_idbn top layers:
                self.sup_optimizer.zero_grad()
                L_sup.backward()
                self.sup_optimizer.step()

            # epoch logging
            mean_cd = sum(epoch_losses) / len(epoch_losses) if epoch_losses else 0.0
            mean_rec = sum(rec_losses) / len(rec_losses) if rec_losses else 0.0
            mean_sup = sum(sup_losses) / len(sup_losses) if sup_losses else 0.0
            writer.add_scalar("JointRBM/CD_loss", mean_cd, epoch)
            writer.add_scalar("JointRBM/rec_loss", mean_rec, epoch)
            writer.add_scalar("JointRBM/sup_loss", mean_sup, epoch)
            #self.log_joint_cosine_similarity(writer, epoch)
            self.log_joint_diagnostics(writer, epoch, num_batches=5)



            if epoch % 10 == 0:
                print(f"[Joint RBM] Epoch {epoch}, CD_loss={mean_cd:.4f}, rec={mean_rec:.4f}, sup={mean_sup:.4f}")
            if epoch % 5 == 0:
                self.log_cross_modality(writer, epoch, num_samples=5)


            torch.cuda.empty_cache()

        writer.close()

    # -----------------
    def save_model(self, path):
        model_copy = copy.deepcopy(self)
        model_copy.dataloader = None
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
            z_txt = self.text_idbn.represent(lbl_data)
            recon_img_from_txt, recon_txt_from_img = self._cross_reconstruct(z_img, z_txt,gibbs_steps=20)

        # ---- immagini ----
        recon_img_from_txt_norm = recon_img_from_txt.clamp(0, 1)
        img_grid = recon_img_from_txt_norm.view(-1, 1,
                        int(recon_img_from_txt_norm.size(1)**0.5),
                        int(recon_img_from_txt_norm.size(1)**0.5))
        writer.add_images('CrossModality/Image_from_Label', img_grid, epoch)
        
        # ---- labels ----
        recon_probs = recon_txt_from_img.clamp(0, 1)

        pred_indices = recon_probs.argmax(dim=1)  # predizione numerica

        rows = []
        for i in range(min(num_samples, recon_probs.size(0))):
            gt = lbl_data[i].argmax().item()
            pred = pred_indices[i].item()
            rows.append(f"Sample {i}: Pred = {pred}, GT = {gt}")

        table_str = " | Sample | Prediction | GroundTruth |\n"
        table_str += "|--------|------------|-------------|\n"
        for i, row in enumerate(rows):
            gt = lbl_data[i].argmax().item()
            pred = pred_indices[i].item()
            table_str += f"| {i} | {pred} | {gt} |\n"

        writer.add_text("CrossModality/Predictions", table_str, epoch)



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
                z_img = self.image_idbn.represent(img_data)
                z_txt = self.text_idbn.represent(labels)

                # Joint hidden reps
                joint_h_img = self.joint_rbm.forward(torch.cat([z_img, torch.zeros_like(z_txt)], dim=1))
                joint_h_txt = self.joint_rbm.forward(torch.cat([torch.zeros_like(z_img), z_txt], dim=1))

                # Cosine similarity per batch
                sim = F.cosine_similarity(joint_h_img, joint_h_txt, dim=1)  # shape: (B,)
                cos_sims.append(sim.mean().item())

        mean_cos = sum(cos_sims) / len(cos_sims)
        writer.add_scalar("JointRBM/cosine_similarity_img_txt", mean_cos, epoch)
        print(f"[Epoch {epoch}] Cosine similarity (img vs txt in joint space): {mean_cos:.4f}")

    import torch.nn.functional as F

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
                z_img = self.image_idbn.represent(imgs)   # (B, D_img)
                z_txt = self.text_idbn.represent(labs)   # (B, D_txt)

                # Normalize for stable cosine computation
                z_img_n = F.normalize(z_img, dim=1)
                z_txt_n = F.normalize(z_txt, dim=1)

                # cosine between z_img and z_txt (per sample)
                cos_z = F.cosine_similarity(z_img_n, z_txt_n, dim=1)  # (B,)
                cos_z_list.append(cos_z.cpu())

                # Joint hidden when feeding only image / only text
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

