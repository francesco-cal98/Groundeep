import math
import os
import pickle
from typing import List, Tuple, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.utils as vutils
import wandb
from tqdm import tqdm
from sklearn.decomposition import PCA

# === tue utility già presenti ===
from src.utils.wandb_utils import (
    plot_2d_embedding_and_correlations,
    plot_3d_embedding_and_correlations,
)
from src.utils.probe_utils import (
    log_linear_probe,
    log_joint_linear_probe,
    compute_val_embeddings_and_features,
    compute_joint_embeddings_and_features,
)
#from src.utils.energy_utils import pick_val_case, trace_single_img2txt, log_single_case_energy
from src.utils.energy_utils import run_and_log_fixed_case
from src.utils.conditional_steps import run_and_log_cross_fixed_case, run_and_log_cross_panel,run_and_log_z_mismatch_check


# -------------------------
# Helpers
# -------------------------
def sigmoid(x: torch.Tensor) -> torch.Tensor:
    return 1 / (1 + torch.exp(-x))


# -------------------------
# RBM (Bernoulli vis/hidden + gruppi softmax opzionali)
# -------------------------
class RBM(nn.Module):
    def __init__(
        self,
        num_visible: int,
        num_hidden: int,
        learning_rate: float,
        weight_decay: float,
        momentum: float,
        dynamic_lr: bool = False,
        final_momentum: float = 0.97,
        sparsity: bool = False,
        sparsity_factor: float = 0.05,
        softmax_groups: Optional[List[Tuple[int, int]]] = None,
    ):
        super().__init__()
        self.num_visible = int(num_visible)
        self.num_hidden = int(num_hidden)
        self.lr = float(learning_rate)
        self.weight_decay = float(weight_decay)
        self.momentum = float(momentum)
        self.dynamic_lr = bool(dynamic_lr)
        self.final_momentum = float(final_momentum)
        self.sparsity = bool(sparsity)
        self.sparsity_factor = float(sparsity_factor)

        # compat vecchi pickle
        self.softmax_groups = softmax_groups or []

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.W = nn.Parameter(
            torch.randn(self.num_visible, self.num_hidden, device=device) / math.sqrt(max(1, self.num_visible))
        )
        self.hid_bias = nn.Parameter(torch.zeros(self.num_hidden, device=device))
        self.vis_bias = nn.Parameter(torch.zeros(self.num_visible, device=device))

        # momentum buffers
        self.W_m = torch.zeros_like(self.W)
        self.hb_m = torch.zeros_like(self.hid_bias)
        self.vb_m = torch.zeros_like(self.vis_bias)

    # ---- p(h|v)
    def forward(self, v: torch.Tensor) -> torch.Tensor:
        return sigmoid(v @ self.W + self.hid_bias)

    # ---- logit(v|h)
    def _visible_logits(self, h: torch.Tensor) -> torch.Tensor:
        return h @ self.W.T + self.vis_bias

    # ---- p(v|h) con gruppi softmax (labels)
    def visible_probs(self, h: torch.Tensor) -> torch.Tensor:
        logits = self._visible_logits(h)
        v_prob = torch.sigmoid(logits)  # default Bernoulli
        groups = getattr(self, "softmax_groups", [])
        for s, e in groups:
            v_prob[:, s:e] = torch.softmax(logits[:, s:e], dim=1)
        return v_prob

    # ---- sample v ~ p(v|h)
    def sample_visible(self, v_prob: torch.Tensor) -> torch.Tensor:
        v = (v_prob > torch.rand_like(v_prob)).float()
        groups = getattr(self, "softmax_groups", [])
        for s, e in groups:
            probs = v_prob[:, s:e].clamp(1e-8, 1)
            idx = torch.distributions.Categorical(probs=probs).sample()
            v[:, s:e] = 0.0
            v[torch.arange(v.size(0), device=v.device), s + idx] = 1.0
        return v

    # ---- decoder compatibile
    def backward(self, h: torch.Tensor, return_logits: bool = False) -> torch.Tensor:
        logits = self._visible_logits(h)
        if return_logits:
            return logits
        return self.visible_probs(h)

    @torch.no_grad()
    def backward_sample(self, h: torch.Tensor) -> torch.Tensor:
        return self.sample_visible(self.visible_probs(h))

    # ---- single Gibbs
    @torch.no_grad()
    def gibbs_step(self, v: torch.Tensor, sample_h: bool = True, sample_v: bool = True):
        h_prob = self.forward(v)
        h = (h_prob > torch.rand_like(h_prob)).float() if sample_h else h_prob
        v_prob = self.visible_probs(h)
        v_next = self.sample_visible(v_prob) if sample_v else v_prob
        return v_next, v_prob, h, h_prob

    # ---- CD-k
    @torch.no_grad()
    def train_epoch(self, data: torch.Tensor, epoch: int, max_epochs: int, CD: int = 1):
        lr = self.lr / (1 + 0.01 * epoch) if self.dynamic_lr else self.lr
        mom = self.momentum if epoch <= 5 else self.final_momentum
        bsz = data.size(0)

        # positive
        pos_h = self.forward(data)
        pos_assoc = data.T @ pos_h

        # negative
        h = (pos_h > torch.rand_like(pos_h)).float()
        for _ in range(int(CD)):
            v_prob = self.visible_probs(h)
            v = self.sample_visible(v_prob)
            h_prob = self.forward(v)
            h = (h_prob > torch.rand_like(h_prob)).float()
        neg_assoc = v.T @ h_prob

        # updates
        self.W_m.mul_(mom).add_(lr * ((pos_assoc - neg_assoc) / bsz - self.weight_decay * self.W))
        self.W.add_(self.W_m)

        self.hb_m.mul_(mom).add_(lr * (pos_h.sum(0) - h_prob.sum(0)) / bsz)
        if self.sparsity:
            Q = pos_h.mean(0)
            self.hb_m.add_(-lr * (Q - self.sparsity_factor))
        self.hid_bias.add_(self.hb_m)

        self.vb_m.mul_(mom).add_(lr * (data.sum(0) - v.sum(0)) / bsz)
        self.vis_bias.add_(self.vb_m)

        loss = torch.mean((data - v_prob) ** 2)
        return loss

    # ---- Gibbs condizionale (re-clamp ad ogni step)
    @torch.no_grad()
    def conditional_gibbs(
        self,
        v_known: torch.Tensor,
        known_mask: torch.Tensor,
        n_steps: int = 30,
        sample_h: bool = False,
        sample_v: bool = False,
    ) -> torch.Tensor:
        km = known_mask
        v = v_known * km + (1 - km) * torch.rand_like(v_known)
        for _ in range(int(n_steps)):
            h_prob = self.forward(v)
            h = (h_prob > torch.rand_like(h_prob)).float() if sample_h else h_prob
            v_prob = self.visible_probs(h)
            v = v_prob * (1 - km) + v_known * km
            if sample_v:
                v = self.sample_visible(v) * (1 - km) + v_known * km
        return self.visible_probs(self.forward(v))

    # ---- clamped-CD (aux)
    @torch.no_grad()
    def train_epoch_clamped(
        self,
        v_known: torch.Tensor,
        known_mask: torch.Tensor,
        epoch: int,
        max_epochs: int,
        CD: int = 1,
        cond_init_steps: int = 50,
        sample_h: bool = True,
        sample_v: bool = False,
    ):
        lr = self.lr / (1 + 0.01 * epoch) if self.dynamic_lr else self.lr
        mom = self.momentum if epoch <= 5 else self.final_momentum
        bsz = v_known.size(0)

        v_plus = self.conditional_gibbs(
            v_known, known_mask, n_steps=cond_init_steps, sample_h=sample_h, sample_v=sample_v
        )
        h_plus = self.forward(v_plus)
        pos_assoc = v_plus.T @ h_plus

        v_neg = v_plus.clone()
        for _ in range(int(CD)):
            h_prob = self.forward(v_neg)
            h = (h_prob > torch.rand_like(h_prob)).float() if sample_h else h_prob
            v_prob = self.visible_probs(h)
            v_neg = v_prob * (1 - known_mask) + v_known * known_mask
            if sample_v:
                v_neg = self.sample_visible(v_neg) * (1 - known_mask) + v_known * known_mask

        h_neg = self.forward(v_neg)
        neg_assoc = v_neg.T @ h_neg

        self.W_m.mul_(mom).add_(lr * ((pos_assoc - neg_assoc) / bsz - self.weight_decay * self.W))
        self.W.add_(self.W_m)
        self.hb_m.mul_(mom).add_(lr * (h_plus.sum(0) - h_neg.sum(0)) / bsz)
        self.hid_bias.add_(self.hb_m)
        self.vb_m.mul_(mom).add_(lr * (v_plus.sum(0) - v_neg.sum(0)) / bsz)
        self.vis_bias.add_(self.vb_m)

        return torch.mean((v_plus - v_neg) ** 2)


# -------------------------
# iDBN (immagini) — con PCA/Probes/AutoRecon
# -------------------------
class iDBN:
    def __init__(self, layer_sizes, params, dataloader, val_loader, device, wandb_run=None):
        self.layers: List[RBM] = []
        self.params = params
        self.dataloader = dataloader
        self.val_loader = val_loader
        self.device = device
        self.wandb_run = wandb_run

        # campi attesi dalle tue utils
        self.text_flag = False
        self.arch_str = "-".join(map(str, layer_sizes))
        self.arch_dir = os.path.join("logs-idbn", f"architecture_{self.arch_str}")
        os.makedirs(self.arch_dir, exist_ok=True)

        self.cd_k = int(self.params.get("CD", 1))
        self.sparsity_last = bool(self.params.get("SPARSITY", False))
        self.sparsity_factor = float(self.params.get("SPARSITY_FACTOR", 0.1))

        # cache val
        try:
            self.val_batch, self.val_labels = next(iter(val_loader))
        except Exception:
            self.val_batch, self.val_labels = None, None

        # features complete (no shuffle sul val_loader!)
        self.features = None
        try:
            indices = val_loader.dataset.indices
            base = val_loader.dataset.dataset
            numeric_labels = torch.tensor([base.labels[i] for i in indices], dtype=torch.float32)
            cumArea_vals = [base.cumArea_list[i] for i in indices]
            convex_hull = [base.CH_list[i] for i in indices]
            density_src = getattr(base, "density_list", None)
            density_vals = [density_src[i] for i in indices] if density_src is not None else None
            self.features = {
                "Cumulative Area": torch.tensor(cumArea_vals, dtype=torch.float32),
                "Convex Hull": torch.tensor(convex_hull, dtype=torch.float32),
                "Labels": numeric_labels,
            }
            if density_vals is not None:
                self.features["Density"] = torch.tensor(density_vals, dtype=torch.float32)
        except Exception:
            pass

        # costruzione RBM
        for i in range(len(layer_sizes) - 1):
            rbm = RBM(
                num_visible=layer_sizes[i],
                num_hidden=layer_sizes[i + 1],
                learning_rate=self.params["LEARNING_RATE"],
                weight_decay=self.params["WEIGHT_PENALTY"],
                momentum=self.params["INIT_MOMENTUM"],
                dynamic_lr=self.params["LEARNING_RATE_DYNAMIC"],
                final_momentum=self.params["FINAL_MOMENTUM"],
                sparsity=(self.sparsity_last and i == len(layer_sizes) - 2),
                sparsity_factor=self.sparsity_factor,
            ).to(self.device)
            self.layers.append(rbm)

    # quali layer monitorare (come nei tuoi log)
    def _layers_to_monitor(self) -> List[int]:
        layers = {len(self.layers)}
        if len(self.layers) > 1:
            layers.add(1)
        return sorted(layers)

    def _layer_tag(self, idx: int) -> str:
        return f"layer{idx}"

    # TRAIN con autorecon + PCA + probes
    def train(self, epochs: int, log_every_pca: int = 25, log_every_probe: int = 10):
        for epoch in tqdm(range(int(epochs)), desc="iDBN"):
            losses = []
            for img, _ in self.dataloader:
                v = img.to(self.device).view(img.size(0), -1).float()
                for rbm in self.layers:
                    loss = rbm.train_epoch(v, epoch, epochs, CD=self.cd_k)
                    v = rbm.forward(v)
                    losses.append(float(loss))
            if self.wandb_run and losses:
                self.wandb_run.log({"idbn/loss": float(np.mean(losses)), "epoch": epoch})

            # Auto-recon snapshot
            if self.wandb_run and self.val_batch is not None and epoch % 5 == 0:
                with torch.no_grad():
                    rec = self.reconstruct(self.val_batch[:8].to(self.device))
                img0 = self.val_batch[:8]
                try:
                    B, C, H, W = img0.shape
                    recv = rec.view(B, C, H, W).clamp(0, 1)
                except Exception:
                    side = int(rec.size(1) ** 0.5)
                    C = 1
                    H = W = side
                    recv = rec.view(-1, C, H, W).clamp(0, 1)
                    img0 = img0.view(-1, C, H, W)
                grid = vutils.make_grid(torch.cat([img0.cpu(), recv.cpu()], dim=0), nrow=img0.size(0))
                self.wandb_run.log({"idbn/auto_recon_grid": wandb.Image(grid.permute(1, 2, 0).numpy()), "epoch": epoch})
                try:
                    mse = F.mse_loss(img0.to(self.device).view(img0.size(0), -1), recv.view(img0.size(0), -1))
                    self.wandb_run.log({"idbn/auto_recon_mse": mse.item(), "epoch": epoch})
                except Exception:
                    pass

            # PCA + PROBES per-layer
            if self.wandb_run and self.val_loader is not None and self.features is not None:
                if epoch % log_every_pca == 0:
                    for layer_idx in self._layers_to_monitor():
                        tag = self._layer_tag(layer_idx)
                        try:
                            E, feats = compute_val_embeddings_and_features(self, upto_layer=layer_idx)
                            if E.numel() == 0:
                                continue
                            emb_np = E.numpy()
                            feat_map = {
                                "Cumulative Area": feats["cum_area"].numpy(),
                                "Convex Hull": feats["convex_hull"].numpy(),
                                "Labels": feats["labels"].numpy(),
                            }
                            if "density" in feats:
                                feat_map["Density"] = feats["density"].numpy()
                            if emb_np.shape[0] > 2 and emb_np.shape[1] > 2:
                                p2 = PCA(n_components=2).fit_transform(emb_np)
                                plot_2d_embedding_and_correlations(
                                    emb_2d=p2,
                                    features=feat_map,
                                    arch_name=f"iDBN_{tag}",
                                    dist_name="val",
                                    method_name="pca",
                                    wandb_run=self.wandb_run,
                                )
                                if emb_np.shape[1] >= 3:
                                    p3 = PCA(n_components=3).fit_transform(emb_np)
                                    plot_3d_embedding_and_correlations(
                                        emb_3d=p3,
                                        features=feat_map,
                                        arch_name=f"iDBN_{tag}",
                                        dist_name="val",
                                        method_name="pca",
                                        wandb_run=self.wandb_run,
                                    )
                        except Exception as e:
                            self.wandb_run.log({f"warn/idbn_pca_error_{tag}": str(e)})

                if epoch % log_every_probe == 0:
                    for layer_idx in self._layers_to_monitor():
                        tag = self._layer_tag(layer_idx)
                        try:
                            log_linear_probe(
                                self,
                                epoch=epoch,
                                n_bins=5,
                                test_size=0.2,
                                steps=1000,
                                lr=1e-2,
                                patience=20,
                                min_delta=0.0,
                                upto_layer=layer_idx,
                                layer_tag=tag,
                            )
                        except Exception as e:
                            self.wandb_run.log({f"warn/idbn_probe_error_{tag}": str(e)})

    @torch.no_grad()
    def represent(self, x: torch.Tensor, upto_layer: Optional[int] = None) -> torch.Tensor:
        v = x.view(x.size(0), -1).float().to(self.device)
        L = len(self.layers) if (upto_layer is None) else max(0, min(len(self.layers), int(upto_layer)))
        for i in range(L):
            v = self.layers[i].forward(v)
        return v

    @torch.no_grad()
    def reconstruct(self, x: torch.Tensor) -> torch.Tensor:
        v = x.view(x.size(0), -1).float().to(self.device)
        cur = v
        for rbm in self.layers:
            cur = rbm.forward(cur)
        for rbm in reversed(self.layers):
            cur = rbm.backward(cur)
        return cur

    def decode(self, top: torch.Tensor) -> torch.Tensor:
        cur = top.to(self.device)
        for rbm in reversed(self.layers):
            cur = rbm.backward(cur)
        return cur

    def save_model(self, path: str):
        model_copy = {"layers": self.layers, "params": self.params}
        with open(path, "wb") as f:
            pickle.dump(model_copy, f)


# -------------------------
# iMDBN (multimodale) — semplice, etichette come softmax block
# -------------------------
class iMDBN(nn.Module):
    """
    Joint RBM su [z_img  ⊕  y_onehot] con y gestito come gruppo softmax.
    Supporta:
      - forma lunga: iMDBN(image_layers, text_layers, joint_hidden, params, ...)
      - forma corta: iMDBN(image_layers, joint_hidden, params, ...)
    """
    def __init__(
        self,
        layer_sizes_img,
        layer_sizes_txt_or_joint=None,          # può essere list/tuple (vecchia API) o int (nuova API)
        joint_layer_size=None,
        params=None,
        dataloader=None,
        val_loader=None,
        device=None,
        text_posenc_dim: int = 0,               # ignorato (no TextRBM)
        num_labels: int = 32,
        embedding_dim: int = 64,
        wandb_run=None,
    ):
        super().__init__()

        # disambiguazione firma
        if isinstance(layer_sizes_txt_or_joint, (list, tuple)):
            # forma lunga
            layer_sizes_txt_unused = layer_sizes_txt_or_joint
            if joint_layer_size is None:
                raise ValueError("joint_layer_size mancante nella forma lunga.")
        else:
            # forma corta
            if joint_layer_size is None:
                joint_layer_size = int(layer_sizes_txt_or_joint)
            layer_sizes_txt_unused = None

        self.params = params or {}
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.dataloader = dataloader
        self.val_loader = val_loader
        self.wandb_run = wandb_run
        self.num_labels = int(num_labels)

        # snapshot val
        try:
            vb_imgs, vb_lbls = next(iter(val_loader))
            self.validation_images = vb_imgs[:8].to(self.device)
            self.validation_labels = vb_lbls[:8].to(self.device)
            self.val_batch = (vb_imgs, vb_lbls)
        except Exception:
            self.validation_images = None
            self.validation_labels = None
            self.val_batch = None

        # iDBN immagine
        self.image_idbn = iDBN(
            layer_sizes=layer_sizes_img,
            params=self.params,
            dataloader=self.dataloader,
            val_loader=self.val_loader,
            device=self.device,
            wandb_run=self.wandb_run,
        )

        # dimensione top immagine + joint
        dz_from_img = int(self.image_idbn.layers[-1].num_hidden)
        self.Dz_img = dz_from_img
        self._build_joint(Dz_img=dz_from_img, joint_hidden=joint_layer_size)

        # knobs
        self.joint_cd = int(self.params.get("JOINT_CD", self.params.get("CD", 1)))
        self.cross_steps = int(self.params.get("CROSS_GIBBS_STEPS", 50))
        self.aux_every_k = int(self.params.get("JOINT_AUX_EVERY_K", 0))
        self.aux_cond_steps = int(self.params.get("JOINT_AUX_COND_STEPS", 50))

        # features joint complete (no shuffle!)
        self.features = None
        try:
            indices = val_loader.dataset.indices
            base = val_loader.dataset.dataset
            numeric_labels = torch.tensor([base.labels[i] for i in indices], dtype=torch.float32)
            cumArea_vals = [base.cumArea_list[i] for i in indices]
            convex_hull = [base.CH_list[i] for i in indices]
            density_src = getattr(base, "density_list", None)
            density_vals = [density_src[i] for i in indices] if density_src is not None else None
            self.features = {
                "Cumulative Area": torch.tensor(cumArea_vals, dtype=torch.float32),
                "Convex Hull": torch.tensor(convex_hull, dtype=torch.float32),
                "Labels": numeric_labels,
            }
            if density_vals is not None:
                self.features["Density"] = torch.tensor(density_vals, dtype=torch.float32)
        except Exception:
            pass

        self.arch_str = f"IMG{'-'.join(map(str,layer_sizes_img))}_JOINT{joint_layer_size}"

    def _build_joint(self, Dz_img: int, joint_hidden: int):
        self.Dz_img = int(Dz_img)  # sempre presente
        K = self.num_labels
        self.joint_rbm = RBM(
            num_visible=self.Dz_img + K,
            num_hidden=int(joint_hidden),
            learning_rate=self.params.get("JOINT_LEARNING_RATE", self.params.get("LEARNING_RATE", 0.1)),
            weight_decay=self.params.get("WEIGHT_PENALTY", 0.0001),
            momentum=self.params.get("INIT_MOMENTUM", 0.5),
            dynamic_lr=self.params.get("LEARNING_RATE_DYNAMIC", True),
            final_momentum=self.params.get("FINAL_MOMENTUM", 0.95),
            softmax_groups=[(self.Dz_img, self.Dz_img + K)],
        ).to(self.device)

    @torch.no_grad()
    def log_latent_trajectory_with_recon_panel(
    self,
    sample_idx: int = 0,
    steps: int = 40,
    tag: str = "pca_traj_with_recon",
    n_frames: int = 8,          # quante tappe mostrare nel pannello (oltre alla GT)
    scatter_size: int = 12,     # punti del cloud
    scatter_alpha: float = 0.35 # trasparenza del cloud
):
        """
        - Fit PCA(2) su TUTTO il validation z_img (come nei tuoi log).
        - Plotta il cloud (colorato con Numerosity/N_list o fallback Labels) + traiettoria TXT->IMG del sample.
        - A destra: pannello con GT + ricostruzioni a tappe della catena.
        """
        import numpy as np
        import matplotlib.pyplot as plt
        from sklearn.decomposition import PCA
        import torch.nn.functional as F

        from src.utils.probe_utils import compute_val_embeddings_and_features

        assert self.val_loader is not None, "val_loader mancante"

        # --- 1) Embedding val top (z_img) via util esistente
        Z_val_t, feats = compute_val_embeddings_and_features(
            self.image_idbn, upto_layer=len(self.image_idbn.layers)
        )
        if Z_val_t is None or Z_val_t.numel() == 0:
            if self.wandb_run: self.wandb_run.log({f"{tag}/warn": "no val embeddings"})
            return

        Z_val = Z_val_t.detach().cpu().numpy()   # [N_val, Dz]
        N_val, Dz = Z_val.shape
        sample_idx = int(max(0, min(sample_idx, N_val - 1)))

        # --- 2) Vettore colori come nei tuoi plot: N_list (Numerosity) se c'è, altrimenti Labels
        color_vec = None
        try:
            base = self.val_loader.dataset.dataset
            indices = self.val_loader.dataset.indices
            if hasattr(base, "N_list"):
                color_vec = np.array([base.N_list[i] for i in indices], dtype=float)
        except Exception:
            pass
        if color_vec is None:
            # fallback: usa le labels dal dict feats (già allineate)
            if "labels" in feats:
                color_vec = feats["labels"].numpy()
            else:
                color_vec = np.zeros(Z_val.shape[0], dtype=float)

        # --- 3) PCA(2) fit su TUTTO il val (coerente con i log)
        pca = PCA(n_components=2)
        Z2 = pca.fit_transform(Z_val)                 # [N_val, 2]
        z_true_2d = Z2[sample_idx:sample_idx+1]       # [1, 2]

        # --- 4) Recupera sample esatto (x_i, y_i) dal val_loader (no shuffle)
        seen = 0
        x_i = None; y_i = None
        for imgs, lbls in self.val_loader:
            b = imgs.size(0)
            if seen + b <= sample_idx:
                seen += b
                continue
            pos = sample_idx - seen
            x_i = imgs[pos:pos+1].to(self.device).view(1, -1).float()  # [1, Npix]
            y_i = lbls[pos:pos+1].to(self.device).float()               # [1, K]
            break
        if x_i is None:
            if self.wandb_run: self.wandb_run.log({f"{tag}/warn": "sample not found"})
            return
        gt_class = int(y_i.argmax(dim=1).item())

        # --- 5) TXT -> IMG mean-field condizionale, salva traiettoria (in PCA) e ricostruzioni immagini
        K = self.num_labels
        V = Dz + K
        v_known = torch.zeros(1, V, device=self.device)
        km      = torch.zeros_like(v_known)
        v_known[:, Dz:] = y_i
        km[:, Dz:] = 1.0

        # init informata con µ_k se disponibile, altrimenti mezzo-step
        try:
            if hasattr(self, "z_class_mean") and self.z_class_mean is not None:
                z0 = self.z_class_mean[y_i.argmax(dim=1)]
                v_cur = v_known.clone()
                v_cur[:, :Dz] = z0
            else:
                h0 = self.joint_rbm.forward(v_known)
                v_prob0 = self.joint_rbm.visible_probs(h0)
                v_cur = v_prob0 * (1 - km) + v_known * km
        except Exception:
            h0 = self.joint_rbm.forward(v_known)
            v_prob0 = self.joint_rbm.visible_probs(h0)
            v_cur = v_prob0 * (1 - km) + v_known * km

        # traiettoria 2D + ricostruzioni
        traj = []
        recon_imgs = []  # ciascuna [H, W] (numpy)
        # GT image per il pannello
        x_gt = x_i.detach().cpu()
        # decodifica shape
        Npix = x_gt.numel()
        side = int(round(Npix ** 0.5))
        if side * side != Npix:
            # fallback "colonna": (H=Npix, W=1)
            H, W = int(Npix), 1
        else:
            H = W = side

        # funzione di util per convertire un vettore in immagine numpy [H, W] 0-1
        def _vec_to_img_np(vec_1xN: torch.Tensor) -> np.ndarray:
            v = vec_1xN.view(-1).detach().cpu()
            if side * side == Npix:
                img = v.view(1, H, W).clamp(0, 1)[0]
            else:
                img = v.view(1, H, W).clamp(0, 1)[0]
            return img.numpy()

        # stato iniziale
        z_init = v_cur[:, :Dz].detach().cpu().numpy()
        traj.append(pca.transform(z_init)[0])
        img0 = self.image_idbn.decode(v_cur[:, :Dz]).detach()  # [1, Npix]
        recon_imgs.append(_vec_to_img_np(img0))

        for _ in range(int(steps)):
            h_prob = self.joint_rbm.forward(v_cur)
            v_prob = self.joint_rbm.visible_probs(h_prob)
            v_cur  = v_prob * (1 - km) + v_known * km   # re-clamp
            z_t = v_cur[:, :Dz].detach().cpu().numpy()
            traj.append(pca.transform(z_t)[0])
            img_t = self.image_idbn.decode(v_cur[:, :Dz]).detach()
            recon_imgs.append(_vec_to_img_np(img_t))

        traj = np.stack(traj, axis=0)  # [steps+1, 2]

        # --- 6) Seleziona n_frames tappe "rappresentative" (oltre alla GT)
        # includiamo sempre step 0 e l’ultimo; il resto equispaziato
        if n_frames < 2: n_frames = 2
        sel_idx = np.unique(np.linspace(0, len(recon_imgs)-1, n_frames, dtype=int)).tolist()
        # costruiamo lista finale: prima la GT vera, poi le ricostruzioni selezionate
        panel_imgs = []
        panel_titles = []

        # GT first
        panel_imgs.append(_vec_to_img_np(x_gt))
        panel_titles.append("GT")

        # poi ricostruzioni ai passi selezionati
        for si in sel_idx:
            panel_imgs.append(recon_imgs[si])
            panel_titles.append(f"step {si}")

        # --- 7) Figura a due colonne: sinistra PCA+traiettoria, destra pannello immagini
        # layout flessibile: a destra facciamo una griglia 2 x ceil((n_frames+1)/2)
        import math
        n_tiles = len(panel_imgs)
        rows = 2
        cols = math.ceil(n_tiles / rows)

        fig = plt.figure(figsize=(8 + cols*2.2, max(6, rows*2.2)))
        gs = fig.add_gridspec(nrows=rows, ncols=cols+4)  # +4 colonne per lasciare spazio al plot PCA

        # --- (A) PCA + Traiettoria occupa le prime 4 colonne
        ax0 = fig.add_subplot(gs[:, :4])
        sc = ax0.scatter(Z2[:, 0], Z2[:, 1],
                        c=color_vec, cmap='viridis',
                        s=scatter_size, alpha=scatter_alpha)
        ax0.scatter(z_true_2d[0, 0], z_true_2d[0, 1],
                    s=80, marker="*", c="k", edgecolor="w", linewidths=0.8,
                    label=f"sample GT (class={gt_class})", zorder=3)
        ax0.scatter(traj[0, 0], traj[0, 1],
                    s=50, marker="D", c="red", edgecolor="k", linewidths=0.5,
                    label="start catena", zorder=3)
        ax0.plot(traj[:, 0], traj[:, 1],
                linewidth=1.6, marker="o", markersize=3, c="red", label="traiettoria", zorder=2)
        # time-stamps ogni ~10% della traiettoria
        for t in range(0, len(traj), max(1, len(traj)//10)):
            ax0.text(traj[t, 0], traj[t, 1], str(t), fontsize=7, color="red")

        ax0.set_title(f"PCA z_img — sample {sample_idx} (class={gt_class}) — steps={steps}")
        ax0.set_xlabel("PC1"); ax0.set_ylabel("PC2")
        cbar = fig.colorbar(sc, ax=ax0, fraction=0.046, pad=0.02)
        cbar.set_label("Numerosity / N_list (fallback: Labels)")
        ax0.legend(loc="best")

        # --- (B) Pannello immagini a destra
        # griglia 2 x cols: posizioniamo da gs[:, 4:] usando subgridspec
        right_gs = gs[:, 4:].subgridspec(nrows=rows, ncols=cols)
        for k, img in enumerate(panel_imgs):
            r = k // cols
            c = k % cols
            ax = fig.add_subplot(right_gs[r, c])
            ax.imshow(img, cmap="gray", vmin=0.0, vmax=1.0)
            ax.set_title(panel_titles[k], fontsize=9)
            ax.axis("off")

        plt.tight_layout()

        # --- 8) Log
        if self.wandb_run:
            self.wandb_run.log({f"{tag}/plot": wandb.Image(fig)})
            plt.close(fig)
        else:
            plt.show()
    # ---- init bias visibili del joint
    @torch.no_grad()
    def init_joint_bias_from_data(self, n_batches: int = 10):
        # fallback robusto
        if not hasattr(self, "Dz_img"):
            if hasattr(self, "joint_rbm"):
                total_v = int(self.joint_rbm.num_visible)
                self.Dz_img = total_v - self.num_labels
            else:
                self.Dz_img = int(self.image_idbn.layers[-1].num_hidden)

        Dz = self.Dz_img
        K = self.num_labels
        sum_z = None
        n = 0
        class_counts = torch.zeros(K, device=self.device)
        for b, (imgs, lbls) in enumerate(self.dataloader):
            if b >= n_batches:
                break
            v = imgs.to(self.device).view(imgs.size(0), -1).float()
            z = self.image_idbn.represent(v)
            sum_z = z.sum(0) if sum_z is None else (sum_z + z.sum(0))
            n += z.size(0)
            class_counts += lbls.to(self.device).float().sum(0)
        if n == 0:
            return
        mean_z = (sum_z / n).clamp(1e-4, 1 - 1e-4)
        priors = class_counts / max(1, class_counts.sum())
        priors = (priors + 1e-6) / (priors.sum() + 1e-6 * K)
        try:
            with torch.no_grad():
                self.z_class_mean = torch.zeros(K, Dz, device=self.device)
                self.z_class_count = torch.zeros(K, device=self.device)
                # seconda passata leggera: accumula per classe (limita a n_batches per costo)
                processed = 0
                for b, (imgs, lbls) in enumerate(self.dataloader):
                    if b >= n_batches:
                        break
                    v = imgs.to(self.device).view(imgs.size(0), -1).float()
                    z = self.image_idbn.represent(v)  # [B, Dz]
                    y_idx = lbls.argmax(dim=1)        # [B]
                    for k in range(K):
                        mask = (y_idx == k)
                        if mask.any():
                            self.z_class_mean[k] += z[mask].sum(0)
                            self.z_class_count[k] += mask.sum()
                    processed += z.size(0)
                # normalizza; fallback alla media globale se una classe ha 0 conteggi
                for k in range(K):
                    if self.z_class_count[k] > 0:
                        self.z_class_mean[k] /= self.z_class_count[k]
                    else:
                        self.z_class_mean[k] = mean_z.clone()
        except Exception as e:
            print(f"[init_joint_bias_from_data] warn: z_class_mean not computed ({e})")
            # fallback: media unica per tutte le classi
            self.z_class_mean = mean_z.unsqueeze(0).repeat(K, 1)

        
        self.joint_rbm.vis_bias[:Dz] = torch.log(mean_z) - torch.log1p(-mean_z)
        self.joint_rbm.vis_bias[Dz : Dz + K] = torch.log(priors)

    # ---- load iDBN pre-allenata
    def load_pretrained_image_idbn(self, path: str) -> bool:
        import pickle
        try:
            with open(path, "rb") as f:
                obj = pickle.load(f)
        except Exception as e:
            print(f"[load_pretrained_image_idbn] errore: {e}")
            return False

        if isinstance(obj, dict) and "layers" in obj:
            self.image_idbn.layers = obj["layers"]
        elif hasattr(obj, "layers"):
            self.image_idbn = obj
            if not hasattr(self.image_idbn, "text_flag"):
                self.image_idbn.text_flag = False
            if not hasattr(self.image_idbn, "arch_dir"):
                self.image_idbn.arch_dir = os.path.join("logs-idbn", "loaded")
                os.makedirs(self.image_idbn.arch_dir, exist_ok=True)
        else:
            print("[load_pretrained_image_idbn] formato non riconosciuto")
            return False

        # porta su device e init buffers
        for rbm in self.image_idbn.layers:
            rbm.W = rbm.W.to(self.device)
            rbm.hid_bias = rbm.hid_bias.to(self.device)
            rbm.vis_bias = rbm.vis_bias.to(self.device)
            rbm.W_m = torch.zeros_like(rbm.W)
            rbm.hb_m = torch.zeros_like(rbm.hid_bias)
            rbm.vb_m = torch.zeros_like(rbm.vis_bias)
            if not hasattr(rbm, "softmax_groups"):
                rbm.softmax_groups = []

        dz_pre = int(self.image_idbn.layers[-1].num_hidden)
        if dz_pre != getattr(self, "Dz_img", dz_pre):
            print(f"[load_pretrained_image_idbn] adatto joint: Dz_img -> {dz_pre}")
            self._build_joint(Dz_img=dz_pre, joint_hidden=self.joint_rbm.num_hidden)

        print(f"[load_pretrained_image_idbn] caricato {path}")
        return True

    # ---- fine tuning leggero dell'ultimo RBM immagine (opzionale)
    def finetune_image_last_layer(self, epochs: int = 0, lr_scale: float = 0.3, cd_k: Optional[int] = None):
        if epochs <= 0:
            return
        last = self.image_idbn.layers[-1]
        old_lr = float(last.lr)
        last.lr = max(1e-8, old_lr * float(lr_scale))
        use_cd = int(cd_k) if cd_k is not None else int(self.image_idbn.cd_k)
        print(f"[finetune_image_last_layer] epochs={epochs}, lr={last.lr:.4g}, CD={use_cd}")
        for ep in range(int(epochs)):
            losses = []
            for img, _ in self.dataloader:
                v = img.to(self.device).view(img.size(0), -1).float()
                for rbm in self.image_idbn.layers[:-1]:
                    v = rbm.forward(v)
                loss = last.train_epoch(v, ep, epochs, CD=use_cd)
                losses.append(float(loss))
            if self.wandb_run and losses:
                self.wandb_run.log({"img_last/finetune_loss": float(np.mean(losses)), "epoch_ft": ep})
        last.lr = old_lr
        print("[finetune_image_last_layer] done")

    # ---- cross reconstruction
    @torch.no_grad()
    def _cross_reconstruct(self, z_img: torch.Tensor, y_onehot: torch.Tensor, steps: Optional[int] = None):
        if steps is None:
            steps = self.cross_steps
        B = z_img.size(0)
        Dz = self.Dz_img
        K = self.num_labels
        V = Dz + K

        # IMG -> TXT
        v_known = torch.zeros(B, V, device=self.device)
        km = torch.zeros_like(v_known)
        v_known[:, :Dz] = z_img
        km[:, :Dz] = 1.0
        v_img2txt = self.joint_rbm.conditional_gibbs(v_known, km, n_steps=steps, sample_h=False, sample_v=False)
        p_y_given_img = v_img2txt[:, Dz:]

        # TXT -> IMG
        v_known.zero_()
        km.zero_()
        v_known[:, Dz:] = y_onehot
        km[:, Dz:] = 1.0

        # === INIZIALIZZAZIONE INFORMATA: z <- µ_k della classe clampata (se presente) ===
        # fallback: mean-field "pulito" dal mezzo step
        with torch.no_grad():
            # stato v corrente: parti dal clamp sui noti
            v_cur = v_known.clone()
            try:
                if hasattr(self, "z_class_mean") and self.z_class_mean is not None:
                    # prendi l'indice di classe per ciascun elemento del batch
                    y_idx = y_onehot.argmax(dim=1)  # [B]
                    z0 = self.z_class_mean[y_idx]   # [B, Dz]
                    v_cur[:, :Dz] = z0
                else:
                    # mezzo step "pulito"
                    h0 = self.joint_rbm.forward(v_known)
                    v_prob0 = self.joint_rbm.visible_probs(h0)
                    v_cur = v_prob0 * (1 - km) + v_known * km
            except Exception:
                h0 = self.joint_rbm.forward(v_known)
                v_prob0 = self.joint_rbm.visible_probs(h0)
                v_cur = v_prob0 * (1 - km) + v_known * km

            # === Gibbs condizionale deterministica (mean-field) con re-clamp a ogni step ===
            for _ in range(int(steps)):
                h_prob = self.joint_rbm.forward(v_cur)
                v_prob = self.joint_rbm.visible_probs(h_prob)
                v_cur = v_prob * (1 - km) + v_known * km  # re-clamp

        z_img_from_y = v_cur[:, :Dz]
        img_from_txt = self.image_idbn.decode(z_img_from_y)
        return img_from_txt, p_y_given_img

    # ---- rappresentazione top (joint hidden)
    @torch.no_grad()
    def represent(self, batch):
        img_data, lbl_data = batch
        img = img_data.to(self.device).view(img_data.size(0), -1).float()
        y = lbl_data.to(self.device).float()
        z_img = self.image_idbn.represent(img)
        v = torch.cat([z_img, y], dim=1)
        return self.joint_rbm.forward(v)

    # ---- train joint: autorecon -> cross-modality + PCA/Probes
    def train_joint(self, epochs, log_every_pca: int = 25, log_every_probe: int = 10, log_every: int = 5, w_rec: float = 1.0, w_sup: float = 0.0):
        print("[iMDBN] joint training (simple)")
        self.init_joint_bias_from_data(n_batches=10)

        for epoch in tqdm(range(int(epochs)), desc="Joint"):
            cd_losses = []
            totals = {"n": 0, "top1": 0, "top3": 0, "ce_sum": 0.0, "mse_sum": 0.0, "npix": None}

            for b_idx, (img, y) in enumerate(self.dataloader):
                img = img.to(self.device).view(img.size(0), -1).float()
                y = y.to(self.device).float()

                with torch.no_grad():
                    z_img = self.image_idbn.represent(img)
                    v_plus = torch.cat([z_img, y], dim=1)

                # CD-k joint (autorecon prima)
                loss_cd = self.joint_rbm.train_epoch(v_plus, epoch, epochs, CD=self.joint_cd)
                cd_losses.append(float(loss_cd))

                # Auxiliary clamped-CD alternata (se richiesto)
                if self.aux_every_k > 0 and (b_idx % self.aux_every_k == 0):
                    B, Dz, K = z_img.size(0), self.Dz_img, self.num_labels
                    V = Dz + K
                    v_known = torch.zeros(B, V, device=self.device)
                    km = torch.zeros_like(v_known)
                    clamp_image = ((b_idx // self.aux_every_k) % 2 == 0)
                    if clamp_image:
                        v_known[:, :Dz] = z_img
                        km[:, :Dz] = 1.0
                    else:
                        v_known[:, Dz:] = y
                        km[:, Dz:] = 1.0
                    _ = self.joint_rbm.train_epoch_clamped(
                        v_known, km, epoch, epochs, CD=1, cond_init_steps=self.aux_cond_steps,
                        sample_h=False, sample_v=False
                    )

                # Cross-modality metrics online
                with torch.no_grad():
                    img_from_txt, p_y_given_img = self._cross_reconstruct(z_img, y, steps=self.cross_steps)
                    gt = y.argmax(dim=1)
                    pred = p_y_given_img.argmax(dim=1)
                    topk = min(3, p_y_given_img.size(1))
                    topk_idx = p_y_given_img.topk(k=topk, dim=1).indices

                    ce = F.binary_cross_entropy(
                        p_y_given_img.clamp(1e-6, 1 - 1e-6),
                        F.one_hot(gt, num_classes=p_y_given_img.size(1)).float(),
                        reduction="sum",
                    )
                    npix = img.view(img.size(0), -1).size(1)
                    mse = F.mse_loss(img_from_txt.view_as(img), img, reduction="sum")

                    totals["n"] += img.size(0)
                    totals["top1"] += (pred == gt).sum().item()
                    totals["top3"] += (topk_idx == gt.unsqueeze(1)).any(dim=1).sum().item()
                    totals["ce_sum"] += float(ce.item())
                    totals["mse_sum"] += float(mse.item())
                    totals["npix"] = npix

            # log epoch
            if self.wandb_run and cd_losses:
                self.wandb_run.log({"joint/cd_loss": float(np.mean(cd_losses)), "epoch": epoch})

            if self.wandb_run and totals["n"] > 0:
                top1 = totals["top1"] / totals["n"]
                top3 = totals["top3"] / totals["n"]
                ce_mean = totals["ce_sum"] / totals["n"]
                mse_mean = totals["mse_sum"] / max(1, totals["n"] * max(1, totals["npix"] or 1))
                self.wandb_run.log({
                    "cross_modality/text_top1": top1,
                    "cross_modality/text_top3": top3,
                    "cross_modality/text_ce": ce_mean,
                    "cross_modality/image_mse": mse_mean,
                    "epoch": epoch
                })

            # PCA + PROBES sul joint (con compute_joint_* → lunghezze coerenti)
            if self.wandb_run and self.val_loader is not None and self.features is not None:
                if epoch % log_every_pca == 0:
                    try:
                        E, feats = compute_joint_embeddings_and_features(self)
                        if E.numel() > 0:
                            emb_np = E.numpy()
                            feat_map = {
                                "Cumulative Area": feats["cum_area"].numpy(),
                                "Convex Hull": feats["convex_hull"].numpy(),
                                "Labels": feats["labels"].numpy(),
                            }
                            if "density" in feats:
                                feat_map["Density"] = feats["density"].numpy()
                            if emb_np.shape[0] > 2 and emb_np.shape[1] > 2:
                                p2 = PCA(n_components=2).fit_transform(emb_np)
                                plot_2d_embedding_and_correlations(
                                    emb_2d=p2,
                                    features=feat_map,
                                    arch_name="Joint_top",
                                    dist_name="val",
                                    method_name="pca",
                                    wandb_run=self.wandb_run,
                                )
                                if emb_np.shape[1] >= 3:
                                    p3 = PCA(n_components=3).fit_transform(emb_np)
                                    plot_3d_embedding_and_correlations(
                                        emb_3d=p3,
                                        features=feat_map,
                                        arch_name="Joint_top",
                                        dist_name="val",
                                        method_name="pca",
                                        wandb_run=self.wandb_run,
                                    )
                    except Exception as e:
                        self.wandb_run.log({"warn/joint_pca_error": str(e)})

                if epoch % log_every_probe == 0:
                    try:
                        log_joint_linear_probe(
                            self,
                            epoch=epoch,
                            n_bins=5,
                            test_size=0.2,
                            steps=1000,
                            lr=1e-2,
                            patience=20,
                            min_delta=0.0,
                            metric_prefix="joint",
                        )
                    except Exception as e:
                        self.wandb_run.log({"warn/joint_probe_error": str(e)})
                if epoch % 5 == 0:
                    # esempio: caso con label 12, 40 step di catena
                    #run_and_log_fixed_case(self, epoch=epoch, target_label=12, steps=40, tag="fixed_lbl12")
                    run_and_log_cross_fixed_case(self, epoch=epoch, target_label=29,
                                 max_steps=self.cross_steps, sample_h=False, sample_v=False,
                                 tag="fixed_lbl12")
                    run_and_log_z_mismatch_check(self, epoch=epoch, max_steps=self.cross_steps, sample_h=False, sample_v=False, tag="val")
                # scegli un indice (o usa la utility che trova il primo con label k)
                    self.log_vecdb_neighbors_for_traj(
                        sample_idx=0,
                        steps=self.cross_steps,
                        k=8,
                        metric="cosine",
                        tag="vecdb",
                        also_l2=True,          # confronta anche L2
                        dedup="image",         # evita “stessi sample” o near-duplicates visuali
                        exclude_self=True
                    )
                    
                

                if epoch % 10 == 0:
                    # pannello fisso: 4 sample per classe (≈128 sample totali)
                    run_and_log_cross_panel(
                        self, epoch=epoch, per_class=4,
                        max_steps=self.cross_steps,
                        sample_h=False, sample_v=False,
                        tag="panel_4_per_class"
    )               # PCA-3 della traiettoria per una classe piccola (es. 4)
                    idx4 = self.find_first_val_index_with_label(4)
                    if idx4 >= 0:
                        self.log_pca3_trajectory(sample_idx=idx4, steps=self.cross_steps, tag="pca3_lbl4")

                    self.log_latent_trajectory_with_recon_panel(sample_idx=idx4, steps=self.cross_steps, tag="sanity_pca_traj")



            # snapshot & autorecon
            if epoch % max(1, int(log_every)) == 0:
                self._log_snapshots(epoch)
                self._log_joint_auto_recon(epoch)

        print("[iMDBN] joint training finished.")

    def _ensure_val_bank(self):
        """Costruisce e memorizza la banca (Z, X, Y) del validation set + un hash grezzo per dedup immagini."""
        if hasattr(self, "_Z_bank"):
            return
        Z_list, X_list, Y_list, H_list = [], [], [], []
        for imgs, lbls in self.val_loader:
            z = self.image_idbn.represent(imgs.view(imgs.size(0), -1).float().to(self.device))  # [B,Dz]
            Z_list.append(z.cpu())
            X_list.append(imgs.cpu())
            Y_list.append(lbls.cpu())
            # hash semplice: somma + somma dei quadrati (veloce, non crittografico ma utile per dedup grezza)
            flat = imgs.view(imgs.size(0), -1).float().cpu()
            h = torch.stack([flat.sum(1), (flat**2).sum(1)], dim=1)  # [B,2]
            H_list.append(h)
        self._Z_bank = torch.cat(Z_list, 0)   # [N,Dz]
        self._X_bank = torch.cat(X_list, 0)   # [N,C,H,W] o [N,Npix]
        self._Y_bank = torch.cat(Y_list, 0)   # [N,K]
        self._H_bank = torch.cat(H_list, 0)   # [N,2]
    
    @torch.no_grad()
    def find_first_val_index_with_label(self, k: int) -> int:
        """Ritorna l'indice (0-based) nel validation set del primo sample con argmax(label)==k; -1 se non trovato."""
        idx = 0
        for imgs, lbls in self.val_loader:
            y = lbls.argmax(1)
            for j in range(y.size(0)):
                if int(y[j].item()) == int(k):
                    return idx + j
            idx += y.size(0)
        return -1

    @torch.no_grad()
    def build_latent_bank_val(self):
        """Costruisce la banca Z_val (Dz) e salva anche le immagini val nello stesso ordine."""
        Z_list, Y_list, X_list = [], [], []
        for imgs, lbls in self.val_loader:
            z = self.image_idbn.represent(imgs.view(imgs.size(0), -1).float().to(self.device))
            Z_list.append(z.cpu())
            Y_list.append(lbls.cpu())
            X_list.append(imgs.cpu())
        self._Z_bank = torch.cat(Z_list, 0)   # [N, Dz]
        self._Y_bank = torch.cat(Y_list, 0)   # [N, K]
        self._X_bank = torch.cat(X_list, 0)   # [N, C,H,W] o [N,Npix]

    @torch.no_grad()
    def topk_similar_in_latent(self, z_query: torch.Tensor, k: int = 8, metric: str = "cosine", same_class_only: bool = None):
        """
        z_query: [B, Dz] sul device
        Ritorna (indices [B,k], scores [B,k]) con similarità tipo vector DB (coseno/inner).
        """
        assert hasattr(self, "_Z_bank"), "chiama prima build_latent_bank_val()"
        Z = self._Z_bank  # [N,Dz] (CPU)
        # normalizza se coseno
        if metric == "cosine":
            Zn = F.normalize(Z, dim=1)
            zq = F.normalize(z_query.detach().cpu(), dim=1)
            scores = zq @ Zn.T                     # [B,N], cos-sim = prodotto interno normalizzato
        elif metric in ("ip", "inner"):
            zq = z_query.detach().cpu()
            scores = zq @ Z.T
        else:
            # distanza euclidea: converti in similarità negativa
            zq = z_query.detach().cpu()
            # ||a-b||^2 = ||a||^2 + ||b||^2 - 2 a·b
            a2 = (zq**2).sum(1, keepdim=True)      # [B,1]
            b2 = (Z**2).sum(1).unsqueeze(0)        # [1,N]
            ip = zq @ Z.T
            d2 = a2 + b2 - 2*ip
            scores = -d2

        if same_class_only is not None:
            # maschera per filtrare solo la stessa classe (o diversa)
            gt = self._Y_bank.argmax(1)            # [N]
            # qui assumiamo z_query B=1 per semplicità; estendibile
            cls = z_query.size(0) == 1 and int(self._current_y.argmax().item()) or None

        topv, topi = torch.topk(scores, k=min(k, Z.size(0)), dim=1)
        return topi, topv

    
    @torch.no_grad()
    def log_pca3_trajectory(self, sample_idx: int, steps: int = 40, tag: str = "pca3_traj"):
        from sklearn.decomposition import PCA
        import matplotlib.pyplot as plt

        # 1) Embed val nel top immagine (Dz)
        Z_val_t, _ = compute_val_embeddings_and_features(self.image_idbn, upto_layer=len(self.image_idbn.layers))
        if Z_val_t is None or Z_val_t.numel() == 0: 
            if self.wandb_run: self.wandb_run.log({f"{tag}/warn":"no val embeddings"})
            return
        Z_val = Z_val_t.cpu().numpy()

        # 2) Prendi il sample e fai la catena TXT→IMG baseline salvando z_t
        #    (riusa esattamente la tua _cross_reconstruct ma accumula v_cur[:,:Dz])
        Dz = self.Dz_img; K = self.num_labels; V = Dz + K
        # sample (x_i,y_i)
        seen, x_i, y_i = 0, None, None
        for imgs, lbls in self.val_loader:
            b = imgs.size(0)
            if seen + b <= sample_idx: seen += b; continue
            pos = sample_idx - seen
            x_i = imgs[pos:pos+1].to(self.device).view(1, -1).float()
            y_i = lbls[pos:pos+1].to(self.device).float()
            break
        v_known = torch.zeros(1, V, device=self.device); km = torch.zeros_like(v_known)
        v_known[:, Dz:] = y_i; km[:, Dz:] = 1.0
        if hasattr(self, "z_class_mean") and self.z_class_mean is not None:
            z0 = self.z_class_mean[y_i.argmax(1)]; v_cur = v_known.clone(); v_cur[:, :Dz] = z0
        else:
            h0 = self.joint_rbm.forward(v_known); v_prob0 = self.joint_rbm.visible_probs(h0)
            v_cur = v_prob0 * (1 - km) + v_known * km
        zs = [v_cur[:, :Dz].detach().cpu().numpy()]
        for _ in range(int(steps)):
            h_prob = self.joint_rbm.forward(v_cur)
            v_prob = self.joint_rbm.visible_probs(h_prob)
            v_cur  = v_prob * (1 - km) + v_known * km
            zs.append(v_cur[:, :Dz].detach().cpu().numpy())
        Z_traj = np.vstack(zs)

        # 3) PCA(3) su tutto il val, proietta la traiettoria
        p3 = PCA(n_components=3).fit(Z_val)
        Z3 = p3.transform(Z_val)
        T3 = p3.transform(Z_traj)

        # 4) Log 3D plot
        from mpl_toolkits.mplot3d import Axes3D  # noqa
        fig = plt.figure(figsize=(6.5,6))
        ax = fig.add_subplot(111, projection="3d")
        ax.scatter(Z3[:,0], Z3[:,1], Z3[:,2], s=6, alpha=0.15)
        ax.plot(T3[:,0], T3[:,1], T3[:,2], c="r", linewidth=1.2)
        ax.set_title("PCA-3 trajectory")
        fig.tight_layout()
        if self.wandb_run: self.wandb_run.log({f"{tag}/pca3": wandb.Image(fig)}); plt.close(fig)

    @torch.no_grad()
    def log_vecdb_neighbors_for_traj(
        self,
        sample_idx: int = 0,
        steps: int = None,
        k: int = 8,
        metric: str = "cosine",          # "cosine" | "inner" | "l2"
        tag: str = "vecdb",
        also_l2: bool = True,            # logga anche i vicini L2
        dedup: str = "index",            # "none" | "index" | "image"
        exclude_self: bool = True        # escludi il sample stesso dalla lista dei vicini
    ):
        """
        Per un sample del validation set:
        - genera la traiettoria TXT->IMG (mean-field) e prende z_true, z0, zT,
        - fa similarity search nel latente completo con metrica scelta,
        - deduplica i vicini e (opzionale) esclude il sample stesso,
        - logga griglie immagini + statistiche + tabella punteggi.
        """
        import numpy as np, torch
        import torch.nn.functional as F
        import torchvision.utils as vutils

        # --- prepara banca
        self._ensure_val_bank()
        Z_bank, X_bank, Y_bank, H_bank = self._Z_bank, self._X_bank, self._Y_bank, self._H_bank
        N, Dz = Z_bank.shape

        # --- recupera (x_i, y_i) con indice coerente col bank
        seen = 0; x_i = None; y_i = None
        for imgs, lbls in self.val_loader:
            b = imgs.size(0)
            if seen + b <= sample_idx: seen += b; continue
            pos = sample_idx - seen
            x_i = imgs[pos:pos+1].to(self.device).view(1, -1).float()
            y_i = lbls[pos:pos+1].to(self.device).float()
            break
        if x_i is None:
            if self.wandb_run: self.wandb_run.log({f"{tag}/warn": "sample_idx out of range"})
            return

        # --- traiettoria TXT->IMG (baseline)
        Dz_img = self.Dz_img; K = self.num_labels; V = Dz_img + K
        v_known = torch.zeros(1, V, device=self.device); km = torch.zeros_like(v_known)
        v_known[:, Dz_img:] = y_i; km[:, Dz_img:] = 1.0
        if hasattr(self, "z_class_mean") and self.z_class_mean is not None:
            z0_init = self.z_class_mean[y_i.argmax(1)]
            v_cur = v_known.clone(); v_cur[:, :Dz_img] = z0_init
        else:
            h0 = self.joint_rbm.forward(v_known)
            v_prob0 = self.joint_rbm.visible_probs(h0)
            v_cur = v_prob0 * (1 - km) + v_known * km

        T = int(self.cross_steps if steps is None else steps)
        zs = [v_cur[:, :Dz_img].detach().cpu().numpy()]
        for _ in range(T):
            h_prob = self.joint_rbm.forward(v_cur)
            v_prob = self.joint_rbm.visible_probs(h_prob)
            v_cur  = v_prob * (1 - km) + v_known * km
            zs.append(v_cur[:, :Dz_img].detach().cpu().numpy())
        Z_traj = np.vstack(zs)  # [T+1, Dz]

        z_true = self.image_idbn.represent(x_i).to(self.device)                # [1,Dz]
        z0     = torch.from_numpy(Z_traj[:1]).to(self.device).float()          # [1,Dz]
        zT     = torch.from_numpy(Z_traj[-1:]).to(self.device).float()         # [1,Dz]

        # --- scorer
        def score(zq: torch.Tensor, met: str) -> torch.Tensor:
            zq = zq.detach().cpu()                 # [1,Dz]
            if met == "cosine":
                return F.normalize(zq, 1) @ F.normalize(Z_bank, 1).T          # [1,N], alto = simile
            elif met in ("inner", "ip"):
                return zq @ Z_bank.T
            else:  # l2
                a2 = (zq**2).sum(1, keepdim=True)  # [1,1]
                b2 = (Z_bank**2).sum(1).unsqueeze(0)    # [1,N]
                ip = zq @ Z_bank.T
                return -(a2 + b2 - 2*ip)           # negativo della dist^2 → alto = vicino

        # --- estrae top-k con dedup e (opzionale) exclude_self
        def topk_dedup(zq: torch.Tensor, met: str, k: int, name: str):
            scores = score(zq, met).squeeze(0)             # [N]
            # ordina (desc)
            vals, idx = torch.sort(scores, descending=True)
            ids = idx.tolist()
            vs  = vals.tolist()

            picked_ids, picked_vs = [], []
            seen_idx = set()
            seen_hash = set()

            # per escludere il sample stesso: quale indice corrisponde?
            # sample_idx è coerente con l'ordine di costruzione del bank, quindi è proprio lui.
            for i, v in zip(ids, vs):
                if exclude_self and i == sample_idx:
                    continue
                if dedup == "index":
                    if i in seen_idx: 
                        continue
                    seen_idx.add(i)
                elif dedup == "image":
                    # dedup grezza per hash [sum, sumsq]
                    key = (float(H_bank[i,0].item()), float(H_bank[i,1].item()))
                    if key in seen_hash:
                        continue
                    seen_hash.add(key)
                # "none": non filtra
                picked_ids.append(i); picked_vs.append(v)
                if len(picked_ids) >= k:
                    break

            if self.wandb_run:
                uniq = len(set(picked_ids))
                self.wandb_run.log({
                    f"{tag}/{name}_uniq_count": uniq,
                    f"{tag}/{name}_score_min": float(min(picked_vs)) if picked_vs else float("nan"),
                    f"{tag}/{name}_score_max": float(max(picked_vs)) if picked_vs else float("nan"),
                    f"{tag}/{name}_score_mean": float(sum(picked_vs)/max(1,len(picked_vs))) if picked_vs else float("nan"),
                })
                # istogramma etichette dei vicini
                y_idx = Y_bank.argmax(1)
                labels = [int(y_idx[i].item()) for i in picked_ids]
                tbl = wandb.Table(columns=["rank","index","label","score"])
                for r,(i,v) in enumerate(zip(picked_ids, picked_vs)):
                    tbl.add_data(r, int(i), labels[r], float(v))
                self.wandb_run.log({f"{tag}/{name}_neighbor_labels": tbl})
            return torch.tensor(picked_ids, dtype=torch.long).unsqueeze(0), torch.tensor(picked_vs).unsqueeze(0)

        # --- prendi vicini secondo metrica principale
        idx_true, sc_true = topk_dedup(z_true, metric, k, "true")
        idx_z0,   sc_z0   = topk_dedup(z0,     metric, k, "z0")
        idx_zT,   sc_zT   = topk_dedup(zT,     metric, k, "zT")

        # --- opzionale: anche L2 (utile se la cosine appiattisce)
        if also_l2 and metric != "l2":
            idx_zT_l2, sc_zT_l2 = topk_dedup(zT, "l2", k, "zT_l2")
        else:
            idx_zT_l2 = sc_zT_l2 = None

        # --- log griglie immagini
        def log_neighbors(indices: torch.Tensor, subtag: str):
            pick = indices[0]
            sel = X_bank[pick]                         # [k,...]
            if sel.ndim == 2:                          # [k,Npix] -> [k,1,H,W]
                Npix = sel.size(1); side = int(round(Npix**0.5))
                sel = sel.view(sel.size(0), 1, side, side)
            grid = vutils.make_grid(sel, nrow=min(4, sel.size(0)))
            if self.wandb_run:
                self.wandb_run.log({f"{tag}/{subtag}": wandb.Image(grid.permute(1,2,0).numpy())})

        log_neighbors(idx_true, "knn_true")
        log_neighbors(idx_z0,   "knn_z0")
        log_neighbors(idx_zT,   "knn_zT")
        if idx_zT_l2 is not None:
            log_neighbors(idx_zT_l2, "knn_zT_l2")

    @torch.no_grad()
    def _log_snapshots(self, epoch, num=8):
        if self.wandb_run is None or self.validation_images is None or self.validation_labels is None:
            return

        imgs = self.validation_images[:num]
        lbls = self.validation_labels[:num]

        # --- IMG->TXT->IMG cross reconstruction ---
        zi = self.image_idbn.represent(imgs.view(imgs.size(0), -1))
        img_from_txt, p_y_given_img = self._cross_reconstruct(zi, lbls, steps=self.cross_steps)
        rec = img_from_txt.clamp(0, 1)  # [B, N] o [B, C*H*W]

        # --- rendi entrambe le immagini 4D (B, C, H, W) in modo robusto
        if imgs.ndim == 4:
            B, C, H, W = imgs.shape
            imgs4 = imgs
            rec4  = rec.view(B, C, H, W)
        else:
            B = imgs.size(0)
            N = imgs.size(1)
            side = int(round(N ** 0.5))
            if side * side != N:
                # fallback: evento raro → tratta come (C=1, H=N, W=1)
                C, H, W = 1, N, 1
            else:
                C, H, W = 1, side, side
            imgs4 = imgs.view(B, C, H, W)
            rec4  = rec.view(B, C, H, W)

        # === GT | REC in un'unica griglia (2 colonne) ===
        pair = torch.stack([imgs4.cpu(), rec4.cpu()], dim=1).view(-1, C, H, W)
        grid_pair = vutils.make_grid(pair, nrow=2)
        self.wandb_run.log({
            "snap/image_from_text": wandb.Image(grid_pair.permute(1, 2, 0).numpy()),
            "epoch": epoch
        })

        # --- Confusion matrix (usa class_names se disponibile)
        class_names = getattr(self, "class_names", None)
        pred = p_y_given_img.argmax(dim=1)
        gt = lbls.argmax(dim=1)
        if class_names and len(class_names) == self.num_labels:
            cm_plot = wandb.plot.confusion_matrix(
                y_true=[class_names[i] for i in gt.cpu().numpy()],
                preds=[class_names[i] for i in pred.cpu().numpy()],
                class_names=class_names,
            )
        else:
            cm_plot = wandb.plot.confusion_matrix(
                y_true=gt.cpu().numpy(),
                preds=pred.cpu().numpy(),
                class_names=[str(i) for i in range(self.num_labels)],
            )
        self.wandb_run.log({"snap/text_confusion": cm_plot, "epoch": epoch})

        # --- MSE immagine (flatten coerente)
        mse = F.mse_loss(imgs4.view(B, -1).to(self.device), rec4.view(B, -1).to(self.device)).item()
        self.wandb_run.log({"snap/image_mse": mse, "epoch": epoch})

        # === Tabella TOP-K per le probabilità testo ===
        # (per ogni sample nello snapshot: gt/pred con indici e nomi, p(pred), p(y_true))
        try:
            probs = p_y_given_img.clamp(1e-9, 1).detach().cpu()  # [B,K]
            topk = min(2, probs.size(1))
            top_vals, top_inds = probs.topk(topk, dim=1)
            cols = ["idx", "gt_idx", "pred_idx", "p_pred", "p_y_true"]
            if class_names and len(class_names) == self.num_labels:
                cols += ["gt_label", "pred_label"]
            tbl = wandb.Table(columns=cols)
            for i in range(B):
                gt_i = int(gt[i].item())
                pred_i = int(pred[i].item())
                p_pred = float(probs[i, pred_i].item())
                p_gt   = float(probs[i, gt_i].item())
                row = [i, gt_i, pred_i, p_pred, p_gt]
                if class_names and len(class_names) == self.num_labels:
                    row += [class_names[gt_i], class_names[pred_i]]
                tbl.add_data(*row)
            self.wandb_run.log({ "snap/text_topk": tbl, "epoch": epoch })
        except Exception as e:
            self.wandb_run.log({ "warn/snap_topk_table_error": str(e), "epoch": epoch })
    
    @torch.no_grad()
    def log_neighbors_images(self, indices: torch.Tensor, tag: str):
        X = self._X_bank  # [N,...]
        pick = indices[0].to(torch.long)
        sel = X[pick]     # [k,...]
        if sel.ndim == 2:  # [k, Npix] -> [k,1,H,W]
            Npix = sel.size(1); side = int(round(Npix**0.5))
            sel = sel.view(sel.size(0), 1, side, side)
        grid = vutils.make_grid(sel, nrow=min(4, sel.size(0)))
        if self.wandb_run:
            self.wandb_run.log({tag: wandb.Image(grid.permute(1,2,0).numpy())})

    @torch.no_grad()
    def _log_joint_auto_recon(self, epoch, num=8):
        if self.wandb_run is None or self.validation_images is None or self.validation_labels is None:
            return

        imgs = self.validation_images[:num]
        lbls = self.validation_labels[:num]

        # --- forward joint
        z_top = self.image_idbn.represent(imgs.view(imgs.size(0), -1))
        v = torch.cat([z_top, lbls.to(self.device).float()], dim=1)
        h = self.joint_rbm.forward(v)
        v_recon = self.joint_rbm.backward(h)  # [B, Dz + K]
        Dz = self.Dz_img
        z_img_hat = v_recon[:, :Dz]
        y_hat = v_recon[:, Dz:]  # softmax block etichette

        # --- decodifica immagine dal top
        rec_img = self.image_idbn.decode(z_img_hat).clamp(0, 1)  # [B, Npix] o [B, C*H*W]

        # --- rendi entrambe le immagini 4D (B, C, H, W)
        if imgs.ndim == 4:
            B, C, H, W = imgs.shape
            imgs4 = imgs
        else:
            B = imgs.size(0)
            N = imgs.size(1)
            side = int(round(N ** 0.5))
            if side * side != N:
                # fallback: niente reshape quadrato → mostra solo la ricostruzione
                side = int(N)  # evitiamo crash; ma sul tuo dataset è 10000 => 100x100
            C, H, W = 1, side, side
            imgs4 = imgs.view(B, C, H, W)

        rec4 = rec_img.view(B, C, H, W)

        # --- grid GT vs Joint
        pair = torch.stack([imgs4.cpu(), rec4.cpu()], dim=1).view(-1, C, H, W)
        grid = vutils.make_grid(pair, nrow=2)
        self.wandb_run.log({"auto_recon/gt_vs_joint": wandb.Image(grid.permute(1, 2, 0).numpy()),
                            "epoch": epoch})

        # --- metriche testo dal joint
        gt = lbls.argmax(dim=1)
        pred = y_hat.argmax(dim=1)
        top1 = (pred == gt).float().mean().item()
        self.wandb_run.log({"auto_recon/text_top1": top1, "epoch": epoch})

        text_bce = F.binary_cross_entropy(y_hat.clamp(1e-6, 1 - 1e-6), lbls.float()).item()
        self.wandb_run.log({"auto_recon/text_bce": text_bce, "epoch": epoch})

        # --- MSE immagine (usa viste flatten coerenti)
        mse = F.mse_loss(imgs4.view(B, -1).to(self.device), rec4.view(B, -1).to(self.device)).item()
        self.wandb_run.log({"auto_recon/image_mse": mse, "epoch": epoch})

    def save_model(self, path: str):
        payload = {
            "image_idbn": self.image_idbn,
            "joint_rbm": self.joint_rbm,
            "num_labels": self.num_labels,
            "params": self.params,
        }
        with open(path, "wb") as f:
            pickle.dump(payload, f)
