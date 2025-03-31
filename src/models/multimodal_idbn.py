import pytorch_lightning as pl
import torch
import torch.nn as nn
import math
from torch.utils.data import DataLoader, TensorDataset

# Utility function
def sigmoid(x):
    return 1 / (1 + torch.exp(-x))

# RBM Model for iterative training
import pytorch_lightning as pl
import torch
import torch.nn as nn
import math

def sigmoid(x):
    return 1 / (1 + torch.exp(-x))


class RBM(nn.Module):
    def __init__(self, num_visible, num_hidden, learning_rate, weight_decay, momentum, dynamic_lr=False, final_momentum=0.97,batch_size = 10):
        super(RBM, self).__init__()

        self.automatic_optimization = False  # Disable automatic optimizatio
        self.num_visible = num_visible
        self.num_hidden = num_hidden
        self.lr = learning_rate
        self.weight_decay = weight_decay
        self.momentum = momentum
        self.dynamic_lr = dynamic_lr
        self.final_momentum = final_momentum
        self.batch_size = batch_size

        # Initialize weights and biases
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.W = nn.Parameter(torch.randn(num_visible, num_hidden, device=device) / math.sqrt(num_visible))
        self.hid_bias = nn.Parameter(torch.zeros(num_hidden, device=device))
        self.vis_bias = nn.Parameter(torch.zeros(num_visible, device=device))

        # Momentum buffers (not Parameters)
        self.W_momentum = torch.zeros_like(self.W)
        self.hid_bias_momentum = torch.zeros_like(self.hid_bias)
        self.vis_bias_momentum = torch.zeros_like(self.vis_bias)

    def forward(self, v):
        """ Compute probabilities of hidden units given visible units. """
        return torch.sigmoid(torch.matmul(v, self.W) + self.hid_bias)

    def backward(self, h):
        """ Compute probabilities of visible units given hidden units. """
        return torch.sigmoid(torch.matmul(h, self.W.T) + self.vis_bias)

    def train_epoch(self, data, epoch, CD=1):
        """
        Performs training for one epoch.
        `data` should have shape (batch_size, num_visible).
        """
        total_loss = 0.0
        lr = self.lr / (1 + 0.01 * epoch) if self.dynamic_lr else self.lr
        batch_size = data.size(0)
        momentum = self.momentum if epoch <= 5 else self.final_momentum
        batch_data = data
     # Shape: (num_visible,)
        with torch.no_grad():
            # Positive phase
            pos_hid_probs = self.forward(batch_data)
            pos_hid_states = (pos_hid_probs > torch.rand_like(pos_hid_probs)).float()
            pos_assoc = torch.matmul(batch_data.T,pos_hid_probs)  # Fix shape

            # Negative phase (Gibbs sampling)
            neg_data = batch_data.clone()
            for _ in range(CD):
                neg_vis_probs = self.backward(pos_hid_states)
                neg_data = (neg_vis_probs > torch.rand_like(neg_vis_probs)).float()
                neg_hid_probs = self.forward(neg_data)
                pos_hid_states = (neg_hid_probs > torch.rand_like(neg_hid_probs)).float()
                neg_assoc = torch.matmul(neg_data.T, neg_hid_probs)  # Fix shape

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

       
        return total_loss / batch_size
# Iterative DBN class
class iDBN(pl.LightningModule):
    def __init__(self, params, layer_sizes=[500, 500],batch_size = 10):
        super(iDBN, self).__init__()
        self.save_hyperparameters()  # Save params for easy access


        self.automatic_optimization = False  # Disable automatic optimizatio
        self.params = params
        self.layer_sizes = layer_sizes
        self.layers = nn.ModuleList()
        self.batch_size = batch_size 

        for i in range(len(layer_sizes) - 1):
            self.layers.append(RBM(
                num_visible=layer_sizes[i],
                num_hidden=layer_sizes[i + 1],
                learning_rate=params['LEARNING_RATE'],
                weight_decay=params['WEIGHT_PENALTY'],
                momentum=params['INIT_MOMENTUM'],
                dynamic_lr=params['LEARNING_RATE_DYNAMIC'],
                final_momentum=params['FINAL_MOMENTUM'],
                batch_size = self.batch_size
            ))

        self.current_layer = 0  # Track which layer is being trained
        self.epochs_per_layer = 5  # Number of epochs to train each layer

    def forward(self, x):
        for rbm in self.layers:
            x = rbm(x)
        return x

    def training_step(self, batch, batch_idx):
        x, _ = batch  # Assuming batch contains (data, labels)
        x = x.to(self.device)
        temp_data = x.clone() 
        for rbm_layer in self.layers:
            layer_loss = 0
            layer_loss = rbm_layer.train_epoch(temp_data, self.current_epoch)
            #self.log(f'layer_{self.current_layer}_loss', layer_loss, on_step=True, on_epoch=True, prog_bar=True)
            temp_data = rbm_layer.forward(temp_data)

        return torch.tensor(layer_loss)


    def configure_optimizers(self):
        """No optimizer needed for RBMs (manual weight updates)."""
        return None

class iMDBN(pl.LightningModule):
    def __init__(self, layer_sizes_img, layer_sizes_txt, joint_layer_size, learning_rate=0.001):
        super(iMDBN, self).__init__()
        self.learning_rate = learning_rate

        # Separate iDBNs for each modality
        self.image_idbn = iDBN(layer_sizes_img)
        self.text_idbn = iDBN(layer_sizes_txt)

        # Joint representation layer
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.W_img_to_joint = nn.Parameter(
            torch.randn(layer_sizes_img[-1], joint_layer_size, device=device) / math.sqrt(layer_sizes_img[-1])
        )
        self.W_txt_to_joint = nn.Parameter(
            torch.randn(layer_sizes_txt[-1], joint_layer_size, device=device) / math.sqrt(layer_sizes_txt[-1])
        )
        self.joint_bias = nn.Parameter(torch.zeros(joint_layer_size, device=device))

    def joint_forward(self, img_representation, txt_representation):
        """Forward pass through the joint layer"""
        return sigmoid(
            torch.matmul(img_representation, self.W_img_to_joint) +
            torch.matmul(txt_representation, self.W_txt_to_joint) + self.joint_bias
        )

    def forward(self, img_data, txt_data):
        """Forward pass for the full Multimodal iDBN"""
        img_rep = self.image_idbn(img_data)
        txt_rep = self.text_idbn(txt_data)
        return self.joint_forward(img_rep, txt_rep)

    def training_step(self, batch, batch_idx):
        """Performs one training step"""
        img_data, txt_data, joint_targets = batch
        img_rep = self.image_idbn(img_data)
        txt_rep = self.text_idbn(txt_data)
        joint_rep = self.joint_forward(img_rep, txt_rep)

        loss = nn.MSELoss()(joint_rep, joint_targets)
        self.log("train_loss", loss)
        return loss

    def configure_optimizers(self):
        """Optimizer setup"""
        return torch.optim.Adam(self.parameters(), lr=self.learning_rate)


# **Data Preparation**
def prepare_dataloader():
    """Simulates multimodal data (images & text)"""
    img_data = torch.randn(100, 3857)  # Random image features
    txt_data = torch.randn(100, 2000)  # Random text features
    joint_targets = torch.randn(100, 256)  # Joint representation targets

    dataset = TensorDataset(img_data, txt_data, joint_targets)
    return DataLoader(dataset, batch_size=32, shuffle=True)


# **Training**
if __name__ == "__main__":
    # Define model
    model = iMDBN(
        layer_sizes_img=[3857, 1024, 512],
        layer_sizes_txt=[2000, 1024, 512],
        joint_layer_size=256,
        learning_rate=0.001
    )

    # Train using Lightning Trainer
    trainer = pl.Trainer(max_epochs=20, gpus=1 if torch.cuda.is_available() else 0)
    dataloader = prepare_dataloader()
    trainer.fit(model, dataloader)
