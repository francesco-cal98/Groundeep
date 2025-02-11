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

class RBM(pl.LightningModule):
    def __init__(self, num_visible, num_hidden, learning_rate, weight_penalty, momentum, dynamic_lr=False, final_momentum=0.97):
        super(RBM, self).__init__()
        self.save_hyperparameters()
        
        self.num_visible = num_visible
        self.num_hidden = num_hidden
        self.lr = learning_rate
        self.weight_decay = weight_penalty
        self.momentum = momentum
        self.dynamic_lr = dynamic_lr
        self.final_momentum = final_momentum

        # Weights and biases
        self.W = nn.Parameter(torch.randn(num_visible, num_hidden) / math.sqrt(num_visible))
        self.hid_bias = nn.Parameter(torch.zeros(num_hidden))
        self.vis_bias = nn.Parameter(torch.zeros(num_visible))

        # Gradients momentum
        self.register_buffer('W_momentum', torch.zeros_like(self.W))
        self.register_buffer('hid_bias_momentum', torch.zeros_like(self.hid_bias))
        self.register_buffer('vis_bias_momentum', torch.zeros_like(self.vis_bias))

    def forward(self, v):
        return sigmoid(torch.matmul(v, self.W) + self.hid_bias)

    def backward(self, h):
        return sigmoid(torch.matmul(h, self.W.T) + self.vis_bias)

    def training_step(self, batch, batch_idx):
        batch_data = batch
        batch_size = batch_data.size(0)
        CD = 1  # You might want to make this configurable

        # Positive phase
        pos_hid_probs = self(batch_data)
        pos_hid_states = (pos_hid_probs > torch.rand_like(pos_hid_probs)).float()
        pos_assoc = torch.matmul(batch_data.T, pos_hid_probs)

        # Negative phase
        neg_data = batch_data
        for _ in range(CD):
            neg_vis_probs = self.backward(pos_hid_states)
            neg_data = (neg_vis_probs > torch.rand_like(neg_vis_probs)).float()
            neg_hid_probs = self(neg_data)
            pos_hid_states = (neg_hid_probs > torch.rand_like(neg_hid_probs)).float()
        neg_assoc = torch.matmul(neg_data.T, neg_hid_probs)

        # Compute gradients
        W_grad = (pos_assoc - neg_assoc) / batch_size - self.weight_decay * self.W
        hid_bias_grad = (pos_hid_probs.sum(0) - neg_hid_probs.sum(0)) / batch_size
        vis_bias_grad = (batch_data.sum(0) - neg_data.sum(0)) / batch_size

        # Update parameters
        self.W_momentum = self.W_momentum * self.momentum + W_grad
        self.hid_bias_momentum = self.hid_bias_momentum * self.momentum + hid_bias_grad
        self.vis_bias_momentum = self.vis_bias_momentum * self.momentum + vis_bias_grad

        self.W.add_(self.W_momentum)
        self.hid_bias.add_(self.hid_bias_momentum)
        self.vis_bias.add_(self.vis_bias_momentum)

        # Compute loss
        loss = torch.sum((batch_data - neg_vis_probs) ** 2) / batch_size
        self.log('train_loss', loss)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.SGD(self.parameters(), lr=self.lr)
        if self.dynamic_lr:
            scheduler = torch.optim.lr_scheduler.LambdaLR(
                optimizer, lr_lambda=lambda epoch: 1 / (1 + 0.01 * epoch)
            )
            return [optimizer], [scheduler]
        return optimizer

    def on_train_epoch_start(self):
        if self.current_epoch > 5:
            self.momentum = self.final_momentum

# Iterative DBN class
class iDBN(pl.LightningModule):
    def __init__(self, params, layer_sizes=[500, 500]):
        super(iDBN, self).__init__()
        self.save_hyperparameters()
        
        self.layers = nn.ModuleList()
        for i in range(len(layer_sizes) - 1):
            self.layers.append(RBM(
                num_visible=layer_sizes[i], 
                num_hidden=layer_sizes[i + 1], 
                learning_rate=params['LEARNING_RATE'],
                weight_penalty=params['WEIGHT_PENALTY'],
                momentum=params['INIT_MOMENTUM'],
                dynamic_lr=params['LEARNING_RATE_DYNAMIC'],
                final_momentum=params['FINAL_MOMENTUM']
            ))

    def forward(self, x):
        for rbm in self.layers:
            x = rbm(x)
        return x

    def training_step(self, batch, batch_idx):
        for layer,rbm in enumerate(self.layers):
            rbm.train()
            rbm.training_step(batch, batch_idx)
        
        x, _ = batch  # Assuming the batch contains both data and labels
        loss = 0
        temp_data = x.clone()
        
        for rbm_layer in self.layers:
            rbm_loss = rbm_layer.training_step({'batch': temp_data}, batch_idx)
            loss += rbm_loss
            temp_data = rbm_layer(temp_data)

        self.log('train_loss', loss)
        return loss

    

    def on_train_epoch_end(self):
        for rbm in self.layers:
            rbm.on_train_epoch_end()# **Multimodal iDBN in PyTorch Lightning**

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
