import wandb
import torchvision.utils as vutils
import torch

def log_reconstructions_to_wandb(original, reconstruction, step=0, num_images=8, name="reconstruction_grid"):
    """
    Log a grid of original and reconstructed images to Weights & Biases.
    """
    orig = torch.tensor((original[:num_images]).reshape(num_images,100,100)).unsqueeze(1)
    recon = torch.tensor((reconstruction[:num_images]).reshape(num_images,100,100)).unsqueeze(1)

    combined = torch.cat([val for pair in zip(orig, recon) for val in pair], dim=0)

    grid = vutils.make_grid(combined.unsqueeze(1), nrow=2, normalize=True)
    wandb.log({name: [wandb.Image(grid, caption=name)]})
