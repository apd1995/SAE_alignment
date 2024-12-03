#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Sep 28 11:46:50 2024

@author: apratimdey
"""

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
import wandb  # Weights and Biases
import numpy as np
import logging
# Set up logging configuration
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
import os
from datetime import datetime
import json

# Create a folder with the current datetime as the name
datetime_folder = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
checkpoint_dir = os.path.join('checkpoints', datetime_folder)

# Ensure the checkpoint folder exists
os.makedirs(checkpoint_dir, exist_ok=True)

config_dict={
    "model_name": 'gpt2-medium',
    "learning_rate": 1,
    "batch_size": 64,
    "epochs": 2000,
    "save_interval": 50,
    "hidden_size": 2,
    "lambda_recon": 0.5,
    "lambda_energy": 0.5,
    "lambda_sparsity": 0.5,
    "sparsity_param": 0.05,
    "sparsity_loss": "kl"
}

# Save the configuration to a JSON file in the checkpoint directory
config_file_path = os.path.join(checkpoint_dir, 'config.json')
with open(config_file_path, 'w') as f:
    json.dump(config_dict, f, indent=4)
    
# Initialize Weights and Biases project
wandb.init(project="SAE_toxicity", config=config_dict, entity="SAE_alignment", resume="allow")

# Get the config parameters from wandb
config = wandb.config

# Model name that saved representations
model_name = config.model_name

# Set device (use GPU if available, else CPU)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Define the Sparse Autoencoder
class SparseAutoencoder(nn.Module):
    def __init__(self, input_size, hidden_size, sparsity_param, lambda_sparsity):
        super(SparseAutoencoder, self).__init__()
        self.encoder = nn.Linear(input_size, hidden_size)
        self.decoder = nn.Linear(hidden_size, input_size)
        self.sparsity_param = sparsity_param  # Target sparsity (e.g., 0.05)
        self.lambda_sparsity = lambda_sparsity  # Regularization strength for sparsity

    def forward(self, x):
        z = torch.sigmoid(self.encoder(x))  # Latent space
        x_hat = torch.sigmoid(self.decoder(z))  # Reconstructed output
        return x_hat, z

    def kl_divergence(self, z):
        rho_hat = torch.mean(z, dim=0)  # Average activation of hidden neurons
        kl_loss = self.sparsity_param * torch.log(self.sparsity_param / torch.clamp(rho_hat, min=1e-10)) + \
                  (1 - self.sparsity_param) * torch.log((1 - self.sparsity_param) / torch.clamp(1 - rho_hat, min=1e-10))
        return kl_loss.sum()

# Energy Distance calculation
def energy_distance(safe_embeddings, transformed_unsafe_embeddings):
    def pairwise_distances(x, y):
        return torch.cdist(x, y, p=2)  # Euclidean distance
    
    # Compute the pairwise distances
    X_Y = pairwise_distances(safe_embeddings, transformed_unsafe_embeddings).mean()
    X_X = pairwise_distances(safe_embeddings, safe_embeddings).mean()
    Y_Y = pairwise_distances(transformed_unsafe_embeddings, transformed_unsafe_embeddings).mean()
    
    # Energy distance formula
    energy_dist = 2 * X_Y - X_X - Y_Y
    return energy_dist

# DataLoader setup
def create_dataloaders(safe_embeddings, unsafe_embeddings, batch_size):
    # Convert numpy arrays to PyTorch tensors
    safe_tensor = torch.tensor(safe_embeddings, dtype=torch.float32)
    unsafe_tensor = torch.tensor(unsafe_embeddings, dtype=torch.float32)
    
    # Create datasets
    safe_dataset = TensorDataset(safe_tensor)
    unsafe_dataset = TensorDataset(unsafe_tensor)
    
    # Create DataLoaders
    safe_loader = DataLoader(safe_dataset, batch_size=batch_size, shuffle=True)
    unsafe_loader = DataLoader(unsafe_dataset, batch_size=batch_size, shuffle=True)
    
    return safe_loader, unsafe_loader

# Load embeddings from .pt files
safe_embeddings = torch.load(f'/scratch/user/sohamghosh/SAE_toxicity/embeddings_code/embeddings_data/{model_name}_embeddings_safe_layer20.pt', weights_only = True)
#safe_embeddings = torch.mean(safe_embeddings, dim = 1)
unsafe_embeddings = torch.load(f'/scratch/user/sohamghosh/SAE_toxicity/embeddings_code/embeddings_data/{model_name}_embeddings_unsafe_layer20.pt', weights_only = True)
#unsafe_embeddings = torch.mean(unsafe_embeddings, dim = 1)

# Hyperparameters from wandb
input_size = safe_embeddings.shape[1]
hidden_size = config.hidden_size
sparsity_param = config.sparsity_param
learning_rate = config.learning_rate
batch_size = config.batch_size
epochs = config.epochs
save_interval = config.save_interval
lambda_recon = config.lambda_recon
lambda_energy = config.lambda_energy
lambda_sparsity = config.lambda_sparsity

# Create the model
model = SparseAutoencoder(input_size=input_size, hidden_size=hidden_size, 
                          sparsity_param=sparsity_param, lambda_sparsity=lambda_sparsity).to(device)
optimizer = optim.Adam(model.parameters(), lr=learning_rate)
#optimizer = optim.RMSprop(model.parameters(), lr=learning_rate)
reconstruction_loss_fn = nn.MSELoss()

# Create DataLoaders
safe_loader, unsafe_loader = create_dataloaders(safe_embeddings, unsafe_embeddings, batch_size)

# Training loop
for epoch in range(epochs):
    model.train()

    total_loss = 0
    total_recon_loss = 0
    total_energy_loss = 0
    total_sparsity_loss = 0

    # Iterate over both safe and unsafe DataLoaders in parallel
    for (safe_batch, ), (unsafe_batch, ) in zip(safe_loader, unsafe_loader):
        
        # Move batches to the device (GPU or CPU)
        safe_batch = safe_batch.to(device)
        unsafe_batch = unsafe_batch.to(device)
        
        # Forward pass for unsafe embeddings
        unsafe_reconstructed, z_unsafe = model(unsafe_batch)
        
        # Forward pass for safe embeddings (optional, for energy distance computation)
        safe_reconstructed, z_safe = model(safe_batch)

        # Compute reconstruction loss (MSE)
        recon_loss = reconstruction_loss_fn(unsafe_reconstructed, unsafe_batch)
        
        # Compute energy distance loss
        energy_dist_loss = energy_distance(safe_batch, unsafe_reconstructed)

        # Compute sparsity loss (KL Divergence)
        sparsity_loss = model.kl_divergence(z_unsafe)

        # Total loss
        loss = lambda_recon * recon_loss + lambda_energy * energy_dist_loss + lambda_sparsity * sparsity_loss

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Accumulate losses for logging
        total_loss += loss.item()
        total_recon_loss += recon_loss.item()
        total_energy_loss += energy_dist_loss.item()
        total_sparsity_loss += sparsity_loss.item()
        
    # Save the model checkpoint every 50 epochs
    if epoch % save_interval == 0:
        checkpoint_path = f"{checkpoint_dir}/epoch_{epoch}.pt"
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
        }, checkpoint_path)

        logging.info(f"Checkpoint saved at epoch {epoch} in {checkpoint_path}")

    # Log metrics to wandb
    wandb.log({
        "Overall Loss": total_loss / len(unsafe_loader),
        "Reconstruction Loss": total_recon_loss / len(unsafe_loader),
        "Energy Distance Loss": total_energy_loss / len(unsafe_loader),
        "Sparsity Loss": total_sparsity_loss / len(unsafe_loader),
        "Epoch": epoch + 1
    })

    # Print losses every epoch
    logging.info(f'Epoch [{epoch + 1}/{epochs}], Overall Loss: {total_loss / len(unsafe_loader):.4f}, '
          f'Recon Loss: {total_recon_loss / len(unsafe_loader):.4f}, Energy Loss: {total_energy_loss / len(unsafe_loader):.4f}, '
          f'Sparsity Loss: {total_sparsity_loss / len(unsafe_loader):.4f}')

checkpoint_path = f"{checkpoint_dir}/epoch_{epoch}.pt"
torch.save({
    'epoch': epoch,
    'model_state_dict': model.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
}, checkpoint_path)

logging.info(f"Checkpoint saved at epoch {epoch} in {checkpoint_path}")

# Finish the wandb run
wandb.finish()
