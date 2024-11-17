import os
import torch
import numpy as np
from sklearn.preprocessing import StandardScaler

# Directories containing the saved .pt files for each layer
input_directory = '/scratch/users/apd1995/SAE_toxicity/embeddings_data'
output_directory = os.path.join(input_directory, 'svd_results')
os.makedirs(output_directory, exist_ok=True)

# Layer index (to be set as an environment variable for each Slurm job)
layer_idx = int(os.environ['LAYER_IDX'])

# Load and compute mean for safe data
safe_file_path = os.path.join(input_directory, f"gpt2-medium_embeddings_safe_layer{layer_idx}.pt")
safe_tensor = torch.load(safe_file_path)
safe_mean = safe_tensor.mean(dim=1).cpu().numpy()

# Load and compute mean for unsafe data
unsafe_file_path = os.path.join(input_directory, f"gpt2-medium_embeddings_unsafe_layer{layer_idx}.pt")
unsafe_tensor = torch.load(unsafe_file_path)
unsafe_mean = unsafe_tensor.mean(dim=1).cpu().numpy()

# Concatenate safe and unsafe data for joint analysis on this layer
layer_data = np.concatenate([safe_mean, unsafe_mean], axis=0)

# Apply StandardScaler to center and scale the data
scaler = StandardScaler()
scaled_data = scaler.fit_transform(layer_data)

# Perform Singular Value Decomposition (SVD)
U, S, Vt = np.linalg.svd(scaled_data, full_matrices=False)

# Save the singular values, U, and Vt
singular_values_path = os.path.join(output_directory, f'layer_{layer_idx}_singular_values.npy')
left_singular_vectors_path = os.path.join(output_directory, f'layer_{layer_idx}_U.npy')
right_singular_vectors_path = os.path.join(output_directory, f'layer_{layer_idx}_Vt.npy')

np.save(singular_values_path, S)
np.save(left_singular_vectors_path, U)
np.save(right_singular_vectors_path, Vt)

print(f"Saved SVD results for layer {layer_idx}: singular values, U, and Vt")
