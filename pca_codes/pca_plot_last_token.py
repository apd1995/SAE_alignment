import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# Directories containing the saved .pt files for each layer
input_directory = '/scratch/users/apd1995/SAE_toxicity/embeddings_data'
output_directory = os.path.join(input_directory, 'pca_plots_1_2_last_token')
os.makedirs(output_directory, exist_ok=True)

layers = 24  # Number of layers in the model

# Loop over each layer
for layer_idx in range(layers):
    # Load and compute mean for safe data
    safe_file_path = os.path.join(input_directory, f"gpt2-medium_embeddings_safe_layer{layer_idx}.pt")
    safe_tensor = torch.load(safe_file_path)
    safe_tensor_max_token = safe_tensor.shape[1]
    safe_mean = safe_tensor[:,safe_tensor_max_token-1,:].cpu().numpy()

    # Load and compute mean for unsafe data
    unsafe_file_path = os.path.join(input_directory, f"gpt2-medium_embeddings_unsafe_layer{layer_idx}.pt")
    unsafe_tensor = torch.load(unsafe_file_path)
    unsafe_tensor_max_token = unsafe_tensor.shape[1]
    unsafe_mean = unsafe_tensor[:,unsafe_tensor_max_token-1,:].cpu().numpy()  # Take the mean along dim=1 and convert to numpy

    # Concatenate safe and unsafe data for joint PCA on this layer
    layer_data = np.concatenate([safe_mean, unsafe_mean], axis=0)
    
    # Apply StandardScaler to center and scale the data
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(layer_data)
    
    # Perform PCA to reduce to two components for this layer
    pca = PCA(n_components=3)
    principal_components = pca.fit_transform(scaled_data)
    
    # Separate the principal components for safe and unsafe data
    n_safe = safe_mean.shape[0]
    safe_pc = principal_components[:n_safe]
    unsafe_pc = principal_components[n_safe:]
    
    # Plot the data points along the second and third principal components for this layer
    plt.figure(figsize=(8, 6))
    plt.scatter(safe_pc[:, 0], safe_pc[:, 1], color='blue', label='Safe', alpha=0.3)
    plt.scatter(unsafe_pc[:, 0], unsafe_pc[:, 1], color='red', label='Unsafe', alpha=0.3)
    plt.xlabel('Principal Component 1')
    plt.ylabel('Principal Component 2')
    plt.title(f'PCA of Layer {layer_idx} Safe and Unsafe Data')
    plt.legend()
    plt.grid(True)
    
    # Save the plot for this layer
    plot_path = os.path.join(output_directory, f'layer_{layer_idx}_pca_plot.png')
    plt.savefig(plot_path)
    plt.close()  # Close the plot to free memory

    print(f"Saved PCA plot for layer {layer_idx} at {plot_path}")
