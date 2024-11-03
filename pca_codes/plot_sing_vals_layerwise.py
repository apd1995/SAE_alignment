import os
import numpy as np
import matplotlib.pyplot as plt

# Directory containing the saved singular values for each layer
input_directory = '/scratch/users/apd1995/SAE_toxicity/embeddings_data/svd_output'
output_directory = os.path.join(input_directory, 'singval_plots')
os.makedirs(output_directory, exist_ok=True)

layers = 24  # Number of layers in the model

# Loop over each layer to load singular values and plot them
for layer_idx in range(layers):
    # Load singular values for safe data
    safe_file_path = os.path.join(input_directory, f"gpt2-medium_embeddings_safe_layer{layer_idx}_S.npy")
    safe_singular_values = np.load(safe_file_path)/np.sqrt(2872)

    # Load singular values for unsafe data
    unsafe_file_path = os.path.join(input_directory, f"gpt2-medium_embeddings_unsafe_layer{layer_idx}_S.npy")
    unsafe_singular_values = np.load(unsafe_file_path)/np.sqrt(18892)

    # Plot singular values for this layer
    plt.figure(figsize=(8, 6))
    plt.plot(safe_singular_values, label='Safe Data', color='blue', alpha=0.7)
    plt.plot(unsafe_singular_values, label='Unsafe Data', color='red', alpha=0.7)

    # Configure plot for this layer
    plt.xlabel('Singular Value Index')
    plt.ylabel('Singular Value Magnitude')
    plt.title(f'Singular Values of Safe and Unsafe Data - Layer {layer_idx}')
    plt.legend()
    plt.yscale('log')  # Optional: log scale to better capture range of values
    plt.grid(True)

    # Save the plot for this layer
    plot_path = os.path.join(output_directory, f'layer_{layer_idx}_singular_values.png')
    plt.savefig(plot_path)
    plt.close()  # Close the plot to free memory

    print(f"Saved singular value plot for layer {layer_idx} at {plot_path}")
