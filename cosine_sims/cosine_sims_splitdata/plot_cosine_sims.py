import os
import numpy as np
import matplotlib.pyplot as plt

# Directory containing saved cosine similarity results
input_directory = '/scratch/user/sohamghosh/SAE_toxicity/embeddings_code/embeddings_data/cosine_similarity_results'
output_directory = os.path.join(input_directory, 'cosine_similarity_plots')
os.makedirs(output_directory, exist_ok=True)

# Number of layers
layers = 24
c_values = [1, 10, 100, 1000, 10000, 100000, 1000000]

# Loop over each layer to generate plots
for c_val in c_values:
    for layer_idx in range(layers):
        # Load cosine similarity data for this layer
        cosine_similarity_path = os.path.join(input_directory, f'layer_{layer_idx}_C_{c_val}_cosine_similarity.npy')
        cosine_similarity = np.load(cosine_similarity_path).flatten()
    
        # Create a plot for the cosine similarity values
        plt.figure(figsize=(8, 6))
        plt.plot(cosine_similarity, marker='o', linestyle='-', markersize=4)
        plt.xlabel('Principal Component Index')
        plt.ylabel('Cosine Similarity')
        plt.title(f'Layer {layer_idx} C {c_val} Cosine Similarities with Logistic Regression Coefficients')
        plt.grid(True)
    
        # Save the plot for this layer
        plot_path = os.path.join(output_directory, f'layer_{layer_idx}_C_{c_val}_cosine_similarity_plot.png')
        plt.savefig(plot_path)
        plt.close()  # Close the plot to free memory
    
        print(f"Saved cosine similarity plot for layer {layer_idx} C {c_val} at {plot_path}")
