import os
import numpy as np

# Directories containing the saved SVD and logistic regression model files
input_directory = '/scratch/users/apd1995/SAE_toxicity/embeddings_data'
svd_directory = os.path.join(input_directory, 'svd_results')
logreg_directory = os.path.join(input_directory, 'logistic_regression_models')
output_directory = os.path.join(input_directory, 'cosine_similarity_results')
os.makedirs(output_directory, exist_ok=True)

# Number of layers
layers = 24

# Initialize a dictionary to store cosine similarity results for each layer
cosine_similarities = {}

# Loop over each layer
for layer_idx in range(layers):
    # Load Vt (principal components) for this layer
    Vt_path = os.path.join(svd_directory, f'layer_{layer_idx}_Vt.npy')
    Vt = np.load(Vt_path)  # shape: (num_components, feature_dim)

    # Load logistic regression coefficients for this layer
    coef_path = os.path.join(logreg_directory, f'logistic_regression_layer_{layer_idx}.npy')
    coef = np.load(coef_path)  # shape: (1, feature_dim)

    # Normalize the coefficients vector (L2 norm)
    coef_norm = coef / np.linalg.norm(coef)

    # Compute the cosine similarity between each principal component and the coefficient vector
    # This is equivalent to Vt @ coef_norm.T where coef_norm is treated as a column vector
    cosine_similarity = Vt @ coef_norm.T  # shape: (num_components, 1)

    # Store the cosine similarity result for this layer
    cosine_similarities[f'layer_{layer_idx}'] = cosine_similarity

    # Save the cosine similarity as an .npy file
    cosine_similarity_path = os.path.join(output_directory, f'layer_{layer_idx}_cosine_similarity.npy')
    np.save(cosine_similarity_path, cosine_similarity)

    print(f"Saved cosine similarity for layer {layer_idx} at {cosine_similarity_path}")

# Optionally, save all cosine similarities in a single file if needed
all_cosine_similarities_path = os.path.join(output_directory, 'all_layers_cosine_similarities.npy')
np.save(all_cosine_similarities_path, cosine_similarities)
print(f"Saved all layers' cosine similarities at {all_cosine_similarities_path}")
