import os
import numpy as np

# Directories containing the saved SVD and logistic regression model files
input_directory = '/scratch/user/sohamghosh/SAE_toxicity/embeddings_code/embeddings_data'
svd_directory = os.path.join(input_directory, 'svd_results')
logreg_directory = os.path.join(input_directory, 'logistic_regression_regularized')
output_directory = os.path.join(input_directory, 'cosine_similarity_results')
os.makedirs(output_directory, exist_ok=True)

# Number of layers
layers = 24
c_values = [1, 10, 100, 1000, 10000, 100000, 1000000]

# Initialize a dictionary to store cosine similarity results for each layer
# cosine_similarities = {}

# Loop over each layer
for c_val in c_values:  # Loop over different c values
    for layer_idx in range(layers):
        # Load Vt (principal components) for this layer
        Vt_path = os.path.join(svd_directory, f'layer_{layer_idx}_Vt.npy')
        Vt = np.load(Vt_path)  # shape: (num_components, feature_dim)
    
        # Load logistic regression coefficients for this layer
        coef_path = os.path.join(logreg_directory, f'logistic_regression_layer_{layer_idx}_C_{float(c_val)}.npy')
        coef = np.load(coef_path)  # shape: (1, feature_dim)
    
        # Normalize the coefficients vector (L2 norm)
        coef_norm = coef / np.linalg.norm(coef)
    
        # Compute the cosine similarity between each principal component and the coefficient vector
        # This is equivalent to Vt @ coef_norm.T where coef_norm is treated as a column vector
        cosine_similarity = Vt @ coef_norm.T  # shape: (num_components, 1)
    
        # Store the cosine similarity result for this layer
        #cosine_similarities[f'layer_{layer_idx}_C_{c_val}'] = cosine_similarity
    
        # Save the cosine similarity as an .npy file
        cosine_similarity_path = os.path.join(output_directory, f'layer_{layer_idx}_C_{c_val}_cosine_similarity.npy')
        np.save(cosine_similarity_path, cosine_similarity)
    
        print(f"Saved cosine similarity for layer {layer_idx} and C {c_val} at {cosine_similarity_path}")

