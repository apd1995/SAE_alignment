import os
import torch
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

# Directories containing the saved .pt files for each layer
input_directory = '/scratch/user/sohamghosh/SAE_toxicity/embeddings_code/embeddings_data/'
output_directory1 = os.path.join(input_directory, 'logistic_regression_regularized')
os.makedirs(output_directory1, exist_ok=True)
output_directory2 = os.path.join(input_directory, 'svd_results')
os.makedirs(output_directory2, exist_ok=True)

# Layer index and C (to be set as an environment variable for each Slurm job)
layer_idx = int(os.environ['LAYER_IDX'])
C_val = float(os.environ['C_VAL'])

# Load and compute mean for safe data
safe_file_path = os.path.join(input_directory, f"gpt2-medium_embeddings_safe_layer{layer_idx}.pt")
safe_tensor = torch.load(safe_file_path, weights_only = True)
safe_mean = safe_tensor.cpu().numpy()
safe_mean1, safe_mean2 = train_test_split(safe_mean, test_size=0.5)

# Load and compute mean for unsafe data
unsafe_file_path = os.path.join(input_directory, f"gpt2-medium_embeddings_unsafe_layer{layer_idx}.pt")
unsafe_tensor = torch.load(unsafe_file_path, weights_only = True)
unsafe_mean = unsafe_tensor.cpu().numpy()
unsafe_mean1, unsafe_mean2 = train_test_split(unsafe_mean, test_size=0.5)

# Concatenate safe and unsafe data and assign labels (0 for safe, 1 for unsafe)
layer_data_probe = np.concatenate([safe_mean1, unsafe_mean1], axis=0)
labels_probe = np.concatenate([np.zeros(safe_mean1.shape[0]), np.ones(unsafe_mean1.shape[0])])

# Apply StandardScaler to center and scale the data
scaler = StandardScaler()
scaled_data_probe = scaler.fit_transform(layer_data_probe)

# Fit logistic regression
logistic_model = LogisticRegression(max_iter=10000, fit_intercept=True, C = C_val)
logistic_model.fit(scaled_data_probe, labels_probe)

# Save the logistic regression model
coef_path = os.path.join(output_directory1, f'logistic_regression_layer_{layer_idx}_C_{C_val}.npy')
np.save(coef_path, logistic_model.coef_)

print(f"Saved logistic regression model for layer {layer_idx} with C {C_val} at {coef_path}")

# Concatenate safe and unsafe data and assign labels (0 for safe, 1 for unsafe)
layer_data_pca = np.concatenate([safe_mean2, unsafe_mean2], axis=0)

# Apply StandardScaler to center and scale the data
scaler = StandardScaler()
scaled_data_pca = scaler.fit_transform(layer_data_pca)

# Perform Singular Value Decomposition (SVD)
U, S, Vt = np.linalg.svd(scaled_data_pca, full_matrices=False)

# Save the singular values, U, and Vt
singular_values_path = os.path.join(output_directory2, f'layer_{layer_idx}_singular_values.npy')
left_singular_vectors_path = os.path.join(output_directory2, f'layer_{layer_idx}_U.npy')
right_singular_vectors_path = os.path.join(output_directory2, f'layer_{layer_idx}_Vt.npy')

np.save(singular_values_path, S)
np.save(left_singular_vectors_path, U)
np.save(right_singular_vectors_path, Vt)

print(f"Saved SVD results for layer {layer_idx}: singular values, U, and Vt")


