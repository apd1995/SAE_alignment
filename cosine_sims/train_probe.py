import os
import torch
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

# Directories containing the saved .pt files for each layer
input_directory = '/scratch/users/apd1995/SAE_toxicity/embeddings_data'
output_directory = os.path.join(input_directory, 'logistic_regression_models')
os.makedirs(output_directory, exist_ok=True)

# Layer index (to be set as an environment variable for each Slurm job)
layer_idx = int(os.environ['LAYER_IDX'])

# Load and compute mean for safe data
safe_file_path = os.path.join(input_directory, f"gpt2-medium_embeddings_safe_layer{layer_idx}.pt")
safe_tensor = torch.load(safe_file_path, weights_only = True)
safe_mean = safe_tensor.mean(dim=1).cpu().numpy()

# Load and compute mean for unsafe data
unsafe_file_path = os.path.join(input_directory, f"gpt2-medium_embeddings_unsafe_layer{layer_idx}.pt")
unsafe_tensor = torch.load(unsafe_file_path, weights_only = True)
unsafe_mean = unsafe_tensor.mean(dim=1).cpu().numpy()

# Concatenate safe and unsafe data and assign labels (0 for safe, 1 for unsafe)
layer_data = np.concatenate([safe_mean, unsafe_mean], axis=0)
labels = np.concatenate([np.zeros(safe_mean.shape[0]), np.ones(unsafe_mean.shape[0])])

# Apply StandardScaler to center and scale the data
scaler = StandardScaler()
scaled_data = scaler.fit_transform(layer_data)

# Fit logistic regression
logistic_model = LogisticRegression(max_iter=10000, fit_intercept=False)
logistic_model.fit(scaled_data, labels)

# Save the logistic regression model
coef_path = os.path.join(output_directory, f'logistic_regression_layer_{layer_idx}.npy')
np.save(coef_path, logistic_model.coef_)

print(f"Saved logistic regression model for layer {layer_idx} at {coef_path}")
