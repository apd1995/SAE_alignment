import numpy as np
from scipy.optimize import linprog
import pandas as pd
import os

# Directories containing the saved .pt files for each layer
input_directory = '/scratch/users/apd1995/SAE_toxicity/embeddings_data/save_matrix_data'
output_directory = os.path.join(input_directory, 'separability')
os.makedirs(output_directory, exist_ok=True)

# Layer index (to be set as an environment variable for each Slurm job)
layer_idx = int(os.environ['LAYER_IDX'])

# Read the dataset and save variables
file_path = os.path.join(input_directory, f'layer_data_with_labels_{layer_idx}.csv')
dat = pd.read_csv(file_path, header=None)
X = dat.iloc[:, 1:]
y = dat.iloc[:, 0]

# Record feature dimensionality and number of classes
num_classes = len(np.unique(y))
num_features = X.shape[1]
num_samples = len(y)

separable = True


# Assign labels: +1 for class k=0, -1 for others
y_k = np.where(y == 0, 1, -1)

# Objective function (zero vector)
c = np.zeros(num_features + 1)

# Inequality constraints: y_i (w^T x_i + b) >= 1
A = -y_k[:, np.newaxis] * np.hstack((X, np.ones((num_samples, 1))))
b_ineq = -np.ones(num_samples)

# Bounds for variables (no bounds)
bounds = [(None, None)] * (num_features + 1)

# Solve LP
res = linprog(c, A_ub=A, b_ub=b_ineq, bounds=bounds, method='highs')

if not res.success:
    print(f"Data is not linearly separable in layer {layer_idx}.")
    separable = False

if separable:
    print("Data is linearly separable.")
else:
    print("Data is not linearly separable.")
