#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Nov  2 09:48:52 2024

@author: apratimdey
"""

import os
import sys
import torch
import numpy as np

# Directory containing .pt files
input_directory = '/scratch/users/apd1995/SAE_toxicity/embeddings_data'

# Directory to save the .npy files
output_directory = os.path.join(input_directory, 'svd_output')
os.makedirs(output_directory, exist_ok=True)  # Create the directory if it doesn't exist

# Get filename from command-line argument
filename = sys.argv[1]
file_path = os.path.join(input_directory, filename)

# Load the 3D tensor
tensor = torch.load(file_path)

# Take the mean along axis 1
tensor_mean = tensor.mean(dim=1)

# Center the matrix by subtracting the mean of each column
column_means = tensor_mean.mean(dim=0, keepdim=True)
centered_tensor = tensor_mean - column_means

# Perform SVD on the 2D matrix
U, S, V = torch.linalg.svd(centered_tensor)

# Convert to numpy arrays
U_np = U.cpu().numpy()
S_np = S.cpu().numpy()
V_np = V.cpu().numpy()

# Save U, S, V as .npy files with the same base filename
base_filename = os.path.splitext(filename)[0]
np.save(os.path.join(output_directory, f"{base_filename}_U.npy"), U_np)
np.save(os.path.join(output_directory, f"{base_filename}_S.npy"), S_np)
np.save(os.path.join(output_directory, f"{base_filename}_V.npy"), V_np)

print(f"Processed {filename}: SVD components saved in '{output_directory}'")
