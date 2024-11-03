#!/bin/bash

# Directory containing the .pt files
input_directory="/scratch/users/apd1995/SAE_toxicity/embeddings_data"

# Create a directory for logs if it doesn't exist
mkdir -p logs

# Loop through each .pt file and submit a job
for filename in "$input_directory"/*.pt; do
  file=$(basename "$filename")
  sbatch save_sing_vals.sh "$file"
done
