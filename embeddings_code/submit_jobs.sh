#!/bin/bash

# Directory to store logs
mkdir -p logs

# Number of layers
num_layers=24  # For gpt2-medium

# Submit jobs for each layer and data type
for ((layer_idx=0; layer_idx<num_layers; layer_idx++)); do
  sbatch save_reps.sh "$layer_idx" "safe"
  sbatch save_reps.sh "$layer_idx" "unsafe"
done
