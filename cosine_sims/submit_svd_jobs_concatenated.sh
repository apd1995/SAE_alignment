#!/bin/bash

# Set job parameters
#SBATCH --job-name=svd_layer_job
#SBATCH --output=/scratch/users/apd1995/SAE_toxicity/logs/layer_%j_%a.out
#SBATCH --error=/scratch/users/apd1995/SAE_toxicity/logs/layer_%j_%a.err
#SBATCH --array=0-23  # 24 jobs for 24 layers
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=50G
#SBATCH --time=01:00:00
#SBATCH --partition=donoho,hns,stat,owners,normal           # Use GPU partition if needed

# Set the layer index for each job in the array
export LAYER_IDX=$SLURM_ARRAY_TASK_ID

# Run the Python script for the given layer
python save_svd.py
