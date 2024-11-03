#!/bin/bash
#SBATCH --job-name=svd_computation
#SBATCH --output=logs/svd_%j.out
#SBATCH --error=logs/svd_%j.err
#SBATCH --time=01:00:00          # Adjust time as needed
#SBATCH --partition=donoho,hns,stat,owners,normal           # Use GPU partition if needed
#SBATCH --mem=100G                 # Adjust memory as needed
#SBATCH --cpus-per-task=4         # Number of CPU cores per job

# Run the Python script with the provided filename argument
srun python save_sing_vals.py "$1"
