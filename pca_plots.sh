#!/bin/bash
#SBATCH --job-name=pca_plots
#SBATCH --output=logs/pca_%j.out
#SBATCH --error=logs/pca_%j.err
#SBATCH --time=02:00:00          # Adjust time as needed
#SBATCH --partition=donoho,hns,stat,owners,normal           # Use GPU partition if needed
#SBATCH --mem=100G                 # Adjust memory as needed
#SBATCH --cpus-per-task=4         # Number of CPU cores per job

# Run the Python script with the provided filename argument
python pca_plots.py