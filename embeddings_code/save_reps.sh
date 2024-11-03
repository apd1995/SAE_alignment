#!/bin/bash
#SBATCH --job-name=save_reps
#SBATCH --output=logs/save_reps_%j.out
#SBATCH --error=logs/save_reps_%j.err
#SBATCH --time=02:00:00           # Set time as needed
#SBATCH --partition=donoho,gpu,owners,hns,stat            # Use GPU partition
#SBATCH --gres=gpu:1               # Request 1 GPU per job
#SBATCH --mem=50G                  # Adjust memory per job as needed
#SBATCH --cpus-per-task=4          # Number of CPU cores per job


# Run the Python script with arguments for each layer and data type
python save_reps.py $1 $2
