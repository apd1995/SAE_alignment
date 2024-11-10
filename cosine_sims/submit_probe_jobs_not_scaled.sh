#!/bin/bash
# submit_logistic_jobs.sh

# Set job parameters
#SBATCH --job-name=logistic_layer_job
#SBATCH --output=/scratch/users/apd1995/SAE_toxicity/logs/layer_%a.out
#SBATCH --error=/scratch/users/apd1995/SAE_toxicity/logs/layer_%a.err
#SBATCH --array=0-23  # 24 jobs for 24 layers
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=100G
#SBATCH --time=30:00
#SBATCH --partition=donoho,owners,hns,stat,normal

# Set the layer index for each job in the array
export LAYER_IDX=$SLURM_ARRAY_TASK_ID

# Run the Python script for logistic regression training on the given layer
python train_probe_not_scaled.py
