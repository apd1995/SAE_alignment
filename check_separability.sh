#!/bin/bash
# submit_logistic_jobs.sh

# Set job parameters
#SBATCH --job-name=separability_layerwise
#SBATCH --output=/scratch/users/apd1995/SAE_toxicity/logs/layer_%a.out
#SBATCH --error=/scratch/users/apd1995/SAE_toxicity/logs/layer_%a.err
#SBATCH --array=0-23  # 24 jobs for 24 layers
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=100G
#SBATCH --time=5:00:00
#SBATCH --partition=donoho,owners,hns,stat,normal

# Set the layer index for each job in the array
export LAYER_IDX=$SLURM_ARRAY_TASK_ID

# Run the Python script for checking separability on the given layer
python check_separability.py