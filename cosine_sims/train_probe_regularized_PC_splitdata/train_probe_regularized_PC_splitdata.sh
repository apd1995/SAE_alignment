#!/bin/bash
# submit_logistic_jobs.sh
# Set job parameters
#SBATCH --job-name=logistic_layer_job
#SBATCH --output=/scratch/user/sohamghosh/SAE_toxicity/embeddings_code/logs/logistic_layer_%j_%a.out
#SBATCH --error=/scratch/user/sohamghosh/SAE_toxicity/embeddings_code/logs/logistic_layer_%j_%a.err
#SBATCH --array=0-167  # 24 layers * 7 C-values = 168 jobs
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=50G
#SBATCH --time=05:00:00
#SBATCH --partition=long,xlong

# Define the range of C-values (logarithmic scale)
C_VALUES=(1 10 100 1000 10000 100000 1000000)

# Calculate the layer index and C index
LAYER_IDX=$((SLURM_ARRAY_TASK_ID / 7))  # Integer division to get the layer index
C_IDX=$((SLURM_ARRAY_TASK_ID % 7))     # Modulus to get the C index

# Get the actual C-value
C_VAL=${C_VALUES[C_IDX]}

# Set environment variables
export LAYER_IDX
export C_VAL

# Print debug information
echo "Running job for layer $LAYER_IDX with C=$C_VAL"

# Run the Python script with the layer index and C-value as arguments
python train_probe_regularized.py
