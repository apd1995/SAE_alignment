#!/bin/bash

# Define the sequences
lambda_recon_values=$(seq 0.1 0.1 0.3)  # From 0.01 to 10.0 in steps of 0.01
lambda_energy_values=$(seq 0.1 0.1 0.3)  # From 0.01 to 10 in steps of 0.01
lambda_sparsity_values=$(seq 0.1 0.1 0.3)  # From 0.01 to 10 in steps of 0.01

# Create directories for configs and logs
CONFIG_DIR="/scratch/users/apd1995/SAE_toxicity/configs_SAE"
LOG_DIR="/scratch/users/apd1995/SAE_toxicity/logs_SAE"
mkdir -p $CONFIG_DIR $LOG_DIR

# Create an array from each sequence
recon_array=($lambda_recon_values)
energy_array=($lambda_energy_values)
sparsity_array=($lambda_sparsity_values)

# Loop over all sequences simultaneously
for i in "${!recon_array[@]}"; do
    for j in "${!energy_array[@]}"; do
        for k in "${!sparsity_array[@]}"; do
          lambda_recon=${recon_array[$i]}
          lambda_energy=${energy_array[$j]}
          lambda_sparsity=${sparsity_array[$k]}
        
          # Submit the job to the Slurm scheduler
          JOB_ID=$(sbatch <<EOT | awk '{print $4}'
#!/bin/bash
#SBATCH --job-name=SAE_training
#SBATCH --output=${LOG_DIR}/SAE_%j.out
#SBATCH --error=${LOG_DIR}/SAE_%j.err
#SBATCH --time=01:00:00
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=100G
#SBATCH --partition=donoho,gpu,owners,hns,stat,normal

# Create a unique JSON configuration file using the job ID
CONFIG_NAME="${CONFIG_DIR}/config_${SLURM_JOB_ID}.json"

cat <<JSON_EOF > \$CONFIG_NAME
{
  "model_name": "gpt2-medium",
  "learning_rate": 1,
  "batch_size": 64,
  "epochs": 1000,
  "save_interval": 50,
  "hidden_size": 128,
  "lambda_recon": ${lambda_recon},
  "lambda_energy": ${lambda_energy},
  "lambda_sparsity": ${lambda_sparsity},
  "sparsity_param": 0.05,
  "sparsity_loss": "kl"
}
JSON_EOF

# Run the Python script with the generated config
export WANDB_API_KEY=""
python train_SAE_kl.py --config \$CONFIG_NAME
EOT
)

          # Log the submitted job and its configuration
          echo "Submitted job ID ${JOB_ID}"
        done
    done
done
