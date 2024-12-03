#!/bin/bash

# Define the sequences
lambda_recon_values=$(seq 0.01 0.1 1.0)  # From 0.01 to 1.0 in steps of 0.1
lambda_energy_values=$(seq 0.1 0.2 2.0)  # From 0.1 to 2.0 in steps of 0.2
lambda_sparsity_values=$(seq 0.05 0.1 0.5)  # From 0.05 to 0.5 in steps of 0.1
learn_rate_values="1e-4 1e-3 1e-2 0.1 1"  # Learning rates
hidden_layer_values="2 16 64 128"  # Hidden layer sizes

# Create directories for configs and logs
CONFIG_DIR="/scratch/user/sohamghosh/SAE_toxicity/SAE/configs_SAE"
LOG_DIR="/scratch/user/sohamghosh/SAE_toxicity/SAE/logs_SAE"
mkdir -p $CONFIG_DIR $LOG_DIR

# Convert sequences into arrays
recon_array=($lambda_recon_values)
energy_array=($lambda_energy_values)
sparsity_array=($lambda_sparsity_values)
learn_rate_array=($learn_rate_values)
hidden_layer_array=($hidden_layer_values)

# Initialize a counter for unique filenames
counter=0

# Loop over all combinations of parameters
for lambda_recon in "${recon_array[@]}"; do
  for lambda_energy in "${energy_array[@]}"; do
    for lambda_sparsity in "${sparsity_array[@]}"; do
      for learn_rate in "${learn_rate_array[@]}"; do
        for hidden_layer in "${hidden_layer_array[@]}"; do

          # Increment the counter
          counter=$((counter + 1))

          # Generate a unique config filename
          CONFIG_NAME="${CONFIG_DIR}/config_${counter}.json"

          # Create the JSON configuration file
          cat <<JSON_EOF > $CONFIG_NAME
{
  "model_name": "gpt2-medium",
  "learning_rate": ${learn_rate},
  "batch_size": 64,
  "epochs": 1000,
  "save_interval": 50,
  "hidden_size": ${hidden_layer},
  "lambda_recon": ${lambda_recon},
  "lambda_energy": ${lambda_energy},
  "lambda_sparsity": ${lambda_sparsity},
  "sparsity_param": 0.05,
  "sparsity_loss": "kl"
}
JSON_EOF

          # Submit the job to the Slurm scheduler
          JOB_ID=$(sbatch <<EOT | awk '{print $4}'
#!/bin/bash
#SBATCH --job-name=SAE_training
#SBATCH --output=${LOG_DIR}/SAE_%j.out
#SBATCH --error=${LOG_DIR}/SAE_%j.err
#SBATCH --time=01:00:00
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=1
#SBATCH --mem=50G
#SBATCH --partition=gpu,xgpu

# Run the Python script with the generated config
export WANDB_API_KEY=""
python train_SAE_kl.py --config $CONFIG_NAME
EOT
)

          # Log the submitted job and its configuration
          echo "Submitted job ID ${JOB_ID} with lambda_recon=${lambda_recon}, lambda_energy=${lambda_energy}, lambda_sparsity=${lambda_sparsity}, learning_rate=${learn_rate}, hidden_layer=${hidden_layer}"

        done
      done
    done
  done
done
