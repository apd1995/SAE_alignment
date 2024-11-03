First, create the environment using env.yml.

Create a folder SAE_toxicity.

To get embeddings, go to embeddings_code. The main script is save_reps.py. Don't do anything to it. Instead, go to save_reps.sh, and change the partition according to your cluster.
Then, in terminal, run:

``chmod +x submit_jobs.sh``

``./submit_jobs.sh``


You will see a bunch of .pt files, each a 3d tensor, with the middle index denoting the number of tokens.
