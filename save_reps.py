#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Sep 28 11:10:42 2024

@author: apratimdey
"""

from datasets import load_dataset
import pandas as pd
import torch
from transformers import GPT2Tokenizer, GPT2Model
import logging
# Set up logging configuration
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

model_name = 'gpt2-medium'

# Load the dataset
dataset = load_dataset('hate_speech_offensive')

# Access training data only
train_dataset = dataset['train']

# Convert to DF
train_df = pd.DataFrame(train_dataset)

# everybody agrees safe
train_df_safe = train_df[train_df['neither_count']==train_df['count']]

# everybody agrees unsafe
train_df_unsafe = train_df[train_df['neither_count']==0]

max_length_safe = train_df_safe['tweet'].apply(len).max()
max_length_unsafe = train_df_unsafe['tweet'].apply(len).max()
max_length = max(max_length_safe, max_length_unsafe)

# Initialize the GPT-2 tokenizer
tokenizer = GPT2Tokenizer.from_pretrained(model_name)

# GPT-2 doesn't have a pad token, so we use the EOS (end-of-sentence) token as a pad token
tokenizer.pad_token = tokenizer.eos_token

# 'padding=True' ensures shorter tweets are padded to the max length,
# and 'truncation=True' ensures longer tweets are truncated.
tokenized_tweets_safe = tokenizer(train_df_safe['tweet'].tolist(),
                             padding=True,
                             truncation=True,
                             max_length=max_length,
                             return_tensors='pt')

tokenized_tweets_unsafe = tokenizer(train_df_safe['tweet'].tolist(),
                             padding=True,
                             truncation=True,
                             max_length=max_length,
                             return_tensors='pt')

# Initialize the GPT-2 model and move it to GPU (if available)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logging.info(f"Device is {device}")

model = GPT2Model.from_pretrained(model_name).to(device)

# Tokenization and batch size
batch_size = 32  # Adjust batch size based on available GPU memory

# Function to process tweets in batches
def get_gpt2_embeddings_in_batches(tokenized_tweets, batch_size, model, device):
    embeddings_list = []
    for i in range(0, len(tokenized_tweets['input_ids']), batch_size):
        logging.info(f"Iteration {i}")
        input_ids_batch = tokenized_tweets['input_ids'][i:i+batch_size].to(device)
        attention_mask_batch = tokenized_tweets['attention_mask'][i:i+batch_size].to(device)

        with torch.no_grad():
            outputs = model(input_ids=input_ids_batch, attention_mask=attention_mask_batch)
            embeddings_batch = outputs.last_hidden_state.cpu() # Move to CPU
            embeddings_list.append(embeddings_batch)

    return torch.cat(embeddings_list, dim=0)  # Concatenate all batches into a single tensor

# Process the safe and unsafe datasets in smaller batches
logging.info("Processing safe data")
embeddings_safe = get_gpt2_embeddings_in_batches(tokenized_tweets_safe, batch_size, model, device)

logging.info("Processing unsafe data")
embeddings_unsafe = get_gpt2_embeddings_in_batches(tokenized_tweets_unsafe, batch_size, model, device)

# Save the embeddings as Torch tensors
torch.save(embeddings_safe, f'embeddings_data/{model_name}_embeddings_safe.pt')
torch.save(embeddings_unsafe, f'embeddings_data/{model_name}_embeddings_unsafe.pt')
