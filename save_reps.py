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
import os
import logging
import sys

# Set up logging configuration
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

model_name = 'gpt2-medium'
layer_idx = int(sys.argv[1])  # Layer index
data_type = sys.argv[2]  # "safe" or "unsafe"

# Load the dataset
dataset = load_dataset('hate_speech_offensive')
train_dataset = dataset['train']
train_df = pd.DataFrame(train_dataset)

# Split into safe and unsafe data
train_df_safe = train_df[train_df['neither_count'] == train_df['count']]
train_df_unsafe = train_df[train_df['neither_count'] == 0]

# Determine maximum sequence length
max_length_safe = train_df_safe['tweet'].apply(len).max()
max_length_unsafe = train_df_unsafe['tweet'].apply(len).max()
max_length = max(max_length_safe, max_length_unsafe)

# Initialize tokenizer and model
tokenizer = GPT2Tokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token  # GPT-2 doesn't have a pad token

# Tokenize data based on data type
if data_type == "safe":
    tokenized_tweets = tokenizer(train_df_safe['tweet'].tolist(), padding=True, truncation=True, max_length=max_length, return_tensors='pt')
elif data_type == "unsafe":
    tokenized_tweets = tokenizer(train_df_unsafe['tweet'].tolist(), padding=True, truncation=True, max_length=max_length, return_tensors='pt')
else:
    raise ValueError("data_type must be 'safe' or 'unsafe'")

# Initialize and distribute the model across GPUs if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = GPT2Model.from_pretrained(model_name).to(device)
if torch.cuda.device_count() > 1:
    model = torch.nn.DataParallel(model)

# Function to process tweets and get embeddings for a specified layer
def get_gpt2_embeddings_for_layer(tokenized_tweets, batch_size, model, device, layer_idx):
    embeddings = []
    for i in range(0, len(tokenized_tweets['input_ids']), batch_size):
        logging.info(f"Processing batch starting at index {i} for layer {layer_idx}")
        input_ids_batch = tokenized_tweets['input_ids'][i:i + batch_size].to(device)
        attention_mask_batch = tokenized_tweets['attention_mask'][i:i + batch_size].to(device)
        with torch.no_grad():
            outputs = model(input_ids=input_ids_batch, attention_mask=attention_mask_batch, output_hidden_states=True)
            hidden_states = outputs.hidden_states
            layer_embeddings = hidden_states[layer_idx].cpu()
            embeddings.append(layer_embeddings)
        del input_ids_batch, attention_mask_batch, outputs
        torch.cuda.empty_cache()
    return torch.cat(embeddings, dim=0)

# Set batch size based on available GPUs
num_gpus_used = torch.cuda.device_count()
batch_size = 32 * max(1, num_gpus_used)
embeddings = get_gpt2_embeddings_for_layer(tokenized_tweets, batch_size, model, device, layer_idx)

# Save the embeddings
output_dir = 'embeddings_data'
os.makedirs(output_dir, exist_ok=True)
torch.save(embeddings, f"{output_dir}/{model_name}_embeddings_{data_type}_layer{layer_idx}.pt")
logging.info(f"Saved embeddings for {data_type} data, layer {layer_idx}")
