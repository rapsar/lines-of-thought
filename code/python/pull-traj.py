"""
pull-traj.py: Extract Token Embedding Trajectories from Language Models

This script processes a given text file to extract the embeddings of the last token 
from each pseudo-sentence (chunk) for all layers of a specified language model. 
The embeddings are saved for further analysis, with options for verbosity, 
saving intermediate results, and limiting the number of processed chunks.

Features:
- Process text input in manageable chunks.
- Extract embeddings from all layers of a specified language model.
- Save partial embeddings periodically, if requested.
- Control verbosity level (silent, minimal, or detailed).
- Limit the number of pseudo-sentences processed with a flag.

Usage:
    python pull-traj.py -chunksize <int> -model <model_name> [options]

Required Arguments:
    -chunksize <int>      Size of each chunk (number of tokens per chunk).
    -model <model_name>   Name of the model to use (e.g., "meta-llama/Llama-2-7b-hf").

Optional Arguments:
    -save_partial         Save intermediate embeddings every 10 chunks (default: False).
    -list_models          Display available models and exit.
    -verbose <int>        Verbosity level:
                          0 = silent
                          1 = minimal (progress bar)
                          2 = detailed (default).
    -maxtrajectories <int> Limit the number of chunks to process (default: process all).

Examples:
1. Process all pseudo-sentences with a chunk size of 50 tokens using Llama-2:
    python pull-traj.py -chunksize 50 -model meta-llama/Llama-2-7b-hf

2. Process the first 1000 pseudo-sentences with a progress bar:
    python pull-traj.py -chunksize 50 -model meta-llama/Llama-2-7b-hf -maxtrajectories 1000 -verbose 1

3. Save intermediate embeddings for every 10 chunks:
    python pull-traj.py -chunksize 50 -model meta-llama/Llama-2-7b-hf -save_partial

4. Display available models:
    python pull-traj.py -list_models

Notes:
- The input text file is expected to be located at '../../txt/walden-thoreau-pg-clean.txt'.
- Ensure that the chunk size does not exceed the model's maximum sequence length.
"""

import argparse
import numpy as np                                                                  # type: ignore
import torch                                                                        # type: ignore
from transformers import AutoModelForCausalLM, AutoTokenizer                        # type: ignore
from tqdm import tqdm  # For progress bar in minimal verbosity                      # type: ignore

def collect_last_token_embeddings(model, chunk, nDim):
    chunk = chunk.unsqueeze(0)  # Add batch dimension
    
    with torch.no_grad():
        # Forward pass through the model with hidden states enabled
        outputs = model(input_ids=chunk, output_hidden_states=True, return_dict_in_generate=True)
    
    hidden_states = outputs.hidden_states  # Hidden states from all layers
    last_token_embeddings = [hidden_states[layer_idx][0, -1, :].cpu().numpy() for layer_idx in range(len(hidden_states))]
    return np.stack(last_token_embeddings, axis=1)  # Shape: (nDim, num_layers)

def main():
    # Argument parser
    parser = argparse.ArgumentParser(description="Extract embeddings for pseudo-sentences from LLMs.")
    parser.add_argument("-chunksize", type=int, help="Size of each chunk.")
    parser.add_argument("-model", type=str, help="Name of the model to use.")
    parser.add_argument("-save_partial", action="store_true", help="Flag to save partial embeddings (default: no).")
    parser.add_argument("-list_models", action="store_true", help="List available models and exit.")
    parser.add_argument("-verbose", type=int, choices=[0, 1, 2], default=1, 
                        help="Verbosity level: 0 (silent), 1 (minimal with progress bar), 2 (full details).")
    parser.add_argument("-maxtrajectories", type=int, default=None, 
                        help="Maximum number of pseudo-sentences to process (default: process all).")
    args = parser.parse_args()

    # List of available models
    available_models = [
        "mistralai/Mistral-7B-v0.1",
        "meta-llama/Llama-2-7b-hf",
        "meta-llama/Llama-3.2-1B",
        "meta-llama/Llama-3.2-3B"
    ]

    if args.list_models:
        print("Available models:")
        for model_name in available_models:
            print(f"- {model_name}")
        return

    if args.model not in available_models:
        raise ValueError(f"Model '{args.model}' is not in the list of available models. Use -list_models to see all options.")

    # Load tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(args.model, output_hidden_states=True)
    model_short_name = args.model.split('/')[-1]
    nDim = model.config.hidden_size

    # Load and clean text
    with open('../../txt/walden-thoreau-pg-clean.txt', 'r') as file:
        long_text = file.read().replace('\n', ' ')

    # Tokenize the text
    inputs = tokenizer(long_text, return_tensors='pt', truncation=False)
    input_ids = inputs['input_ids']

    # Split into chunks of n tokens each
    chunks = input_ids[0].split(args.chunksize)
    total_sentences = len(chunks)
    if args.verbose > 0:
        print(f"Total number of pseudo-sentences (chunks): {total_sentences}")

    # Limit the number of trajectories processed if maxtrajectories is set
    if args.maxtrajectories:
        chunks = chunks[:args.maxtrajectories]
        total_sentences = len(chunks)
        if args.verbose > 0:
            print(f"Processing limited to the first {args.maxtrajectories} pseudo-sentences.")

    # Initialize the array to store embeddings
    all_embeddings = []

    # Loop through each chunk
    progress_bar = tqdm(range(total_sentences), disable=(args.verbose != 1))
    for i, chunk in enumerate(chunks):
        if args.verbose == 2:
            pseudo_sentence = tokenizer.decode(chunk)
            print(f"Processing pseudo-sentence {i + 1}: {pseudo_sentence}")

        if chunk.shape[-1] > model.config.max_position_embeddings:
            chunk = chunk[:model.config.max_position_embeddings]

        chunk_embeddings = collect_last_token_embeddings(model, chunk, nDim)
        all_embeddings.append(chunk_embeddings)

        if args.verbose == 2:
            print(f"Processed sentence {i + 1} / {total_sentences}")
        elif args.verbose == 1:
            progress_bar.update(1)

        # Save partial embeddings if the flag is set
        if args.save_partial and (i + 1) % 10 == 0:
            partial_embeddings_array = np.stack(all_embeddings, axis=2)
            np.save(f'final_embeddings_{model_short_name}_chunk{args.chunksize}_partial.npy', partial_embeddings_array)
            if args.verbose > 0:
                print(f"Saved intermediate embeddings up to sentence {i + 1}")

    if args.verbose == 1:
        progress_bar.close()

    # Save the final embeddings after all sentences are processed
    final_embeddings_array = np.stack(all_embeddings, axis=2)
    np.save(f'final_embeddings_{model_short_name}_chunk{args.chunksize}.npy', final_embeddings_array)
    if args.verbose > 0:
        print(f"All {total_sentences} pseudo-sentences processed and saved.")

if __name__ == "__main__":
    main()