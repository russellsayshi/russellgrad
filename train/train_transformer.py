import time
from typing import List
from generate import generate
from russelltransformer import Tokenizer, Transformer
from tokenizer_constants import n_words, pad_token_id
from optimizers.adam import AdamOptimizer
from loss import cross_entropy_loss
import numpy as np
np.random.seed(42)

def load_dataset():
    with open("datasets/shakespeare.txt", "r") as f:
        result = f.read()
    # split dataset into chunks of 64 characters
    chunk_len = 64
    chunks = [result[i:i+chunk_len] for i in range(0, len(result), chunk_len)]

    # split chunks into `n` batches each of batch size 32
    batch_size = 64
    batches = [chunks[i:i+batch_size] for i in range(0, len(chunks), batch_size)]
    return batches

def make_target(input_tokens: List[int]) -> List[int]:
    # since we right-pad, this is safe
    # otherwise we would need to consider the input padding mask
    return np.concatenate((input_tokens[1:], np.array([pad_token_id])))

def train():
    # -- constants --
    n_layers = 8
    n_attn_heads = 8
    ff_scale_factor = 4
    max_seq_len = 64
    word_embedding_dim = 384
    
    # Training hyperparameters
    num_epochs = 20  # Define how many epochs you want
    
    tokenizer = Tokenizer(n_words, max_seq_len, word_embedding_dim, pad_token_id)
    transformer = Transformer(tokenizer, n_layers, n_attn_heads, ff_scale_factor)
    
    batches = load_dataset()
    total_steps = num_epochs * len(batches)
    warmup_steps = 5
    
    # Use cosine schedule with warmup
    optimizer = AdamOptimizer(
        transformer.parameters(), 
        lr=1e-3,
        lr_schedule='cosine_with_warmup',
        min_lr=1e-5,
        total_steps=total_steps,
        warmup_steps=warmup_steps
    )

    epoch = 0
    while epoch < num_epochs:
        for i, batch in enumerate(batches):
            if optimizer.global_step % 20 == 0:
                # every 20 steps, let's give a sample output
                print("SAMPLED STRING:", generate(tokenizer, transformer))

            optimizer.zero_grad()
            tokenized, token_indices, padding_masks = tokenizer.forward(batch)
            targets = np.array([make_target(t) for t in token_indices])

            output_transformer = transformer.forward_tokenized(tokenized, padding_masks)
            loss = cross_entropy_loss(output_transformer, targets, pad_token_id)
            print('epoch', epoch, '| batch', i, '/', len(batches), f'| lr {optimizer.lr:.5f}', '| loss', loss.data.item())
            loss.backward()
            optimizer.step()
            loss.zero_grad_graph() # prevents memory leak by removing circular dependencies
        epoch += 1

if __name__ == "__main__":
    train()