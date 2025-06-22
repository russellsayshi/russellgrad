from typing import List
from russelltransformer import Tokenizer, Transformer
from tokenizer_constants import n_words, pad_token_id
from optimizer import SGDOptimizer
from loss import cross_entropy_loss
import numpy as np
np.random.seed(42)

def load_dataset():
    with open("datasets/shakespeare.txt", "r") as f:
        result = f.read()
    # split dataset into chunks of 512 characters
    chunks = [result[i:i+512] for i in range(0, len(result), 512)]

    # split chunks into `n` batches each of batch size 16
    batches = [chunks[i:i+8] for i in range(0, len(chunks), 8)]
    return batches

def make_target(input_tokens: List[int]) -> List[int]:
    # since we right-pad, this is safe
    # otherwise we would need to consider the input padding mask
    return np.concatenate((input_tokens[1:], np.array([pad_token_id])))

def train():
    # -- constants --
    n_layers = 4
    n_attn_heads = 4
    ff_scale_factor = 4
    max_seq_len = 512
    word_embedding_dim = 64

    tokenizer = Tokenizer(n_words, max_seq_len, word_embedding_dim, pad_token_id)
    transformer = Transformer(tokenizer, n_layers, n_attn_heads, ff_scale_factor)
    
    optimizer = SGDOptimizer(transformer.parameters(), lr=1e-2)

    batches = load_dataset()

    epoch = 0
    while True:
        for i, batch in enumerate(batches):
            optimizer.zero_grad()
            tokenized, token_indices, padding_masks = tokenizer.forward(batch)
            targets = np.array([make_target(t) for t in token_indices])

            output_transformer = transformer.forward_tokenized(tokenized, padding_masks)
            loss = cross_entropy_loss(output_transformer, targets, pad_token_id)
            print('epoch', epoch, '| batch', i, '/', len(batches), ' | loss', loss.data.item())
            loss.backward()
            optimizer.step()
        epoch += 1

if __name__ == "__main__":
    train()