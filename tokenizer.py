from tensor import Tensor
import math
import numpy as np
from typing import List, Tuple
from tokenizer_constants import vocabulary_map

class Tokenizer:
    def __init__(self, vocab_size: int, max_seq_len: int, word_embedding_dim: int, pad_token_id: int) -> None:
        # we add 1 to vocab size for pad token
        embed_std = 1 / math.sqrt(vocab_size)
        self.word_embedding_dim = word_embedding_dim
        word_embedding_weights_numpy = np.random.normal(0, embed_std, (vocab_size+1, word_embedding_dim))
        word_embedding_weights_numpy[-1] = np.zeros(word_embedding_dim)
        self.word_embedding_weights = Tensor(word_embedding_weights_numpy) # (n_words+1, embedding_dim)
        self.absolute_positional_encoding = Tensor(np.random.normal(0, embed_std, (max_seq_len, word_embedding_dim))) # (max_seq_len, word_embedding_dim)
        self.pad_token_id = pad_token_id

    def parameters(self):
        return [self.word_embedding_weights, self.absolute_positional_encoding]

    def forward(self, input_strings: List[str]) -> Tuple[Tensor, np.ndarray, np.ndarray]: # returns encoded words & token indices & padding mask
        max_len = max([len(x) for x in input_strings])
        # note: just stripping invalid tokens instead of putting <UNK> for now
        strings_with_invalid_tokens_stripped = [
            "".join([c for c in input_string if c in vocabulary_map]) for input_string in input_strings
        ]
        token_indices = np.array([[vocabulary_map[c] for c in input_string] + [self.pad_token_id] * (max_len - len(input_string)) for input_string in strings_with_invalid_tokens_stripped])

        word_embeddings = self.word_embedding_weights[token_indices] # (batch, token, word_embedding_dim)
        positionally_encoded_words = word_embeddings + self.absolute_positional_encoding[0:max_len, :]
        padding_mask = token_indices == self.pad_token_id
        return positionally_encoded_words, token_indices, padding_mask