from typing import List, Tuple
from tensor import Tensor
from tokenizer import Tokenizer
from tokenizer_constants import vocabulary_map
from softmax import softmax
import numpy as np
np.random.seed(42)

# ---- constants ----
epsilon = 1e-5

class AttentionHead:
    def __init__(self, d_kq: int, d_v: int, hidden_size: int):
        self.d_kq = d_kq
        self.d_v = d_v
        self.W_Q = Tensor(np.random.normal(0, 0.01, (hidden_size, d_kq)))
        self.W_K = Tensor(np.random.normal(0, 0.01, (hidden_size, d_kq)))
        self.W_V = Tensor(np.random.normal(0, 0.01, (hidden_size, d_v)))

    def parameters(self):
        return [
            self.W_Q,
            self.W_K,
            self.W_V
        ]

    def q_k_v(self, input_embeds: Tensor):
        shape = input_embeds.shape
        if len(shape) != 3:
            raise ValueError("expected input of (batch, token, embed)")
        Q = input_embeds @ self.W_Q # (batch, token, d_kq)
        K = input_embeds @ self.W_K # (batch, token, d_kq)
        V = input_embeds @ self.W_V # (batch, token, d_v)
        return Q, K, V

    def self_attn(self, input_embeds: Tensor, padding_mask: np.ndarray):
        Q, K, V = self.q_k_v(input_embeds)
        attn_scores = Q @ K.transpose((0, 2, 1))
        attn_scores = attn_scores / np.sqrt(self.d_kq)
        # let's do the causal mask
        pad = padding_mask
        pad_columns = pad[:, :, None]
        pad_rows = pad[:, None, :]

        causal_mask = np.tril(np.ones_like(attn_scores.data)) == 0
        attn_scores = attn_scores.masked_fill(causal_mask | pad_columns | pad_rows, float('-1e9'))
        # each row is the query vector
        # each column is per output vector
        scaled = softmax(attn_scores)
        # result is (batch, token, token)
        # now multiply by value
        result = scaled @ V
        return result

class AttentionLayer:
    def __init__(self, n_heads: int, hidden_size: int) -> None:
        if hidden_size % n_heads != 0:
            raise ValueError("n_heads should divide hidden_size")
        self.hidden_size = hidden_size
        self.per_head_dim = hidden_size // n_heads
        self.n_heads = n_heads
        self.heads = []
        for _ in range(n_heads):
            self.heads.append(AttentionHead(self.per_head_dim, self.per_head_dim, hidden_size))

    def parameters(self):
        params = []
        for head in self.heads:
            params.extend(head.parameters())
        return params

    def forward(self, x: Tensor, padding_mask: np.ndarray):
        shape = x.shape
        if len(shape) != 3:
            raise ValueError("expected input of (batch, token, embed)")
        result = []
        for head in self.heads:
            result.append(head.self_attn(x, padding_mask))
        # each head gives (batch, token, self.per_head_dim)
        result = Tensor.concatenate(result, axis=2)
        assert result.shape[2] == self.hidden_size
        return result

class FFLayer:
    def __init__(self, proj_dim: int, hidden_size: int) -> None:
        if proj_dim < hidden_size:
            raise ValueError("up project dim must be greater than hidden_size")
        self.up_proj_weights = Tensor(np.random.normal(0, 0.01, (hidden_size, proj_dim)))
        self.up_proj_bias = Tensor(np.zeros((proj_dim)))
        self.down_proj_weights = Tensor(np.random.normal(0, 0.01, (proj_dim, hidden_size)))
        self.down_proj_bias = Tensor(np.zeros((hidden_size)))

    def parameters(self):
        return [
            self.up_proj_weights,
            self.up_proj_bias,
            self.down_proj_weights,
            self.down_proj_bias
        ]

    def forward(self, x: Tensor):
        x = x @ self.up_proj_weights
        x = x + self.up_proj_bias
        x = x.leaky_relu()
        x = x @ self.down_proj_weights
        x = x + self.down_proj_bias
        return x

class LayerNorm:
    def __init__(self, dim: int):
        self.dim = dim
        self.gamma = Tensor(np.ones((dim,)))
        self.beta = Tensor(np.zeros((dim,)))

    def parameters(self):
        return [self.gamma, self.beta]

    def forward(self, x: Tensor):
        shape = x.shape
        if len(shape) != 3:
            raise ValueError("expected input of (batch, token, embed)")
        mean = x.mean(axis=-1, keepdims=True)
        variance = (x ** 2).mean(axis=-1, keepdims=True) - mean ** 2
        normalized = (x - mean) / (variance + epsilon).sqrt()

        transformed = normalized * self.gamma + self.beta
        return transformed

def block(hidden: Tensor, attention_layer: AttentionLayer, ff_layer: FFLayer, layer_norm_pairs: Tuple[LayerNorm, LayerNorm], padding_mask: np.ndarray) -> Tensor:
    x1 = hidden
    layer_normed = layer_norm_pairs[0].forward(x1)
    attn_out = attention_layer.forward(layer_normed, padding_mask)
    x2 = x1 + attn_out # residual connection
    layer_normed2 = layer_norm_pairs[1].forward(x2)
    ff_out = ff_layer.forward(layer_normed2)
    result = ff_out + x2 # residual connection
    return result

class Transformer():
    def __init__(self, tokenizer: Tokenizer, n_layers: int, n_attn_heads: int, ff_scale_factor: int = 4) -> None:
        self.hidden_size = tokenizer.word_embedding_dim
        self.tokenizer = tokenizer
        self.n_layers = n_layers
        self.attn_layers = [AttentionLayer(n_attn_heads, self.hidden_size) for _ in range(n_layers)]
        self.ff_layers = [FFLayer(self.hidden_size * ff_scale_factor, self.hidden_size) for _ in range(n_layers)]
        self.layer_norms = [(LayerNorm(self.hidden_size), LayerNorm(self.hidden_size)) for _ in range(n_layers)]

    def parameters(self):
        params = []
        params.extend(self.tokenizer.parameters())
        for layer in range(self.n_layers):
            params.extend(self.attn_layers[layer].parameters())
            params.extend(self.ff_layers[layer].parameters())
            params.extend(self.layer_norms[layer][0].parameters())
            params.extend(self.layer_norms[layer][1].parameters())
        return params

    def forward(self, input_strings: List[str]):
        positionally_encoded_words, _, padding_mask = self.tokenizer.forward(input_strings)
        return self.forward_tokenized(positionally_encoded_words, padding_mask)

    def forward_tokenized(self, positionally_encoded_words: Tensor, padding_mask: np.ndarray):
        last_layer_output = positionally_encoded_words
        for layer_index in range(self.n_layers):
            last_layer_output = block(last_layer_output, self.attn_layers[layer_index], self.ff_layers[layer_index], self.layer_norms[layer_index], padding_mask)

        # convert to logits with tied embedding weights
        return last_layer_output @ self.tokenizer.word_embedding_weights.transpose((1, 0))
