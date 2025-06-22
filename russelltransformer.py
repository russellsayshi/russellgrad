from tensor import Tensor
import math
import numpy as np
np.random.seed(42)

def softmax(x, axis=-1): # TODO: remember this
    e_x = (x - x.max(axis=axis, keepdims=True)).exp()
    return e_x / e_x.sum(axis=axis, keepdims=True)

# NOTE: Have chatgpt give you versions of this with things subtly wrong and you have to guess what's wrong

vocabulary = list("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ 0123456789.,!?")
vocabulary_map = {}
for i, c in enumerate(vocabulary):
    vocabulary_map[c] = i
n_words = len(vocabulary)

# ---- constants ----
word_embedding_dim = 16
max_seq_len = 256
pad_token_id = n_words
epsilon = 1e-6
leaky_relu_alpha = 0.2
embed_std = 1 / math.sqrt(word_embedding_dim)

class AttentionHead:
    def __init__(self, d_kq, d_v):
        self.d_kq = d_kq
        self.d_v = d_v
        self.W_Q = Tensor(np.random.normal(0, 0.02, (word_embedding_dim, d_kq)))
        self.W_K = Tensor(np.random.normal(0, 0.02, (word_embedding_dim, d_kq)))
        self.W_V = Tensor(np.random.normal(0, 0.02, (word_embedding_dim, d_v)))

    def q_k_v(self, input_embeds):
        shape = input_embeds.shape
        if len(shape) != 3:
            raise ValueError("expected input of (batch, token, embed)")
        Q = input_embeds @ self.W_Q # (batch, token, d_kq)
        K = input_embeds @ self.W_K # (batch, token, d_kq)
        V = input_embeds @ self.W_V # (batch, token, d_v)
        return Q, K, V

    def self_attn(self, input_embeds, padding_mask):
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
    def __init__(self, n_heads) -> None:
        if word_embedding_dim % n_heads != 0:
            raise ValueError("n_heads should divide word_embedding_dim")
        self.per_head_dim = word_embedding_dim // n_heads
        self.n_heads = n_heads
        self.heads = []
        for _ in range(n_heads):
            self.heads.append(AttentionHead(self.per_head_dim, self.per_head_dim))

    def forward(self, x, padding_mask):
        shape = x.shape
        if len(shape) != 3:
            raise ValueError("expected input of (batch, token, embed)")
        result = []
        for head in self.heads:
            result.append(head.self_attn(x, padding_mask))
        # each head gives (batch, token, self.per_head_dim)
        result = Tensor.concatenate(result, axis=2)
        assert result.shape[2] == word_embedding_dim
        return result

if __name__ == "__main__":
    import torch
    import torch.nn.functional as F
    
    # Test attention head gradients
    print("Testing AttentionHead gradients...")
    
    # Create test data
    batch_size = 2
    seq_len = 4
    d_kq = 8
    d_v = 6
    
    # RussellGrad version
    attn_head = AttentionHead(d_kq, d_v)
    input_embeds = Tensor(np.random.randn(batch_size, seq_len, word_embedding_dim))
    padding_mask = np.array([
        [False, False, False, True],   # first sequence has 3 tokens
        [False, False, True, True]     # second sequence has 2 tokens
    ])
    
    output = attn_head.self_attn(input_embeds, padding_mask)
    loss = output.sum()
    loss.backward()
    
    # PyTorch version
    torch.manual_seed(42)  # For reproducibility
    
    # Create PyTorch attention head
    class PyTorchAttentionHead(torch.nn.Module):
        def __init__(self, embed_dim, d_kq, d_v):
            super().__init__()
            self.d_kq = d_kq
            self.d_v = d_v
            self.W_Q = torch.nn.Linear(embed_dim, d_kq, bias=False, dtype=torch.float64)
            self.W_K = torch.nn.Linear(embed_dim, d_kq, bias=False, dtype=torch.float64)
            self.W_V = torch.nn.Linear(embed_dim, d_v, bias=False, dtype=torch.float64)
            
        def forward(self, input_embeds, padding_mask):
            Q = self.W_Q(input_embeds)  # (batch, seq_len, d_kq)
            K = self.W_K(input_embeds)  # (batch, seq_len, d_kq)
            V = self.W_V(input_embeds)  # (batch, seq_len, d_v)
            
            # Compute attention scores
            attn_scores = torch.matmul(Q, K.transpose(-2, -1))  # (batch, seq_len, seq_len)
            attn_scores = attn_scores / torch.sqrt(torch.tensor(self.d_kq, dtype=torch.float64))
            
            # Apply causal mask
            seq_len = input_embeds.size(1)
            causal_mask = torch.tril(torch.ones(seq_len, seq_len, dtype=torch.float64)) == 0
            causal_mask = causal_mask.unsqueeze(0).expand(input_embeds.size(0), -1, -1)
            
            # Apply padding mask
            pad_columns = padding_mask.unsqueeze(-1).expand(-1, -1, seq_len)
            pad_rows = padding_mask.unsqueeze(1).expand(-1, seq_len, -1)
            
            # Combine masks
            combined_mask = causal_mask | pad_columns | pad_rows
            attn_scores = attn_scores.masked_fill(combined_mask, float('-1e9'))
            
            # Apply softmax
            attn_weights = F.softmax(attn_scores, dim=-1)
            
            # Apply attention to values
            output = torch.matmul(attn_weights, V)
            return output
    
    # Initialize PyTorch model with same weights
    torch_attn = PyTorchAttentionHead(word_embedding_dim, d_kq, d_v)
    with torch.no_grad():
        torch_attn.W_Q.weight.copy_(torch.tensor(attn_head.W_Q.data.T, dtype=torch.float64))
        torch_attn.W_K.weight.copy_(torch.tensor(attn_head.W_K.data.T, dtype=torch.float64))
        torch_attn.W_V.weight.copy_(torch.tensor(attn_head.W_V.data.T, dtype=torch.float64))
    
    # Convert inputs to PyTorch
    input_embeds_torch = torch.tensor(input_embeds.data, dtype=torch.float64, requires_grad=True)
    padding_mask_torch = torch.tensor(padding_mask, dtype=torch.bool)
    
    # Forward pass
    output_torch = torch_attn(input_embeds_torch, padding_mask_torch)
    loss_torch = output_torch.sum()
    loss_torch.backward()
    
    # Compare gradients
    print("Gradient comparisons:")
    print(f"W_Q grad diff: {np.abs(attn_head.W_Q.grad - torch_attn.W_Q.weight.grad.numpy().T).max():.2e}")
    print(f"W_K grad diff: {np.abs(attn_head.W_K.grad - torch_attn.W_K.weight.grad.numpy().T).max():.2e}")
    print(f"W_V grad diff: {np.abs(attn_head.W_V.grad - torch_attn.W_V.weight.grad.numpy().T).max():.2e}")
    print(f"Input grad diff: {np.abs(input_embeds.grad - input_embeds_torch.grad.numpy()).max():.2e}")
    
    # Compare outputs
    print(f"Output diff: {np.abs(output.data - output_torch.detach().numpy()).max():.2e}")
    print(f"Loss diff: {abs(loss.data - loss_torch.item()):.2e}")
    
    print("AttentionHead gradient test completed!")
    
    # Test attention layer gradients
    print("\nTesting AttentionLayer gradients...")
    
    # Create test data for attention layer
    n_heads = 4
    
    # RussellGrad version
    attn_layer = AttentionLayer(n_heads)
    input_embeds_layer = Tensor(np.random.randn(batch_size, seq_len, word_embedding_dim))
    
    output_layer = attn_layer.forward(input_embeds_layer, padding_mask)
    loss_layer = output_layer.sum()
    loss_layer.backward()
    
    # PyTorch version
    class PyTorchAttentionLayer(torch.nn.Module):
        def __init__(self, embed_dim, n_heads):
            super().__init__()
            if embed_dim % n_heads != 0:
                raise ValueError("n_heads should divide embed_dim")
            self.per_head_dim = embed_dim // n_heads
            self.n_heads = n_heads
            self.heads = torch.nn.ModuleList([
                PyTorchAttentionHead(embed_dim, self.per_head_dim, self.per_head_dim)
                for _ in range(n_heads)
            ])
            
        def forward(self, x, padding_mask):
            results = []
            for head in self.heads:
                results.append(head(x, padding_mask))
            # Concatenate along the last dimension
            return torch.cat(results, dim=-1)
    
    # Initialize PyTorch model with same weights
    torch_attn_layer = PyTorchAttentionLayer(word_embedding_dim, n_heads)
    with torch.no_grad():
        for i, head in enumerate(torch_attn_layer.heads):
            head.W_Q.weight.copy_(torch.tensor(attn_layer.heads[i].W_Q.data.T, dtype=torch.float64))
            head.W_K.weight.copy_(torch.tensor(attn_layer.heads[i].W_K.data.T, dtype=torch.float64))
            head.W_V.weight.copy_(torch.tensor(attn_layer.heads[i].W_V.data.T, dtype=torch.float64))
    
    # Convert inputs to PyTorch
    input_embeds_layer_torch = torch.tensor(input_embeds_layer.data, dtype=torch.float64, requires_grad=True)
    
    # Forward pass
    output_layer_torch = torch_attn_layer(input_embeds_layer_torch, padding_mask_torch)
    loss_layer_torch = output_layer_torch.sum()
    loss_layer_torch.backward()
    
    # Compare gradients for each head
    print("AttentionLayer gradient comparisons:")
    for i in range(n_heads):
        print(f"Head {i}:")
        print(f"  W_Q grad diff: {np.abs(attn_layer.heads[i].W_Q.grad - torch_attn_layer.heads[i].W_Q.weight.grad.numpy().T).max():.2e}")
        print(f"  W_K grad diff: {np.abs(attn_layer.heads[i].W_K.grad - torch_attn_layer.heads[i].W_K.weight.grad.numpy().T).max():.2e}")
        print(f"  W_V grad diff: {np.abs(attn_layer.heads[i].W_V.grad - torch_attn_layer.heads[i].W_V.weight.grad.numpy().T).max():.2e}")
    
    print(f"Input grad diff: {np.abs(input_embeds_layer.grad - input_embeds_layer_torch.grad.numpy()).max():.2e}")
    
    # Compare outputs
    print(f"Output diff: {np.abs(output_layer.data - output_layer_torch.detach().numpy()).max():.2e}")
    print(f"Loss diff: {abs(loss_layer.data - loss_layer_torch.item()):.2e}")
    
    print("AttentionLayer gradient test completed!")
