from typing import List, Tuple
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
epsilon = 1e-5
leaky_relu_alpha = 0.2
embed_std = 1 / math.sqrt(word_embedding_dim)

class AttentionHead:
    def __init__(self, d_kq: int, d_v: int, hidden_size: int):
        self.d_kq = d_kq
        self.d_v = d_v
        self.W_Q = Tensor(np.random.normal(0, 0.02, (hidden_size, d_kq)))
        self.W_K = Tensor(np.random.normal(0, 0.02, (hidden_size, d_kq)))
        self.W_V = Tensor(np.random.normal(0, 0.02, (hidden_size, d_v)))

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
        self.per_head_dim = word_embedding_dim // n_heads
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
        self.up_proj_weights = Tensor(np.random.normal(0, 0.02, (hidden_size, proj_dim)))
        self.up_proj_bias = Tensor(np.zeros((proj_dim)))
        self.down_proj_weights = Tensor(np.random.normal(0, 0.02, (proj_dim, hidden_size)))
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

class Tokenizer:
    def __init__(self, vocab_size: int, max_seq_len: int, word_embedding_dim: int) -> None:
        # we add 1 to vocab size for pad token
        embed_std = 1 / math.sqrt(vocab_size)
        self.word_embedding_dim = word_embedding_dim
        word_embedding_weights_numpy = np.random.normal(0, embed_std, (vocab_size+1, word_embedding_dim))
        word_embedding_weights_numpy[-1] = np.zeros(word_embedding_dim)
        self.word_embedding_weights = Tensor(word_embedding_weights_numpy) # (n_words+1, embedding_dim)
        self.absolute_positional_encoding = Tensor(np.random.normal(0, embed_std, (max_seq_len, word_embedding_dim))) # (max_seq_len, word_embedding_dim)

    def parameters(self):
        return [self.word_embedding_weights, self.absolute_positional_encoding]

    def forward(self, input_strings: List[str]) -> Tuple[Tensor, np.ndarray]: # returns encoded words & padding mask
        max_len = max([len(x) for x in input_strings])
        token_indices = np.array([[vocabulary_map[c] for c in input_string if c in vocabulary_map] + [pad_token_id] * (max_len - len(input_string)) for input_string in input_strings])

        print(token_indices)
        word_embeddings = self.word_embedding_weights[token_indices] # (batch, token, word_embedding_dim)
        positionally_encoded_words = word_embeddings + self.absolute_positional_encoding[0:max_len, :]
        padding_mask = token_indices == pad_token_id
        return positionally_encoded_words, padding_mask

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
        for layer in range(self.n_layers):
            params.extend(self.attn_layers[layer].parameters())
            params.extend(self.ff_layers[layer].parameters())
            params.extend(self.layer_norms[layer][0].parameters())
            params.extend(self.layer_norms[layer][1].parameters())
        return params

    def forward(self, input_strings: List[str]):
        positionally_encoded_words, padding_mask = self.tokenizer.forward(input_strings)
        last_layer_output = positionally_encoded_words
        for layer_index in range(self.n_layers):
            last_layer_output = block(last_layer_output, self.attn_layers[layer_index], self.ff_layers[layer_index], self.layer_norms[layer_index], padding_mask)

        # convert to logits with tied embedding weights
        return last_layer_output @ self.tokenizer.word_embedding_weights.transpose((1, 0))

if __name__ == "__main__":
    import torch
    import torch.nn.functional as F
    
    # ANSI color codes for beautiful output
    class Colors:
        HEADER = '\033[95m'
        OKBLUE = '\033[94m'
        OKCYAN = '\033[96m'
        OKGREEN = '\033[92m'
        WARNING = '\033[93m'
        FAIL = '\033[91m'
        ENDC = '\033[0m'
        BOLD = '\033[1m'
        UNDERLINE = '\033[4m'
    
    def print_test_result(test_name, passed, tolerance=1e-6):
        status = f"{Colors.OKGREEN}✓ PASSED{Colors.ENDC}" if passed else f"{Colors.FAIL}✗ FAILED{Colors.ENDC}"
        print(f"{Colors.BOLD}{test_name}:{Colors.ENDC} {status} (tol: {tolerance:.0e})")
    
    def print_section_header(title):
        print(f"\n{Colors.HEADER}{Colors.BOLD}{'='*60}{Colors.ENDC}")
        print(f"{Colors.HEADER}{Colors.BOLD}{title.center(60)}{Colors.ENDC}")
        print(f"{Colors.HEADER}{Colors.BOLD}{'='*60}{Colors.ENDC}")
    
    def print_subsection(title):
        print(f"\n{Colors.OKCYAN}{Colors.BOLD}{title}{Colors.ENDC}")
        print(f"{Colors.OKCYAN}{'-' * len(title)}{Colors.ENDC}")
    
    # Test attention head gradients
    print_section_header("ATTENTION HEAD GRADIENT TESTING")
    
    # Create test data
    batch_size = 2
    seq_len = 4
    d_kq = 8
    d_v = 6
    tolerance = 1e-12
    
    print(f"{Colors.OKBLUE}Test Configuration:{Colors.ENDC}")
    print(f"  Batch size: {batch_size}")
    print(f"  Sequence length: {seq_len}")
    print(f"  Key/Query dimension: {d_kq}")
    print(f"  Value dimension: {d_v}")
    print(f"  Tolerance: {tolerance:.0e}")
    
    # RussellGrad version
    attn_head = AttentionHead(d_kq, d_v, word_embedding_dim)
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
    
    # Compare gradients and outputs
    print_subsection("Gradient Comparisons")
    
    wq_grad_close = np.allclose(attn_head.W_Q.grad, torch_attn.W_Q.weight.grad.numpy().T, atol=tolerance)
    print_test_result("W_Q gradient", wq_grad_close, tolerance)
    
    wk_grad_close = np.allclose(attn_head.W_K.grad, torch_attn.W_K.weight.grad.numpy().T, atol=tolerance)
    print_test_result("W_K gradient", wk_grad_close, tolerance)
    
    wv_grad_close = np.allclose(attn_head.W_V.grad, torch_attn.W_V.weight.grad.numpy().T, atol=tolerance)
    print_test_result("W_V gradient", wv_grad_close, tolerance)
    
    input_grad_close = np.allclose(input_embeds.grad, input_embeds_torch.grad.numpy(), atol=tolerance)
    print_test_result("Input gradient", input_grad_close, tolerance)
    
    print_subsection("Output Comparisons")
    
    output_close = np.allclose(output.data, output_torch.detach().numpy(), atol=tolerance)
    print_test_result("Forward output", output_close, tolerance)
    
    loss_close = np.allclose(loss.data, loss_torch.item(), atol=tolerance)
    print_test_result("Loss value", loss_close, tolerance)
    
    # Test attention layer gradients
    print_section_header("ATTENTION LAYER GRADIENT TESTING")
    
    # Create test data for attention layer
    n_heads = 4
    
    print(f"{Colors.OKBLUE}Test Configuration:{Colors.ENDC}")
    print(f"  Number of heads: {n_heads}")
    print(f"  Per-head dimension: {word_embedding_dim // n_heads}")
    print(f"  Tolerance: {tolerance:.0e}")
    
    # RussellGrad version
    attn_layer = AttentionLayer(n_heads, word_embedding_dim)
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
    print_subsection("Per-Head Gradient Comparisons")
    all_heads_passed = True
    
    for i in range(n_heads):
        print(f"\n{Colors.OKBLUE}Head {i+1}:{Colors.ENDC}")
        
        wq_close = np.allclose(attn_layer.heads[i].W_Q.grad, torch_attn_layer.heads[i].W_Q.weight.grad.numpy().T, atol=tolerance)
        print_test_result(f"  W_Q gradient", wq_close, tolerance)
        
        wk_close = np.allclose(attn_layer.heads[i].W_K.grad, torch_attn_layer.heads[i].W_K.weight.grad.numpy().T, atol=tolerance)
        print_test_result(f"  W_K gradient", wk_close, tolerance)
        
        wv_close = np.allclose(attn_layer.heads[i].W_V.grad, torch_attn_layer.heads[i].W_V.weight.grad.numpy().T, atol=tolerance)
        print_test_result(f"  W_V gradient", wv_close, tolerance)
        
        all_heads_passed = all_heads_passed and wq_close and wk_close and wv_close
    
    print_subsection("Overall Layer Comparisons")
    
    input_layer_grad_close = np.allclose(input_embeds_layer.grad, input_embeds_layer_torch.grad.numpy(), atol=tolerance)
    print_test_result("Input gradient", input_layer_grad_close, tolerance)
    
    output_layer_close = np.allclose(output_layer.data, output_layer_torch.detach().numpy(), atol=tolerance)
    print_test_result("Forward output", output_layer_close, tolerance)
    
    loss_layer_close = np.allclose(loss_layer.data, loss_layer_torch.item(), atol=tolerance)
    print_test_result("Loss value", loss_layer_close, tolerance)
    
    overall_layer_passed = all_heads_passed and input_layer_grad_close and output_layer_close and loss_layer_close
    print(f"\n{Colors.BOLD}Overall AttentionLayer Test: {Colors.OKGREEN if overall_layer_passed else Colors.FAIL}{'PASSED' if overall_layer_passed else 'FAILED'}{Colors.ENDC}")
    
    # Test FF layer gradients
    print_section_header("FEEDFORWARD LAYER GRADIENT TESTING")
    
    # Create test data for FF layer
    proj_dim = 128
    
    print(f"{Colors.OKBLUE}Test Configuration:{Colors.ENDC}")
    print(f"  Input dimension: {word_embedding_dim}")
    print(f"  Projection dimension: {proj_dim}")
    print(f"  Tolerance: {tolerance:.0e}")
    
    # RussellGrad version
    ff_layer = FFLayer(proj_dim, word_embedding_dim)
    input_embeds_ff = Tensor(np.random.randn(batch_size, seq_len, word_embedding_dim))
    
    output_ff = ff_layer.forward(input_embeds_ff)
    loss_ff = output_ff.sum()
    loss_ff.backward()
    
    # PyTorch version
    class PyTorchFFLayer(torch.nn.Module):
        def __init__(self, embed_dim, proj_dim):
            super().__init__()
            self.up_proj = torch.nn.Linear(embed_dim, proj_dim, dtype=torch.float64)
            self.down_proj = torch.nn.Linear(proj_dim, embed_dim, dtype=torch.float64)
            
        def forward(self, x):
            x = self.up_proj(x)
            x = F.leaky_relu(x, negative_slope=0.2)
            x = self.down_proj(x)
            return x
    
    # Initialize PyTorch model with same weights
    torch_ff_layer = PyTorchFFLayer(word_embedding_dim, proj_dim)
    with torch.no_grad():
        torch_ff_layer.up_proj.weight.copy_(torch.tensor(ff_layer.up_proj_weights.data.T, dtype=torch.float64))
        torch_ff_layer.up_proj.bias.copy_(torch.tensor(ff_layer.up_proj_bias.data, dtype=torch.float64))
        torch_ff_layer.down_proj.weight.copy_(torch.tensor(ff_layer.down_proj_weights.data.T, dtype=torch.float64))
        torch_ff_layer.down_proj.bias.copy_(torch.tensor(ff_layer.down_proj_bias.data, dtype=torch.float64))
    
    # Convert inputs to PyTorch
    input_embeds_ff_torch = torch.tensor(input_embeds_ff.data, dtype=torch.float64, requires_grad=True)
    
    # Forward pass
    output_ff_torch = torch_ff_layer(input_embeds_ff_torch)
    loss_ff_torch = output_ff_torch.sum()
    loss_ff_torch.backward()
    
    # Compare gradients
    print_subsection("Weight and Bias Gradient Comparisons")
    
    up_weights_close = np.allclose(ff_layer.up_proj_weights.grad, torch_ff_layer.up_proj.weight.grad.numpy().T, atol=tolerance)
    print_test_result("Up projection weights", up_weights_close, tolerance)
    
    up_bias_close = np.allclose(ff_layer.up_proj_bias.grad, torch_ff_layer.up_proj.bias.grad.numpy(), atol=tolerance)
    print_test_result("Up projection bias", up_bias_close, tolerance)
    
    down_weights_close = np.allclose(ff_layer.down_proj_weights.grad, torch_ff_layer.down_proj.weight.grad.numpy().T, atol=tolerance)
    print_test_result("Down projection weights", down_weights_close, tolerance)
    
    down_bias_close = np.allclose(ff_layer.down_proj_bias.grad, torch_ff_layer.down_proj.bias.grad.numpy(), atol=tolerance)
    print_test_result("Down projection bias", down_bias_close, tolerance)
    
    input_ff_grad_close = np.allclose(input_embeds_ff.grad, input_embeds_ff_torch.grad.numpy(), atol=tolerance)
    print_test_result("Input gradient", input_ff_grad_close, tolerance)
    
    print_subsection("Output Comparisons")
    
    output_ff_close = np.allclose(output_ff.data, output_ff_torch.detach().numpy(), atol=tolerance)
    print_test_result("Forward output", output_ff_close, tolerance)
    
    loss_ff_close = np.allclose(loss_ff.data, loss_ff_torch.item(), atol=tolerance)
    print_test_result("Loss value", loss_ff_close, tolerance)
    
    overall_ff_passed = (up_weights_close and up_bias_close and down_weights_close and 
                        down_bias_close and input_ff_grad_close and output_ff_close and loss_ff_close)
    print(f"\n{Colors.BOLD}Overall FFLayer Test: {Colors.OKGREEN if overall_ff_passed else Colors.FAIL}{'PASSED' if overall_ff_passed else 'FAILED'}{Colors.ENDC}")
    
    # Test LayerNorm gradients
    print_section_header("LAYER NORM GRADIENT TESTING")
    
    print(f"{Colors.OKBLUE}Test Configuration:{Colors.ENDC}")
    print(f"  Input dimension: {word_embedding_dim}")
    print(f"  Tolerance: {tolerance:.0e}")
    
    # RussellGrad version
    layer_norm = LayerNorm(word_embedding_dim)
    input_embeds_ln = Tensor(np.random.randn(batch_size, seq_len, word_embedding_dim))
    
    output_ln = layer_norm.forward(input_embeds_ln)
    loss_ln = output_ln.sum()
    loss_ln.backward()
    
    # PyTorch version
    torch_layer_norm = torch.nn.LayerNorm(word_embedding_dim, dtype=torch.float64)
    with torch.no_grad():
        torch_layer_norm.weight.copy_(torch.tensor(layer_norm.gamma.data, dtype=torch.float64))
        torch_layer_norm.bias.copy_(torch.tensor(layer_norm.beta.data, dtype=torch.float64))
    
    # Convert inputs to PyTorch
    input_embeds_ln_torch = torch.tensor(input_embeds_ln.data, dtype=torch.float64, requires_grad=True)
    
    # Forward pass
    output_ln_torch = torch_layer_norm(input_embeds_ln_torch)
    loss_ln_torch = output_ln_torch.sum()
    loss_ln_torch.backward()
    
    # Compare gradients
    print_subsection("Parameter Gradient Comparisons")
    
    gamma_grad_close = np.allclose(layer_norm.gamma.grad, torch_layer_norm.weight.grad.numpy(), atol=tolerance)
    print_test_result("Gamma (weight) gradient", gamma_grad_close, tolerance)
    
    beta_grad_close = np.allclose(layer_norm.beta.grad, torch_layer_norm.bias.grad.numpy(), atol=tolerance)
    print_test_result("Beta (bias) gradient", beta_grad_close, tolerance)
    
    input_ln_grad_close = np.allclose(input_embeds_ln.grad, input_embeds_ln_torch.grad.numpy(), atol=tolerance)
    print_test_result("Input gradient", input_ln_grad_close, tolerance)
    
    print_subsection("Output Comparisons")
    
    output_ln_close = np.allclose(output_ln.data, output_ln_torch.detach().numpy(), atol=tolerance)
    print_test_result("Forward output", output_ln_close, tolerance)
    
    loss_ln_close = np.allclose(loss_ln.data, loss_ln_torch.item(), atol=tolerance)
    print_test_result("Loss value", loss_ln_close, tolerance)
    
    overall_ln_passed = (gamma_grad_close and beta_grad_close and input_ln_grad_close and 
                        output_ln_close and loss_ln_close)
    print(f"\n{Colors.BOLD}Overall LayerNorm Test: {Colors.OKGREEN if overall_ln_passed else Colors.FAIL}{'PASSED' if overall_ln_passed else 'FAILED'}{Colors.ENDC}")
    
    # Test Tokenizer
    print_section_header("TOKENIZER TESTING")
    
    vocab_size = n_words
    max_seq_len = 8
    
    print(f"{Colors.OKBLUE}Test Configuration:{Colors.ENDC}")
    print(f"  Vocabulary size: {vocab_size}")
    print(f"  Max sequence length: {max_seq_len}")
    print(f"  Word embedding dimension: {word_embedding_dim}")
    print(f"  Tolerance: {tolerance:.0e}")
    
    # RussellGrad version
    tokenizer = Tokenizer(vocab_size, max_seq_len, word_embedding_dim)
    test_strings = ["hello", "world", "ai"]
    
    output_tok, padding_mask_tok = tokenizer.forward(test_strings)
    loss_tok = output_tok.sum()
    loss_tok.backward()
    
    print(f"\n{Colors.WARNING}=== GRADIENT DEBUG ==={Colors.ENDC}")
    print(f"\n{Colors.OKCYAN}RussellGrad word embedding gradients shape:{Colors.ENDC}", tokenizer.word_embedding_weights.grad.shape)
    print(f"\n{Colors.OKCYAN}RussellGrad gradients for padding token (last row):{Colors.ENDC}")
    print(tokenizer.word_embedding_weights.grad[-1])
    print(f"\n{Colors.OKCYAN}RussellGrad gradients for first few tokens:{Colors.ENDC}")
    print(tokenizer.word_embedding_weights.grad[:5])
    
    # PyTorch version
    class PyTorchTokenizer(torch.nn.Module):
        def __init__(self, vocab_size, max_seq_len, embed_dim):
            super().__init__()
            self.word_embedding = torch.nn.Embedding(vocab_size + 1, embed_dim, dtype=torch.float64)
            self.pos_embedding = torch.nn.Parameter(torch.randn(max_seq_len, embed_dim, dtype=torch.float64))
            
        def forward(self, input_strings):
            max_len = max([len(x) for x in input_strings])
            token_indices = np.array([[vocabulary_map[c] for c in input_string if c in vocabulary_map] + [pad_token_id] * (max_len - len(input_string)) for input_string in input_strings])
            
            token_indices_torch = torch.tensor(token_indices, dtype=torch.long)
            word_embeddings = self.word_embedding(token_indices_torch)
            positionally_encoded_words = word_embeddings + self.pos_embedding[0:max_len, :]
            padding_mask = torch.tensor(token_indices == pad_token_id, dtype=torch.bool)  # Convert to PyTorch tensor
            return positionally_encoded_words, padding_mask
    
    # Initialize PyTorch model with same weights
    torch_tokenizer = PyTorchTokenizer(vocab_size, max_seq_len, word_embedding_dim)
    with torch.no_grad():
        torch_tokenizer.word_embedding.weight.copy_(torch.tensor(tokenizer.word_embedding_weights.data, dtype=torch.float64))
        torch_tokenizer.pos_embedding.copy_(torch.tensor(tokenizer.absolute_positional_encoding.data, dtype=torch.float64))
    
    # Forward pass
    output_tok_torch, padding_mask_tok_torch = torch_tokenizer(test_strings)
    loss_tok_torch = output_tok_torch.sum()
    loss_tok_torch.backward()
    
    print(f"\n{Colors.OKCYAN}PyTorch word embedding gradients shape:{Colors.ENDC}", torch_tokenizer.word_embedding.weight.grad.shape)
    print(f"\n{Colors.OKCYAN}PyTorch gradients for padding token (last row):{Colors.ENDC}")
    print(torch_tokenizer.word_embedding.weight.grad[-1].numpy())
    print(f"\n{Colors.OKCYAN}PyTorch gradients for first few tokens:{Colors.ENDC}")
    print(torch_tokenizer.word_embedding.weight.grad[:5].numpy())

    # Check which tokens were actually used
    print(f"\n{Colors.OKCYAN}Token indices used:{Colors.ENDC}")
    print("Token indices array:")
    max_len = max([len(s) for s in test_strings])
    token_indices = np.array([[vocabulary_map[c] for c in input_string if c in vocabulary_map] + [pad_token_id] * (max_len - len(input_string)) for input_string in test_strings])
    print(token_indices)
    print(f"Unique token indices: {np.unique(token_indices)}")

    # Compare specific gradients
    print(f"\n{Colors.OKCYAN}Gradient comparison for used tokens:{Colors.ENDC}")
    for idx in np.unique(token_indices):
        russell_grad = tokenizer.word_embedding_weights.grad[idx]
        pytorch_grad = torch_tokenizer.word_embedding.weight.grad[idx].numpy()
        diff = np.abs(russell_grad - pytorch_grad).max()
        print(f"Token {idx}: max diff = {diff:.2e}")

    print(f"\n{Colors.WARNING}=== END GRADIENT DEBUG ==={Colors.ENDC}\n")

    # Compare outputs
    print_subsection("Output Comparisons")
    
    output_tok_close = np.allclose(output_tok.data, output_tok_torch.detach().numpy(), atol=tolerance)
    print_test_result("Forward output", output_tok_close, tolerance)
    
    padding_mask_tok_close = np.array_equal(padding_mask_tok, padding_mask_tok_torch)
    print_test_result("Padding mask", padding_mask_tok_close, tolerance)
    
    loss_tok_close = np.allclose(loss_tok.data, loss_tok_torch.item(), atol=tolerance)
    print_test_result("Loss value", loss_tok_close, tolerance)
    
    overall_tok_passed = (output_tok_close and padding_mask_tok_close and loss_tok_close)
    print(f"\n{Colors.BOLD}Overall Tokenizer Test: {Colors.OKGREEN if overall_tok_passed else Colors.FAIL}{'PASSED' if overall_tok_passed else 'FAILED'}{Colors.ENDC}")
    
    # Test Transformer (simplified test with fewer layers)
    print_section_header("TRANSFORMER GRADIENT TESTING")
    
    n_layers = 2
    n_attn_heads = 2
    ff_scale_factor = 2
    
    print(f"{Colors.OKBLUE}Test Configuration:{Colors.ENDC}")
    print(f"  Number of layers: {n_layers}")
    print(f"  Number of attention heads: {n_attn_heads}")
    print(f"  FF scale factor: {ff_scale_factor}")
    print(f"  Tolerance: {tolerance:.0e}")
    
    # RussellGrad version
    transformer_tokenizer = Tokenizer(vocab_size, max_seq_len, word_embedding_dim)
    transformer = Transformer(transformer_tokenizer, n_layers, n_attn_heads, ff_scale_factor)
    
    test_strings_transformer = ["hi", "bye"]
    output_transformer = transformer.forward(test_strings_transformer)
    loss_transformer = output_transformer.sum()
    loss_transformer.backward()
    
    # PyTorch version
    class PyTorchTransformer(torch.nn.Module):
        def __init__(self, tokenizer, n_layers, n_attn_heads, ff_scale_factor):
            super().__init__()
            self.tokenizer = tokenizer
            self.n_layers = n_layers
            hidden_size = tokenizer.pos_embedding.size(-1)
            
            self.attn_layers = torch.nn.ModuleList([
                PyTorchAttentionLayer(hidden_size, n_attn_heads) for _ in range(n_layers)
            ])
            self.ff_layers = torch.nn.ModuleList([
                PyTorchFFLayer(hidden_size, hidden_size * ff_scale_factor) for _ in range(n_layers)
            ])
            self.layer_norms = torch.nn.ModuleList([
                torch.nn.ModuleList([
                    torch.nn.LayerNorm(hidden_size, dtype=torch.float64),
                    torch.nn.LayerNorm(hidden_size, dtype=torch.float64)
                ]) for _ in range(n_layers)
            ])
            
        def forward(self, input_strings):
            positionally_encoded_words, padding_mask = self.tokenizer(input_strings)
            last_layer_output = positionally_encoded_words
            
            for layer_index in range(self.n_layers):
                # Attention block with residual connection and layer norm
                attn_input = self.layer_norms[layer_index][0](last_layer_output)
                attn_output = self.attn_layers[layer_index](attn_input, padding_mask)
                last_layer_output = last_layer_output + attn_output
                
                # FF block with residual connection and layer norm
                ff_input = self.layer_norms[layer_index][1](last_layer_output)
                ff_output = self.ff_layers[layer_index](ff_input)
                last_layer_output = last_layer_output + ff_output
            
            # Convert to logits with tied embedding weights
            return torch.matmul(last_layer_output, self.tokenizer.word_embedding.weight.T)
    
    # Initialize PyTorch model with same weights
    torch_transformer_tokenizer = PyTorchTokenizer(vocab_size, max_seq_len, word_embedding_dim)
    with torch.no_grad():
        torch_transformer_tokenizer.word_embedding.weight.copy_(torch.tensor(transformer_tokenizer.word_embedding_weights.data, dtype=torch.float64))
        torch_transformer_tokenizer.pos_embedding.copy_(torch.tensor(transformer_tokenizer.absolute_positional_encoding.data, dtype=torch.float64))
    
    torch_transformer = PyTorchTransformer(torch_transformer_tokenizer, n_layers, n_attn_heads, ff_scale_factor)
    
    # Copy weights from RussellGrad to PyTorch
    with torch.no_grad():
        for layer_idx in range(n_layers):
            # Copy attention layer weights
            for head_idx in range(n_attn_heads):
                torch_transformer.attn_layers[layer_idx].heads[head_idx].W_Q.weight.copy_(
                    torch.tensor(transformer.attn_layers[layer_idx].heads[head_idx].W_Q.data.T, dtype=torch.float64))
                torch_transformer.attn_layers[layer_idx].heads[head_idx].W_K.weight.copy_(
                    torch.tensor(transformer.attn_layers[layer_idx].heads[head_idx].W_K.data.T, dtype=torch.float64))
                torch_transformer.attn_layers[layer_idx].heads[head_idx].W_V.weight.copy_(
                    torch.tensor(transformer.attn_layers[layer_idx].heads[head_idx].W_V.data.T, dtype=torch.float64))
            
            # Copy FF layer weights
            torch_transformer.ff_layers[layer_idx].up_proj.weight.copy_(
                torch.tensor(transformer.ff_layers[layer_idx].up_proj_weights.data.T, dtype=torch.float64))
            torch_transformer.ff_layers[layer_idx].up_proj.bias.copy_(
                torch.tensor(transformer.ff_layers[layer_idx].up_proj_bias.data, dtype=torch.float64))
            torch_transformer.ff_layers[layer_idx].down_proj.weight.copy_(
                torch.tensor(transformer.ff_layers[layer_idx].down_proj_weights.data.T, dtype=torch.float64))
            torch_transformer.ff_layers[layer_idx].down_proj.bias.copy_(
                torch.tensor(transformer.ff_layers[layer_idx].down_proj_bias.data, dtype=torch.float64))
            
            # Copy layer norm weights
            torch_transformer.layer_norms[layer_idx][0].weight.copy_(
                torch.tensor(transformer.layer_norms[layer_idx][0].gamma.data, dtype=torch.float64))
            torch_transformer.layer_norms[layer_idx][0].bias.copy_(
                torch.tensor(transformer.layer_norms[layer_idx][0].beta.data, dtype=torch.float64))
            torch_transformer.layer_norms[layer_idx][1].weight.copy_(
                torch.tensor(transformer.layer_norms[layer_idx][1].gamma.data, dtype=torch.float64))
            torch_transformer.layer_norms[layer_idx][1].bias.copy_(
                torch.tensor(transformer.layer_norms[layer_idx][1].beta.data, dtype=torch.float64))
    
    # Forward pass
    output_transformer_torch = torch_transformer(test_strings_transformer)
    loss_transformer_torch = output_transformer_torch.sum()
    loss_transformer_torch.backward()
    
    # Compare outputs (gradients would be too many to check individually)
    print_subsection("Output Comparisons")
    
    output_transformer_close = np.allclose(output_transformer.data, output_transformer_torch.detach().numpy(), atol=tolerance)
    print_test_result("Forward output", output_transformer_close, tolerance)
    
    loss_transformer_close = np.allclose(loss_transformer.data, loss_transformer_torch.item(), atol=tolerance)
    print_test_result("Loss value", loss_transformer_close, tolerance)
    
    # Check a few key gradients
    print_subsection("Key Gradient Comparisons")
    
    tokenizer_word_embed_grad_close = np.allclose(
        transformer.tokenizer.word_embedding_weights.grad, 
        torch_transformer.tokenizer.word_embedding.weight.grad.numpy(), 
        atol=tolerance
    )
    print_test_result("Tokenizer word embedding gradient", tokenizer_word_embed_grad_close, tolerance)
    
    first_layer_first_head_wq_grad_close = np.allclose(
        transformer.attn_layers[0].heads[0].W_Q.grad,
        torch_transformer.attn_layers[0].heads[0].W_Q.weight.grad.numpy().T,
        atol=tolerance
    )
    print_test_result("First layer, first head W_Q gradient", first_layer_first_head_wq_grad_close, tolerance)
    
    overall_transformer_passed = (output_transformer_close and loss_transformer_close and 
                                 tokenizer_word_embed_grad_close and first_layer_first_head_wq_grad_close)
    print(f"\n{Colors.BOLD}Overall Transformer Test: {Colors.OKGREEN if overall_transformer_passed else Colors.FAIL}{'PASSED' if overall_transformer_passed else 'FAILED'}{Colors.ENDC}")
    
    # Final summary
    print_section_header("TEST SUMMARY")
    
    all_tests_passed = (wq_grad_close and wk_grad_close and wv_grad_close and input_grad_close and 
                       output_close and loss_close and overall_layer_passed and overall_ff_passed and
                       overall_ln_passed and overall_tok_passed and overall_transformer_passed)
    
    if all_tests_passed:
        print(f"{Colors.OKGREEN}{Colors.BOLD}🎉 ALL TESTS PASSED! 🎉{Colors.ENDC}")
        print(f"{Colors.OKGREEN}Your RussellGrad implementation matches PyTorch perfectly!{Colors.ENDC}")
    else:
        print(f"{Colors.FAIL}{Colors.BOLD}❌ SOME TESTS FAILED ❌{Colors.ENDC}")
        print(f"{Colors.WARNING}Check the failed tests above for debugging.{Colors.ENDC}")
    
    print(f"\n{Colors.OKCYAN}Test completed with tolerance: {tolerance:.0e}{Colors.ENDC}")
