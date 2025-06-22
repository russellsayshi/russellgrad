import numpy as np
import torch
import torch.nn as nn
from tensor import Tensor

# Test LayerNorm gradient computation
batch_size = 2
seq_len = 4
embed_dim = 16

# Create LayerNorm implementations
class SimpleLayerNorm:
    def __init__(self, dim):
        self.gamma = Tensor(np.ones((dim,)))
        self.beta = Tensor(np.zeros((dim,)))
        
    def forward_layer(self, x):
        mean = x.mean(axis=-1, keepdims=True)
        variance = (x ** 2).mean(axis=-1, keepdims=True) - mean ** 2
        normalized = (x - mean) / (variance + 1e-6).sqrt()
        transformed = normalized * self.gamma + self.beta
        return transformed

    def forward(self, x: Tensor):
        shape = x.shape
        if len(shape) != 3:
            raise ValueError("expected input of (batch, token, embed)")
        mean = x.mean(axis=-1, keepdims=True)
        variance = ((x - mean) ** 2).mean(axis=-1, keepdims=True)
        normalized = (x - mean) / (variance + 1e-5).sqrt()

        transformed = normalized * self.gamma + self.beta
        return transformed

# RussellGrad version
layer_norm = SimpleLayerNorm(embed_dim)
input_data = Tensor(np.random.randn(batch_size, seq_len, embed_dim))

# Store intermediate values for debugging
mean = input_data.mean(axis=-1, keepdims=True)
variance = (input_data ** 2).mean(axis=-1, keepdims=True) - mean ** 2
normalized = (input_data - mean) / (variance + 1e-6).sqrt()

output = layer_norm.forward(input_data)
loss = output.sum()
loss.backward()

# PyTorch version
torch_layer_norm = nn.LayerNorm(embed_dim, dtype=torch.float64)
with torch.no_grad():
    torch_layer_norm.weight.copy_(torch.tensor(layer_norm.gamma.data, dtype=torch.float64))
    torch_layer_norm.bias.copy_(torch.tensor(layer_norm.beta.data, dtype=torch.float64))

input_torch = torch.tensor(input_data.data, dtype=torch.float64, requires_grad=True)
output_torch = torch_layer_norm(input_torch)
loss_torch = output_torch.sum()
loss_torch.backward()

# Debug prints
print("Shapes:")
print(f"  Input: {input_data.shape}")
print(f"  Gamma: {layer_norm.gamma.shape}")
print(f"  Beta: {layer_norm.beta.shape}")
print(f"  Normalized: {normalized.shape}")
print(f"  Output: {output.shape}")

print("\nGradients:")
print(f"  RussellGrad gamma grad shape: {layer_norm.gamma.grad.shape}")
print(f"  PyTorch gamma grad shape: {torch_layer_norm.weight.grad.shape}")
print(f"  RussellGrad gamma grad: {layer_norm.gamma.grad}")
print(f"  PyTorch gamma grad: {torch_layer_norm.weight.grad.numpy()}")
print(f"  Difference: {np.abs(layer_norm.gamma.grad - torch_layer_norm.weight.grad.numpy()).max()}")

print("\nBeta gradients (for comparison):")
print(f"  RussellGrad beta grad: {layer_norm.beta.grad}")
print(f"  PyTorch beta grad: {torch_layer_norm.bias.grad.numpy()}")
print(f"  Difference: {np.abs(layer_norm.beta.grad - torch_layer_norm.bias.grad.numpy()).max()}")