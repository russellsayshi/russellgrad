# RussellGrad

A minimal neural network library inspired by [Andrej Karpathy's micrograd](https://github.com/karpathy/micrograd), built from scratch with automatic differentiation and transformer architecture implementation.

## Overview

This project implements a complete neural network framework featuring:
- **Custom Tensor class** with automatic differentiation
- **Transformer architecture** with multi-head attention
- **Training pipeline** for language modeling
- **Comprehensive testing** against PyTorch for gradient correctness

## Architecture

### Core Components

#### Tensor (`tensor.py`)
Full-featured tensor implementation with:
- Broadcasting support for element-wise operations
- Matrix multiplication with batch dimension handling
- Comprehensive automatic differentiation engine
- Operations: `+`, `-`, `*`, `/`, `**`, `@` (matmul), `tanh`, `exp`, `log`, `sqrt`
- Reduction operations: `sum`, `mean`, `max`
- Advanced operations: `masked_fill`, `masked_multiply`, `transpose`, indexing
- Gradient computation with topological sorting

#### Transformer (`russelltransformer.py`)
Complete transformer implementation:
- **Multi-head attention** with causal masking and padding support
- **Feed-forward layers** with configurable projection dimensions
- **Layer normalization** for training stability  
- **Positional encoding** for sequence understanding
- **Residual connections** throughout the architecture

#### Training Components
- **SGD Optimizer** (`optimizer.py`) with learning rate control
- **Cross-entropy loss** (`loss.py`) with padding token masking
- **Character-level tokenizer** (`tokenizer.py`) with vocabulary mapping
- **Training script** (`train/train_transformer.py`) for Shakespeare text

## Features

### Automatic Differentiation
The tensor implementation supports full backward-mode automatic differentiation:
```python
x = Tensor([2.0])
y = x ** 2 + 3 * x + 1
y.backward()
print(x.grad)  # [7.0] = 2*x + 3 = 2*2 + 3
```

### Broadcasting
Automatic broadcasting for tensor operations:
```python
a = Tensor([[1, 2], [3, 4]])  # (2, 2)
b = Tensor([10])              # (1,)
c = a + b                     # Broadcasts to (2, 2)
```

### Transformer Training
End-to-end transformer training on text data:
```python
tokenizer = Tokenizer(vocab_size, max_seq_len, embed_dim, pad_token_id)
transformer = Transformer(tokenizer, n_layers=4, n_attn_heads=4)
optimizer = SGDOptimizer(transformer.parameters(), lr=1e-2)

# Training loop
for batch in batches:
    optimizer.zero_grad()
    output = transformer.forward(batch)
    loss = cross_entropy_loss(output, targets, pad_token_id)
    loss.backward()
    optimizer.step()
```

## Testing

Comprehensive test suite (`tests/`) validates gradient computation against PyTorch:
- **Attention head gradients** - Multi-head self-attention mechanism
- **Feed-forward gradients** - Up/down projection layers
- **Layer norm gradients** - Normalization layer parameters
- **Tokenizer gradients** - Embedding and positional encoding
- **Full transformer gradients** - End-to-end architecture

All tests pass with numerical precision tolerance of `1e-12`, ensuring mathematical correctness.

## File Structure

```
micrograd/
├── tensor.py                    # Core tensor with autograd
├── manual_tensor.py            # Simplified tensor (no broadcasting)
├── russelltransformer.py       # Transformer architecture
├── tokenizer.py                # Character-level tokenizer
├── tokenizer_constants.py      # Vocabulary definitions
├── softmax.py                  # Softmax implementation
├── loss.py                     # Cross-entropy loss
├── optimizer.py                # SGD optimizer
├── train/
│   └── train_transformer.py    # Training script
└── tests/
    ├── test_layernorm.py       # Layer normalization tests
    └── test_russelltransformer.py # Full architecture tests
```

## Usage

### Basic Tensor Operations
```python
from tensor import Tensor
import numpy as np

# Create tensors
x = Tensor([1.0, 2.0, 3.0])
y = Tensor([4.0, 5.0, 6.0])

# Operations with automatic differentiation
z = x @ y.transpose((0,))  # Dot product
z.backward()
print(x.grad)  # Gradients computed automatically
```

### Training a Transformer
```python
from train.train_transformer import train

# Start training on Shakespeare dataset
train()
```

### Running Tests
```python
# Test individual components
python tests/test_layernorm.py
python tests/test_russelltransformer.py

# Or run from the test files directly
python -m tests.test_russelltransformer
```

## Implementation Notes

- **Broadcasting**: Full NumPy-compatible broadcasting with gradient support
- **Memory efficiency**: Gradient accumulation with proper shape handling
- **Numerical stability**: Epsilon values and careful computation order
- **Modular design**: Clean separation between tensor ops and neural network layers
- **PyTorch compatibility**: Gradients match PyTorch implementation exactly

## Educational Value

This implementation serves as an excellent educational resource for understanding:
- How automatic differentiation works under the hood
- Transformer architecture implementation details
- Gradient computation in complex neural networks
- The relationship between forward and backward passes
- Broadcasting and tensor operation mechanics

The codebase prioritizes clarity and correctness over performance, making it ideal for learning the fundamentals of deep learning frameworks.