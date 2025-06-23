import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
import numpy as np
import math
from typing import List, Tuple

################################################################################
#################################### NOTE ######################################
################################################################################
# Had GPT write this file with me, it's a reimplemntation of all of my stuff but in pytorch so that I can check to make sure it matches the performance

# Set random seeds for reproducibility
torch.manual_seed(42)
np.random.seed(42)

# Constants from tokenizer_constants.py
vocabulary = list("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ 0123456789.,!?")
vocabulary_map = {c: i for i, c in enumerate(vocabulary)}
n_words = len(vocabulary)
pad_token_id = n_words  # tokenizer uses last index as pad

# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


class PyTorchTokenizer(nn.Module):
    def __init__(self, vocab_size: int, max_seq_len: int, word_embedding_dim: int, pad_token_id: int):
        super().__init__()
        self.word_embedding_dim = word_embedding_dim
        self.pad_token_id = pad_token_id
        self.max_seq_len = max_seq_len
        
        # Word embeddings with padding token
        embed_std = 1 / math.sqrt(vocab_size)
        self.word_embedding_weights = nn.Embedding(vocab_size + 1, word_embedding_dim, padding_idx=pad_token_id)
        # Initialize embeddings
        nn.init.normal_(self.word_embedding_weights.weight, mean=0, std=embed_std)
        # Set padding embedding to zero
        with torch.no_grad():
            self.word_embedding_weights.weight[pad_token_id] = 0
        
        # Absolute positional encoding
        self.absolute_positional_encoding = nn.Parameter(
            torch.normal(0, embed_std, (max_seq_len, word_embedding_dim))
        )
    
    def forward(self, input_strings: List[str]) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        max_len = max([len(x) for x in input_strings])
        
        # Strip invalid tokens
        strings_with_invalid_tokens_stripped = [
            "".join([c for c in input_string if c in vocabulary_map]) for input_string in input_strings
        ]
        
        # Convert to token indices
        token_indices = []
        for input_string in strings_with_invalid_tokens_stripped:
            indices = [vocabulary_map[c] for c in input_string]
            indices += [self.pad_token_id] * (max_len - len(input_string))
            token_indices.append(indices)
        
        token_indices = torch.tensor(token_indices, dtype=torch.long, device=device)
        
        # Get word embeddings
        word_embeddings = self.word_embedding_weights(token_indices)
        
        # Add positional encoding
        positionally_encoded_words = word_embeddings + self.absolute_positional_encoding[:max_len, :]
        
        # Create padding mask
        padding_mask = token_indices == self.pad_token_id
        
        return positionally_encoded_words, token_indices, padding_mask


class MultiHeadAttention(nn.Module):
    def __init__(self, hidden_size: int, n_heads: int):
        super().__init__()
        assert hidden_size % n_heads == 0
        self.hidden_size = hidden_size
        self.n_heads = n_heads
        self.per_head_dim = hidden_size // n_heads
        
        # Linear projections for Q, K, V
        self.W_Q = nn.Linear(hidden_size, hidden_size, bias=False)
        self.W_K = nn.Linear(hidden_size, hidden_size, bias=False)
        self.W_V = nn.Linear(hidden_size, hidden_size, bias=False)
        
        # Initialize weights with small values
        nn.init.normal_(self.W_Q.weight, mean=0, std=0.01)
        nn.init.normal_(self.W_K.weight, mean=0, std=0.01)
        nn.init.normal_(self.W_V.weight, mean=0, std=0.01)
    
    def forward(self, x: torch.Tensor, padding_mask: torch.Tensor):
        batch_size, seq_len, _ = x.shape
        
        # Project to Q, K, V
        Q = self.W_Q(x)  # (batch, seq_len, hidden_size)
        K = self.W_K(x)  # (batch, seq_len, hidden_size)
        V = self.W_V(x)  # (batch, seq_len, hidden_size)
        
        # Reshape for multi-head attention
        Q = Q.view(batch_size, seq_len, self.n_heads, self.per_head_dim).transpose(1, 2)  # (batch, n_heads, seq_len, per_head_dim)
        K = K.view(batch_size, seq_len, self.n_heads, self.per_head_dim).transpose(1, 2)
        V = V.view(batch_size, seq_len, self.n_heads, self.per_head_dim).transpose(1, 2)
        
        # Compute attention scores
        attn_scores = torch.matmul(Q, K.transpose(-1, -2)) / math.sqrt(self.per_head_dim)  # (batch, n_heads, seq_len, seq_len)
        
        # Apply causal mask
        causal_mask = torch.triu(torch.ones(seq_len, seq_len, device=device), diagonal=1).bool()
        attn_scores = attn_scores.masked_fill(causal_mask, float('-inf'))
        
        # Apply padding mask
        padding_mask = padding_mask.unsqueeze(1).unsqueeze(2)  # (batch, 1, 1, seq_len)
        attn_scores = attn_scores.masked_fill(padding_mask, float('-inf'))
        
        # Softmax
        attn_weights = F.softmax(attn_scores, dim=-1)
        
        # Apply attention to values
        attn_output = torch.matmul(attn_weights, V)  # (batch, n_heads, seq_len, per_head_dim)
        
        # Concatenate heads
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, seq_len, self.hidden_size)
        
        return attn_output


class FeedForward(nn.Module):
    def __init__(self, hidden_size: int, ff_scale_factor: int):
        super().__init__()
        self.up_proj = nn.Linear(hidden_size, hidden_size * ff_scale_factor)
        self.down_proj = nn.Linear(hidden_size * ff_scale_factor, hidden_size)
        
        # Initialize weights
        nn.init.normal_(self.up_proj.weight, mean=0, std=0.01)
        nn.init.zeros_(self.up_proj.bias)
        nn.init.normal_(self.down_proj.weight, mean=0, std=0.01)
        nn.init.zeros_(self.down_proj.bias)
    
    def forward(self, x: torch.Tensor):
        x = self.up_proj(x)
        x = F.leaky_relu(x, negative_slope=0.01)
        x = self.down_proj(x)
        return x


class TransformerBlock(nn.Module):
    def __init__(self, hidden_size: int, n_heads: int, ff_scale_factor: int):
        super().__init__()
        self.attention = MultiHeadAttention(hidden_size, n_heads)
        self.feed_forward = FeedForward(hidden_size, ff_scale_factor)
        self.layer_norm1 = nn.LayerNorm(hidden_size, eps=1e-5)
        self.layer_norm2 = nn.LayerNorm(hidden_size, eps=1e-5)
    
    def forward(self, x: torch.Tensor, padding_mask: torch.Tensor):
        # First sub-layer: multi-head attention with residual connection
        x_norm = self.layer_norm1(x)
        attn_output = self.attention(x_norm, padding_mask)
        x = x + attn_output
        
        # Second sub-layer: feed-forward with residual connection
        x_norm = self.layer_norm2(x)
        ff_output = self.feed_forward(x_norm)
        x = x + ff_output
        
        return x


class PyTorchTransformer(nn.Module):
    def __init__(self, tokenizer: PyTorchTokenizer, n_layers: int, n_attn_heads: int, ff_scale_factor: int = 4):
        super().__init__()
        self.tokenizer = tokenizer
        self.hidden_size = tokenizer.word_embedding_dim
        self.n_layers = n_layers
        
        # Transformer blocks
        self.blocks = nn.ModuleList([
            TransformerBlock(self.hidden_size, n_attn_heads, ff_scale_factor)
            for _ in range(n_layers)
        ])
    
    def forward(self, input_strings: List[str]):
        positionally_encoded_words, token_indices, padding_mask = self.tokenizer(input_strings)
        return self.forward_tokenized(positionally_encoded_words, padding_mask)
    
    def forward_tokenized(self, positionally_encoded_words: torch.Tensor, padding_mask: torch.Tensor):
        x = positionally_encoded_words
        
        # Pass through transformer blocks
        for block in self.blocks:
            x = block(x, padding_mask)
        
        # Convert to logits with tied embedding weights
        logits = F.linear(x, self.tokenizer.word_embedding_weights.weight[:-1])  # Exclude padding token from output
        
        return logits


def make_target(input_tokens: torch.Tensor) -> torch.Tensor:
    """Shift tokens left by 1 for next-token prediction"""
    return torch.cat([input_tokens[:, 1:], torch.full((input_tokens.shape[0], 1), pad_token_id, device=device)], dim=1)


def generate(tokenizer: PyTorchTokenizer, transformer: PyTorchTransformer, prompt: str = "The", num_tokens: int = 40, temperature: float = 0.5) -> str:
    """Generate text from the model"""
    transformer.eval()
    
    with torch.no_grad():
        current_string = prompt
        
        for _ in range(num_tokens):
            # Tokenize current string
            positionally_encoded_words, token_indices, padding_mask = tokenizer([current_string])
            
            # Get model output
            logits = transformer.forward_tokenized(positionally_encoded_words, padding_mask)
            
            # Get logits for the last non-padded token
            seq_len = len(current_string)
            next_token_logits = logits[0, seq_len - 1, :]
            
            # Apply temperature
            next_token_logits = next_token_logits / temperature
            
            # Sample from the distribution
            probs = F.softmax(next_token_logits, dim=-1)
            next_token_id = torch.multinomial(probs, 1).item()
            
            # Convert to character and append
            if next_token_id < len(vocabulary):
                next_char = vocabulary[next_token_id]
                current_string += next_char
            else:
                break
    
    transformer.train()
    return current_string


def load_dataset():
    with open("datasets/shakespeare.txt", "r") as f:
        result = f.read()
    
    # Split dataset into chunks of 64 characters
    chunk_len = 64
    chunks = [result[i:i+chunk_len] for i in range(0, len(result), chunk_len)]
    
    # Split chunks into batches
    batch_size = 64
    batches = [chunks[i:i+batch_size] for i in range(0, len(chunks), batch_size)]
    
    return batches


class CosineAnnealingWithWarmupLR(torch.optim.lr_scheduler._LRScheduler):
    """Cosine annealing with linear warmup"""
    def __init__(self, optimizer, total_steps, warmup_steps, min_lr=0):
        self.total_steps = total_steps
        self.warmup_steps = warmup_steps
        self.min_lr = min_lr
        self.base_lr = optimizer.param_groups[0]['lr']
        super().__init__(optimizer)
    
    def get_lr(self):
        if self.last_epoch < self.warmup_steps:
            # Linear warmup
            return [self.base_lr * self.last_epoch / self.warmup_steps for _ in self.optimizer.param_groups]
        else:
            # Cosine annealing
            progress = (self.last_epoch - self.warmup_steps) / (self.total_steps - self.warmup_steps)
            return [self.min_lr + (self.base_lr - self.min_lr) * 0.5 * (1 + math.cos(math.pi * progress))
                    for _ in self.optimizer.param_groups]


def train():
    # Constants
    n_layers = 8
    n_attn_heads = 8
    ff_scale_factor = 4
    max_seq_len = 64
    word_embedding_dim = 384
    
    # Training hyperparameters
    num_epochs = 20
    learning_rate = 1e-3
    min_lr = 1e-5
    warmup_steps = 5
    
    # Initialize model components
    tokenizer = PyTorchTokenizer(n_words, max_seq_len, word_embedding_dim, pad_token_id)
    transformer = PyTorchTransformer(tokenizer, n_layers, n_attn_heads, ff_scale_factor)
    transformer.to(device)
    
    # Load dataset
    batches = load_dataset()
    total_steps = num_epochs * len(batches)
    
    # Initialize optimizer
    optimizer = optim.Adam(transformer.parameters(), lr=learning_rate)
    
    # Initialize learning rate scheduler
    scheduler = CosineAnnealingWithWarmupLR(optimizer, total_steps, warmup_steps, min_lr)
    
    # Training loop
    global_step = 0
    epoch = 0
    
    while epoch < num_epochs:
        for i, batch in enumerate(batches):
            if global_step % 20 == 0:
                # Generate sample output every 20 steps
                print("SAMPLED STRING:", generate(tokenizer, transformer))
            
            # Zero gradients
            optimizer.zero_grad()
            
            # Forward pass
            positionally_encoded_words, token_indices, padding_mask = tokenizer(batch)
            targets = make_target(token_indices)
            
            output = transformer.forward_tokenized(positionally_encoded_words, padding_mask)
            
            # Compute loss (cross entropy with ignore_index for padding)
            loss = F.cross_entropy(
                output.view(-1, n_words),  # Flatten to (batch*seq_len, vocab_size)
                targets.view(-1),          # Flatten to (batch*seq_len,)
                ignore_index=pad_token_id
            )
            
            # Print progress
            current_lr = scheduler.get_lr()[0]
            print(f'epoch {epoch} | batch {i}/{len(batches)} | lr {current_lr:.5f} | loss {loss.item():.4f}')
            
            # Backward pass
            loss.backward()
            
            # Update weights
            optimizer.step()
            scheduler.step()
            
            global_step += 1
        
        epoch += 1
    
    print("Training completed!")


if __name__ == "__main__":
    train() 