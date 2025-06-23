import numpy as np
from russelltransformer import Tokenizer, Transformer
from tokenizer_constants import n_words, pad_token_id, vocabulary, vocabulary_map
from softmax import softmax

def generate(tokenizer: Tokenizer, transformer: Transformer, prompt: str = "The", num_tokens: int = 40, temperature: float = 0.5) -> str:
    """
    Generate text by sampling tokens autoregressively from the model.
    
    Args:
        tokenizer: Pre-initialized Tokenizer instance
        transformer: Pre-initialized and trained Transformer instance
        prompt: The initial string to start generation from
        num_tokens: Number of tokens to generate
        temperature: Sampling temperature (higher = more random, lower = more deterministic)
                    0 = greedy decoding (argmax)
    
    Returns:
        The generated string (prompt + new tokens)
    """
    # Create inverse vocabulary map for converting token IDs back to characters
    inverse_vocab = {i: c for c, i in vocabulary_map.items()}
    
    # Start with the prompt
    generated_text = prompt
    
    for _ in range(num_tokens):
        # Tokenize current text (batch size 1)
        tokenized, token_indices, padding_masks = tokenizer.forward([generated_text])
        
        # Forward pass through transformer
        logits = transformer.forward_tokenized(tokenized, padding_masks)
        
        # Get logits for the last non-padding position
        # Find the last actual token position (before padding)
        seq_len = len(generated_text)
        last_token_logits = logits[0, seq_len - 1, :]  # shape: (vocab_size + 1,)
        # last_token_logits[-1] = -1e9 TODO: make this work
        
        # Temperature 0 means greedy decoding (argmax)
        if temperature == 0:
            # Don't consider padding token
            logits_numpy = last_token_logits.data.copy()
            logits_numpy[pad_token_id] = -np.inf
            next_token_id = np.argmax(logits_numpy)
        else:
            # Apply temperature scaling
            if temperature != 1.0:
                last_token_logits = last_token_logits / temperature
            
            # Convert to probabilities using softmax
            probs = softmax(last_token_logits, axis=-1)
            
            # Don't sample the padding token
            probs_numpy = probs.data.copy()
            probs_numpy[pad_token_id] = 0.0
            
            # Renormalize after removing padding token
            probs_numpy = probs_numpy / probs_numpy.sum()
            
            # Sample from the probability distribution
            next_token_id = np.random.choice(len(probs_numpy), p=probs_numpy)
        
        # Convert token ID back to character
        if next_token_id in inverse_vocab:
            next_char = inverse_vocab[next_token_id]
            generated_text += next_char
        
    return generated_text

# Example usage
if __name__ == "__main__":
    # Initialize model with same parameters as training
    n_layers = 4
    n_attn_heads = 4
    ff_scale_factor = 4
    max_seq_len = 512
    word_embedding_dim = 64

    tokenizer = Tokenizer(n_words, max_seq_len, word_embedding_dim, pad_token_id)
    transformer = Transformer(tokenizer, n_layers, n_attn_heads, ff_scale_factor)
    
    # Generate with default temperature
    result = generate(tokenizer, transformer, "The official", num_tokens=20)
    print(f"Generated text: '{result}'")
    
    # Try with different temperatures
    print("\nWith temperature 0.5 (more deterministic):")
    result_low_temp = generate(tokenizer, transformer, "The official", num_tokens=20, temperature=0.5)
    print(f"Generated text: '{result_low_temp}'")
    
    print("\nWith temperature 1.5 (more random):")
    result_high_temp = generate(tokenizer, transformer, "The official", num_tokens=20, temperature=1.5)
    print(f"Generated text: '{result_high_temp}'")
    
    print("\nWith temperature 0 (greedy decoding):")
    result_greedy = generate(tokenizer, transformer, "The official", num_tokens=20, temperature=0)
    print(f"Generated text: '{result_greedy}'")