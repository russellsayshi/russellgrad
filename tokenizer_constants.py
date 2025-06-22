# we are making a really simple tokenizer here, just preprocess all this at the top of the file
vocabulary = list("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ 0123456789.,!?")
vocabulary_map = {}
for i, c in enumerate(vocabulary):
    vocabulary_map[c] = i
n_words = len(vocabulary)
pad_token_id = n_words # tokenizer uses last index as pad