import numpy as np
import tiktoken

# Read the binary file
train_data = np.fromfile('train.bin', dtype=np.uint16)

print(f"Total tokens: {len(train_data)}")
print(f"Unique tokens: {len(set(train_data))}")
print(f"First few tokens: {train_data[:10]}")
print(f"File size: {train_data.nbytes / 1024:.2f} KB")

# Initialize the tokenizer
enc = tiktoken.get_encoding("gpt2")

# Print first 20 tokens and their decoded text
for token in train_data[:20]:
    print(f"Token: {token} -> Text: {enc.decode([token])}")