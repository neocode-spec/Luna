import os
import torch

SRC = 'luna_1.4_weights.pt'

print("Loading Luna 1.4 weights...")
weights = torch.load(SRC, map_location='cpu')
before = os.path.getsize(SRC) / 1e6

print("Converting weights to half-precision (FP16) to reduce file size...")
# The output layer is TIED to the embedding (same tensor under two names). Convert each
# shared tensor ONCE and reuse it, otherwise torch.save stores two separate copies and
# the file barely shrinks.
converted = {}
compressed_weights = {}
for k, v in weights.items():
    if not v.is_floating_point():
        compressed_weights[k] = v
        continue
    ptr = v.data_ptr()
    if ptr not in converted:
        converted[ptr] = v.half()
    compressed_weights[k] = converted[ptr]

torch.save(compressed_weights, SRC)
after = os.path.getsize(SRC) / 1e6
print(f"Success! Luna 1.4 weights compressed and overwritten: {before:.1f} MB -> {after:.1f} MB")