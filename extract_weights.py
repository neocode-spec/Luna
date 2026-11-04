"""
extract_weights.py  -  turns your big luna_1.4_ckpt.pt into a small luna_1.4_weights.pt

A checkpoint file often carries extra training data (optimizer state), which makes it
too big for normal GitHub uploads. The server only needs the weights.

Run in the folder that contains luna_1.4_ckpt.pt:
    py extract_weights.py
"""

import os
import torch

SOURCE = "luna_1.4_ckpt.pt"
TARGET = "luna_1.4_weights.pt"

if not os.path.exists(SOURCE):
    raise FileNotFoundError(f"{SOURCE} not found. Run this in the folder that has it.")

try:
    raw = torch.load(SOURCE, map_location="cpu")
except Exception:
    raw = torch.load(SOURCE, map_location="cpu", weights_only=False)

state = raw
if isinstance(state, dict):
    for key in ("model_state_dict", "state_dict", "model"):
        if key in state and isinstance(state[key], dict):
            state = state[key]
            break

state = {(k[7:] if k.startswith("module.") else k): v for k, v in state.items()}
if state and all(k.startswith("model.") for k in state):
    state = {k[6:]: v for k, v in state.items()}

torch.save(state, TARGET)

params = sum(v.numel() for v in state.values() if hasattr(v, "numel")) / 1e6
print(f"Saved {TARGET}: {params:.2f}M values, {os.path.getsize(TARGET) / 1e6:.1f} MB")
print("First keys:", ", ".join(list(state.keys())[:6]))