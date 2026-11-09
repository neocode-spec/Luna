
"""
WANING 1.5
GPT-2 based language model configuration.

Architecture:
- 13 transformer layers
- 640 hidden dimensions
- 10 attention heads
- 1024 context length
- 16000 vocabulary size

Model weights and tokenizer are loaded from the
Hugging Face checkpoint in the project directory.
"""

import os
import torch

from transformers import (
    GPT2Config,
    GPT2LMHeadModel,
    AutoTokenizer,
    AutoModelForCausalLM,
)

# ============================================================
# CONFIGURATION
# ============================================================

MODEL_NAME = "WANING 1.5"
MODEL_VERSION = "1.5.0"

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

VOCAB_SIZE = 16000
CONTEXT_LENGTH = 1024
EMBEDDING_SIZE = 640
NUM_LAYERS = 13
NUM_HEADS = 10
INNER_SIZE = 2560

DEVICE = torch.device("cpu")


# ============================================================
# DEVICE CONFIGURATION
# ============================================================

torch.set_num_threads(1)


# ============================================================
# MODEL CONFIGURATION
# ============================================================

def create_config():

    return GPT2Config(
        vocab_size=VOCAB_SIZE,
        n_positions=CONTEXT_LENGTH,
        n_ctx=CONTEXT_LENGTH,
        n_embd=EMBEDDING_SIZE,
        n_layer=NUM_LAYERS,
        n_head=NUM_HEADS,
        n_inner=INNER_SIZE,
        activation_function="gelu_new",
        resid_pdrop=0.1,
        embd_pdrop=0.1,
        attn_pdrop=0.1,
        layer_norm_epsilon=1e-5,
        initializer_range=0.02,
        use_cache=True,
        bos_token_id=2,
        eos_token_id=3,
        pad_token_id=0
    )


# ============================================================
# LOAD TOKENIZER
# ============================================================

def load_tokenizer():

    print("Loading WANING 1.5 tokenizer...")

    tokenizer = AutoTokenizer.from_pretrained(
        BASE_DIR,
        local_files_only=True
    )

    if len(tokenizer) != VOCAB_SIZE:
        raise ValueError(
            f"Expected {VOCAB_SIZE} tokenizer tokens, "
            f"found {len(tokenizer)}."
        )

    return tokenizer


# ============================================================
# LOAD TRAINED MODEL
# ============================================================

def load_model():

    print("Loading WANING 1.5 checkpoint...")

    model = AutoModelForCausalLM.from_pretrained(
        BASE_DIR,
        local_files_only=True
    )

    config = model.config

    expected = {
        "vocab_size": VOCAB_SIZE,
        "n_positions": CONTEXT_LENGTH,
        "n_embd": EMBEDDING_SIZE,
        "n_layer": NUM_LAYERS,
        "n_head": NUM_HEADS,
    }

    for key, expected_value in expected.items():

        actual_value = getattr(config, key)

        if actual_value != expected_value:
            raise ValueError(
                f"Architecture mismatch: {key}. "
                f"Expected {expected_value}, "
                f"found {actual_value}."
            )

    model.to(DEVICE)
    model.eval()

    print("WANING 1.5 loaded successfully.")

    print(
        "Parameters:",
        f"{sum(p.numel() for p in model.parameters()):,}"
    )

    return model


# ============================================================
# MODEL INFORMATION
# ============================================================

def get_model_info(model):

    return {
        "name": MODEL_NAME,
        "version": MODEL_VERSION,
        "parameters": sum(
            p.numel() for p in model.parameters()
        ),
        "vocabulary": model.config.vocab_size,
        "layers": model.config.n_layer,
        "embedding_size": model.config.n_embd,
        "attention_heads": model.config.n_head,
        "context_length": model.config.n_positions,
        "device": str(DEVICE),
    }
