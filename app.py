
import os
import torch

from transformers import AutoTokenizer, AutoModelForCausalLM
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field


# ============================================================
# WANING 1.5 CONFIGURATION
# ============================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = BASE_DIR

MODEL_NAME = "WANING 1.5"
MODEL_VERSION = "1.5.0"

MAX_NEW_TOKENS = 40
MAX_INPUT_TOKENS = 1024

DEVICE = torch.device("cpu")

torch.set_num_threads(1)

print("=" * 55)
print("INITIALIZING WANING 1.5")
print("=" * 55)


# ============================================================
# LOAD TOKENIZER
# ============================================================

print("Loading WANING 1.5 tokenizer...")

try:
    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_DIR,
        local_files_only=True
    )

except Exception as error:
    raise RuntimeError(
        f"Tokenizer loading failed: {error}"
    ) from error


# ============================================================
# LOAD MODEL
# ============================================================

print("Loading WANING 1.5 model...")

try:
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_DIR,
        local_files_only=True
    )

except Exception as error:
    raise RuntimeError(
        f"WANING 1.5 model loading failed: {error}"
    ) from error


# ============================================================
# MODEL VERIFICATION
# ============================================================

model.to(DEVICE)
model.eval()

parameter_count = sum(
    parameter.numel()
    for parameter in model.parameters()
)

embedding_size = (
    model.get_input_embeddings().weight.shape[0]
)

tokenizer_size = len(tokenizer)

print("\nMODEL VERIFICATION")

print("Model:", MODEL_NAME)
print("Parameters:", f"{parameter_count:,}")
print("Tokenizer vocabulary:", tokenizer_size)
print("Model vocabulary:", model.config.vocab_size)
print("Embedding vocabulary:", embedding_size)
print("Device:", DEVICE)

if model.config.vocab_size != embedding_size:
    raise RuntimeError(
        "Model configuration and embedding vocabulary mismatch."
    )

if tokenizer_size > embedding_size:
    raise RuntimeError(
        "Tokenizer vocabulary exceeds model embedding size."
    )

if tokenizer.pad_token_id is None:
    raise RuntimeError(
        "WANING tokenizer is missing its PAD token."
    )

if tokenizer.eos_token_id is None:
    raise RuntimeError(
        "WANING tokenizer is missing its EOS token."
    )

if tokenizer.pad_token_id >= embedding_size:
    raise RuntimeError(
        "PAD token ID exceeds model vocabulary."
    )

if tokenizer.eos_token_id >= embedding_size:
    raise RuntimeError(
        "EOS token ID exceeds model vocabulary."
    )

print("\nWANING 1.5 LOADED SUCCESSFULLY")
print("=" * 55)


# ============================================================
# FASTAPI APPLICATION
# ============================================================

app = FastAPI(
    title="WANING 1.5 API",
    description="WANING 1.5 Language Model API",
    version=MODEL_VERSION
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"]
)


# ============================================================
# REQUEST SCHEMA
# ============================================================

class PromptRequest(BaseModel):

    prompt: str = ""
    message: str = ""
    text: str = ""
    input: str = ""

    max_tokens: int = Field(
        default=40,
        ge=1,
        le=40
    )

    language: str = "english"
    model: str = "waning-1.5"
    thinking_level: str = "medium"


# ============================================================
# HEALTH CHECK
# ============================================================

@app.get("/")
def root():

    return {
        "status": "online",
        "model": MODEL_NAME,
        "version": MODEL_VERSION,
        "parameters": parameter_count,
        "device": str(DEVICE)
    }


@app.get("/health")
def health():

    return {
        "status": "healthy",
        "model": MODEL_NAME,
        "loaded": True
    }


# ============================================================
# GENERATION ENDPOINT
# ============================================================

@app.post("/generate")
@app.post("/generate/")
def generate_text(req: PromptRequest):

    user_prompt = (
        req.prompt
        or req.message
        or req.text
        or req.input
    ).strip()

    if not user_prompt:

        raise HTTPException(
            status_code=400,
            detail="A prompt is required."
        )

    tokens_to_generate = min(
        max(req.max_tokens, 1),
        MAX_NEW_TOKENS
    )

    # Conversation format used by the existing
    # WANING inference implementation.

    formatted_prompt = (
        f"User: {user_prompt}\n"
        "Assistant:"
    )

    try:

        # Tokenize input.

        inputs = tokenizer(
            formatted_prompt,
            return_tensors="pt",
            truncation=True,
            max_length=MAX_INPUT_TOKENS
        )

        inputs = {
            key: value.to(DEVICE)
            for key, value in inputs.items()
        }

        # Generate response.

        with torch.inference_mode():

            output = model.generate(
                **inputs,

                max_new_tokens=tokens_to_generate,

                do_sample=True,

                temperature=0.7,

                top_p=0.9,

                repetition_penalty=1.1,

                pad_token_id=tokenizer.pad_token_id,

                eos_token_id=tokenizer.eos_token_id,

                use_cache=True
            )

        # Extract only newly generated tokens.

        input_length = inputs["input_ids"].shape[1]

        generated_tokens = output[0][input_length:]

        response = tokenizer.decode(
            generated_tokens,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=True
        ).strip()

        # Stop at the beginning of another conversation turn.

        stop_markers = [
            "\nUser:",
            "\nHuman:",
            "\nuser:",
            "\nHuman ",
        ]

        for marker in stop_markers:

            if marker in response:

                response = response.split(
                    marker,
                    1
                )[0].strip()

        return {
            "response": response,
            "model": "waning-1.5",
            "version": MODEL_VERSION
        }

    except Exception as error:

        print("GENERATION ERROR:", str(error))

        raise HTTPException(
            status_code=500,
            detail="WANING 1.5 generation failed."
        )


# ============================================================
# STARTUP INFORMATION
# ============================================================

@app.on_event("startup")
async def startup_event():

    print("=" * 55)
    print("WANING 1.5 API READY")
    print("Model:", MODEL_NAME)
    print("Parameters:", f"{parameter_count:,}")
    print("Device:", DEVICE)
    print("=" * 55)
