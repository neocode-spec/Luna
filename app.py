
import os
import logging
import torch

from transformers import AutoTokenizer, AutoModelForCausalLM
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, ConfigDict


# ============================================================
# WANING 1.5 CONFIGURATION
# ============================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.getenv("MODEL_DIR", BASE_DIR)

MODEL_NAME = "WANING 1.5"
MODEL_VERSION = "1.5.1"

MAX_NEW_TOKENS = 300
MAX_CONTEXT_LENGTH = 1024
MAX_INPUT_TOKENS = MAX_CONTEXT_LENGTH - MAX_NEW_TOKENS

DEFAULT_NEW_TOKENS = 150

DEVICE = torch.device("cpu")

torch.set_num_threads(1)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("waning")

logger.info("=" * 55)
logger.info("INITIALIZING WANING 1.5")
logger.info("=" * 55)


# ============================================================
# LOAD TOKENIZER
# ============================================================

logger.info("Loading WANING tokenizer...")

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

logger.info("Loading WANING model...")

try:
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_DIR,
        local_files_only=True,
        use_safetensors=True
    )

except Exception as error:
    raise RuntimeError(
        f"Model loading failed: {error}"
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

embedding_size = model.get_input_embeddings().weight.shape[0]
tokenizer_size = len(tokenizer)

logger.info("MODEL VERIFICATION")
logger.info("Name: %s", MODEL_NAME)
logger.info("Parameters: %s", f"{parameter_count:,}")
logger.info("Tokenizer vocabulary: %s", tokenizer_size)
logger.info("Model vocabulary: %s", model.config.vocab_size)
logger.info("Embedding vocabulary: %s", embedding_size)
logger.info("Context length: %s", model.config.n_positions)
logger.info("Device: %s", DEVICE)

if model.config.vocab_size != embedding_size:
    raise RuntimeError("Model vocabulary and embeddings mismatch.")

if tokenizer_size != model.config.vocab_size:
    raise RuntimeError(
        f"Tokenizer/model vocabulary mismatch: "
        f"{tokenizer_size} vs {model.config.vocab_size}"
    )

if tokenizer.pad_token_id is None:
    raise RuntimeError("Tokenizer is missing PAD token.")

if tokenizer.eos_token_id is None:
    raise RuntimeError("Tokenizer is missing EOS token.")

if tokenizer.pad_token_id >= embedding_size:
    raise RuntimeError("PAD token ID exceeds model vocabulary.")

if tokenizer.eos_token_id >= embedding_size:
    raise RuntimeError("EOS token ID exceeds model vocabulary.")

if model.config.n_positions < MAX_CONTEXT_LENGTH:
    raise RuntimeError(
        "Configured generation context exceeds model context."
    )

logger.info("WANING 1.5 LOADED SUCCESSFULLY")


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

    model_config = ConfigDict(extra="ignore")

    prompt: str = ""
    message: str = ""
    text: str = ""
    input: str = ""

    max_tokens: int = Field(
        default=DEFAULT_NEW_TOKENS,
        ge=1,
        le=MAX_NEW_TOKENS
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
        "device": str(DEVICE),
        "max_tokens": MAX_NEW_TOKENS,
        "context_length": MAX_CONTEXT_LENGTH
    }


@app.get("/health")
def health():

    return {
        "status": "healthy",
        "model": MODEL_NAME,
        "version": MODEL_VERSION,
        "loaded": True
    }


@app.get("/generate")
@app.get("/generate/")
def generate_get_fallback():

    return {
        "status": "ready",
        "model": MODEL_NAME,
        "message": "Send a POST request to generate text."
    }


# ============================================================
# TEXT GENERATION
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

    # Respect the model's actual context window.

    tokens_to_generate = min(
        req.max_tokens,
        MAX_NEW_TOKENS
    )

    formatted_prompt = (
        f"User: {user_prompt}\n"
        "Assistant:"
    )

    try:

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

        input_length = inputs["input_ids"].shape[1]

        if input_length + tokens_to_generate > MAX_CONTEXT_LENGTH:
            raise HTTPException(
                status_code=400,
                detail="Prompt exceeds available model context."
            )

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

        generated_tokens = output[0][input_length:]

        response = tokenizer.decode(
            generated_tokens,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=True
        ).strip()

        stop_markers = [
            "\nUser:",
            "\nHuman:",
            "\nuser:",
            "\nHuman "
        ]

        for marker in stop_markers:

            if marker in response:
                response = response.split(marker, 1)[0].strip()

        if not response:
            logger.warning("Model returned an empty response.")

        return {
            "response": response,
            "model": "waning-1.5",
            "version": MODEL_VERSION,
            "tokens_generated": len(generated_tokens)
        }

    except HTTPException:
        raise

    except Exception:

        logger.exception("WANING generation failed.")

        raise HTTPException(
            status_code=500,
            detail="WANING 1.5 generation failed."
        )


# ============================================================
# STARTUP
# ============================================================

@app.on_event("startup")
async def startup_event():

    logger.info("=" * 55)
    logger.info("WANING 1.5 API READY")
    logger.info("Parameters: %s", f"{parameter_count:,}")
    logger.info("Device: %s", DEVICE)
    logger.info("Maximum generation: %s", MAX_NEW_TOKENS)
    logger.info("=" * 55)
