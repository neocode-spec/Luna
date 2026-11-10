
import os
import logging
import zipfile
import shutil
from pathlib import Path

import torch

from transformers import AutoTokenizer, AutoModelForCausalLM
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, ConfigDict


# ============================================================
# WANING 1.5 CONFIGURATION
# ============================================================

BASE_DIR = Path(__file__).resolve().parent

MODEL_ARCHIVE = os.getenv("MODEL_ARCHIVE", "").strip()

EXTRACT_DIR = Path(
    os.getenv("MODEL_EXTRACT_DIR", "/tmp/waning-1.5-model")
)

MODEL_DIR = Path(
    os.getenv("MODEL_DIR", str(BASE_DIR))
)

MODEL_NAME = "WANING 1.5"
MODEL_VERSION = "1.5.1"

MAX_CONTEXT_LENGTH = 1024
MAX_NEW_TOKENS = 300
DEFAULT_NEW_TOKENS = 150

DEVICE = torch.device("cpu")

torch.set_num_threads(1)

logging.basicConfig(level=logging.INFO)

logger = logging.getLogger("waning")


# ============================================================
# LOGGING
# ============================================================

logger.info("=" * 60)
logger.info("INITIALIZING WANING 1.5")
logger.info("=" * 60)


# ============================================================
# FIND MODEL ARCHIVE
# ============================================================

def find_model_archive():

    if MODEL_ARCHIVE:

        archive = Path(MODEL_ARCHIVE)

        if not archive.is_absolute():
            archive = BASE_DIR / archive

        if not archive.is_file():
            raise FileNotFoundError(
                f"Configured model archive not found: {archive}"
            )

        return archive

    archives = list(BASE_DIR.glob("*.zip"))

    if len(archives) == 0:

        logger.info("No ZIP archive found. Checking MODEL_DIR.")

        return None

    if len(archives) > 1:

        raise RuntimeError(
            "Multiple ZIP archives found. Set MODEL_ARCHIVE "
            "to the exact WANING 1.5 archive filename."
        )

    return archives[0]


# ============================================================
# SAFE ZIP EXTRACTION
# ============================================================

def extract_model_archive(archive):

    logger.info("Found model archive: %s", archive.name)

    if not zipfile.is_zipfile(archive):

        raise RuntimeError(
            "The WANING archive is not a valid ZIP file."
        )

    if EXTRACT_DIR.exists():

        logger.info("Removing previous extracted model.")

        shutil.rmtree(EXTRACT_DIR)

    EXTRACT_DIR.mkdir(parents=True, exist_ok=True)

    with zipfile.ZipFile(archive, "r") as zip_file:

        members = zip_file.infolist()

        for member in members:

            target = (EXTRACT_DIR / member.filename).resolve()

            if not target.is_relative_to(EXTRACT_DIR.resolve()):

                raise RuntimeError(
                    "Unsafe path detected inside model archive."
                )

        logger.info(
            "Extracting %s archive entries.",
            len(members)
        )

        zip_file.extractall(EXTRACT_DIR)

    logger.info("WANING archive extracted successfully.")


# ============================================================
# LOCATE ACTUAL MODEL DIRECTORY
# ============================================================

def locate_model_directory(root):

    root = Path(root)

    for current, directories, files in os.walk(root):

        directories[:] = [
            directory
            for directory in directories
            if directory != "__MACOSX"
        ]

        file_set = set(files)

        has_config = "config.json" in file_set

        has_weights = (
            "model.safetensors" in file_set
            or "model.safetensors.index.json" in file_set
        )

        has_tokenizer = "tokenizer.json" in file_set

        if has_config and has_weights and has_tokenizer:

            return Path(current)

    raise FileNotFoundError(
        "Could not locate a complete WANING model directory. "
        "Expected config.json, tokenizer.json and "
        "model.safetensors or model.safetensors.index.json."
    )


# ============================================================
# PREPARE MODEL DIRECTORY
# ============================================================

archive = find_model_archive()

if archive is not None:

    extract_model_archive(archive)

    MODEL_DIR = locate_model_directory(EXTRACT_DIR)

else:

    MODEL_DIR = locate_model_directory(MODEL_DIR)


logger.info("MODEL DIRECTORY: %s", MODEL_DIR)


# ============================================================
# VERIFY REQUIRED FILES
# ============================================================

required_files = [
    "config.json",
    "tokenizer.json"
]

for filename in required_files:

    if not (MODEL_DIR / filename).is_file():

        raise FileNotFoundError(
            f"Missing required model file: {filename}"
        )

if not (
    (MODEL_DIR / "model.safetensors").is_file()
    or (MODEL_DIR / "model.safetensors.index.json").is_file()
):

    raise FileNotFoundError(
        "WANING model weights were not found."
    )


# ============================================================
# LOAD TOKENIZER
# ============================================================

logger.info("Loading WANING tokenizer...")

try:

    tokenizer = AutoTokenizer.from_pretrained(
        str(MODEL_DIR),
        local_files_only=True
    )

except Exception as error:

    logger.exception("Tokenizer loading failed.")

    raise RuntimeError(
        f"WANING tokenizer loading failed: {error}"
    ) from error


# ============================================================
# LOAD MODEL
# ============================================================

logger.info("Loading WANING trained weights...")

try:

    model = AutoModelForCausalLM.from_pretrained(
        str(MODEL_DIR),
        local_files_only=True,
        use_safetensors=True,
        low_cpu_mem_usage=True
    )

except Exception as error:

    logger.exception("Model loading failed.")

    raise RuntimeError(
        f"WANING model loading failed: {error}"
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

model_vocab_size = model.config.vocab_size

model_context = getattr(
    model.config,
    "n_positions",
    getattr(
        model.config,
        "max_position_embeddings",
        MAX_CONTEXT_LENGTH
    )
)

logger.info("=" * 60)
logger.info("WANING 1.5 MODEL VERIFICATION")
logger.info("=" * 60)

logger.info("Model: %s", MODEL_NAME)
logger.info("Version: %s", MODEL_VERSION)

logger.info(
    "Parameters: %s",
    f"{parameter_count:,}"
)

logger.info("Tokenizer vocabulary: %s", tokenizer_size)
logger.info("Model vocabulary: %s", model_vocab_size)
logger.info("Embedding vocabulary: %s", embedding_size)

logger.info("Model context: %s", model_context)
logger.info("Device: %s", DEVICE)


if model_vocab_size != embedding_size:

    raise RuntimeError(
        "Model vocabulary and embedding sizes do not match."
    )


if tokenizer_size != model_vocab_size:

    raise RuntimeError(
        f"Tokenizer/model vocabulary mismatch: "
        f"{tokenizer_size} vs {model_vocab_size}"
    )


if tokenizer.pad_token_id is None:

    raise RuntimeError("Tokenizer is missing PAD token.")


if tokenizer.eos_token_id is None:

    raise RuntimeError("Tokenizer is missing EOS token.")


if tokenizer.pad_token_id >= embedding_size:

    raise RuntimeError("PAD token ID exceeds model vocabulary.")


if tokenizer.eos_token_id >= embedding_size:

    raise RuntimeError("EOS token ID exceeds model vocabulary.")


if model_context < 2:

    raise RuntimeError("Invalid model context length.")


if model_context < MAX_CONTEXT_LENGTH:

    logger.warning(
        "Model context is %s. Adjusting API context.",
        model_context
    )

    MAX_CONTEXT_LENGTH = model_context


MAX_INPUT_TOKENS = (
    MAX_CONTEXT_LENGTH - MAX_NEW_TOKENS
)

MAX_INPUT_TOKENS = max(1, MAX_INPUT_TOKENS)

logger.info("=" * 60)
logger.info("WANING 1.5 LOADED SUCCESSFULLY")
logger.info("=" * 60)


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
        "context_length": MAX_CONTEXT_LENGTH,
        "archive_loaded": archive is not None
    }


@app.get("/health")
def health():

    return {
        "status": "healthy",
        "model": MODEL_NAME,
        "version": MODEL_VERSION,
        "loaded": True,
        "parameters": parameter_count
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

    tokens_to_generate = min(
        req.max_tokens,
        MAX_NEW_TOKENS,
        MAX_CONTEXT_LENGTH
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

                response = response.split(
                    marker,
                    1
                )[0].strip()

        if not response:

            logger.warning(
                "WANING returned an empty response."
            )

        return {
            "response": response,
            "model": "waning-1.5",
            "version": MODEL_VERSION,
            "tokens_generated": len(generated_tokens)
        }

    except Exception as error:

        logger.exception("WANING generation failed.")

        if isinstance(error, HTTPException):
            raise

        raise HTTPException(
            status_code=500,
            detail="WANING 1.5 generation failed."
        )


# ============================================================
# STARTUP
# ============================================================

@app.on_event("startup")
async def startup_event():

    logger.info("=" * 60)
    logger.info("WANING 1.5 API READY")
    logger.info("Parameters: %s", f"{parameter_count:,}")
    logger.info("Device: %s", DEVICE)
    logger.info("Maximum generation: %s", MAX_NEW_TOKENS)
    logger.info("Model directory: %s", MODEL_DIR)
    logger.info("=" * 60)
