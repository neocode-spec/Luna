
import os
import torch

from transformers import AutoTokenizer, AutoModelForCausalLM
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# =====================================================
# WANING 1.5 CONFIGURATION
# =====================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = BASE_DIR

MAX_NEW_TOKENS = 40
DEVICE = "cpu"

torch.set_num_threads(1)

print("Loading WANING 1.5...")

# =====================================================
# LOAD TOKENIZER
# =====================================================

tokenizer = AutoTokenizer.from_pretrained(
    MODEL_DIR,
    local_files_only=True
)

# =====================================================
# LOAD MODEL
# =====================================================

model = AutoModelForCausalLM.from_pretrained(
    MODEL_DIR,
    local_files_only=True,
    use_safetensors=True
)

model.to(DEVICE)
model.eval()

print("WANING 1.5 loaded successfully.")
print("Parameters:", sum(p.numel() for p in model.parameters()))
print("Vocabulary:", model.config.vocab_size)

# =====================================================
# FASTAPI
# =====================================================

app = FastAPI(
    title="WANING 1.5 API",
    version="1.5.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"]
)


class PromptRequest(BaseModel):
    prompt: str = ""
    message: str = ""
    text: str = ""
    input: str = ""
    max_tokens: int = 40
    language: str = "english"
    model: str = "waning-1.5"
    thinking_level: str = "medium"


# =====================================================
# HEALTH CHECK
# =====================================================

@app.get("/")
def root():
    return {
        "status": "WANING 1.5 backend is live",
        "model": "WANING 1.5",
        "parameters": sum(p.numel() for p in model.parameters())
    }


@app.get("/generate")
@app.get("/generate/")
def generate_get_fallback():
    return {
        "status": "WANING 1.5 is ready. Send a POST request."
    }


# =====================================================
# TEXT GENERATION
# =====================================================

@app.post("/generate")
@app.post("/generate/")
def generate_text(req: PromptRequest):

    user_prompt = (
        req.prompt or
        req.message or
        req.text or
        req.input
    ).strip()

    if not user_prompt:
        raise HTTPException(
            status_code=400,
            detail="A prompt is required."
        )

    tokens_to_generate = max(
        1,
        min(req.max_tokens, MAX_NEW_TOKENS)
    )

    # Match the Assistant-style format observed
    # during the WANING 1.5 generation test.
    formatted_prompt = (
        f"User: {user_prompt}\n"
        "Assistant:"
    )

    inputs = tokenizer(
        formatted_prompt,
        return_tensors="pt"
    )

    inputs = {
        key: value.to(DEVICE)
        for key, value in inputs.items()
    }

    try:
        with torch.inference_mode():
            output = model.generate(
                **inputs,
                max_new_tokens=tokens_to_generate,
                do_sample=True,
                temperature=0.7,
                top_p=0.9,
                repetition_penalty=1.1,
                pad_token_id=tokenizer.eos_token_id,
                eos_token_id=tokenizer.eos_token_id
            )

        # Decode only newly generated tokens.
        generated_tokens = output[0][inputs["input_ids"].shape[1]:]

        response = tokenizer.decode(
            generated_tokens,
            skip_special_tokens=True
        ).strip()

        # Stop if the model starts another conversation turn.
        for marker in ["\nUser:", "\nHuman:"]:
            if marker in response:
                response = response.split(marker)[0].strip()

        return {
            "response": response
        }

    except Exception as error:
        raise HTTPException(
            status_code=500,
            detail=f"Generation failed: {str(error)}"
        )
