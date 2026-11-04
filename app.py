import os
import torch
import torch.nn as nn
from torch.nn import functional as F
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import tiktoken

# --- Luna 1.4 backend. These settings MUST match train.py exactly. ---
block_size = 128
device = 'cpu'
n_embd = 128
n_head = 4
n_layer = 4
weights_path = 'luna_1.4_weights.pt'
vocab_map_path = 'luna_1.4_vocab_map.pt'   # only exists if you trained with COMPACT_VOCAB = True
MAX_NEW_TOKENS = 30                        # cap per request (Render free CPU is tiny); raise if responses feel short

# Render free gives ~0.1 CPU: extra threads only add overhead.
torch.set_num_threads(1)

tokenizer = tiktoken.get_encoding("gpt2")

# --- Load weights FIRST, so the vocab size always comes from the file itself ---
state_dict = None
if os.path.exists(weights_path):
    state_dict = torch.load(weights_path, map_location=device)
    vocab_size = state_dict['token_embedding_table.weight'].shape[0]
else:
    vocab_size = tokenizer.n_vocab
    print(f"WARNING: {weights_path} not found. Serving an UNTRAINED model (random output).")

# Compact vocab support: map GPT-2 token ids <-> the smaller ids the model was trained with
used_tokens = None
remap = None
if vocab_size != tokenizer.n_vocab:
    if not os.path.exists(vocab_map_path):
        raise RuntimeError(
            f"Weights have vocab {vocab_size} but {vocab_map_path} is missing. "
            f"Upload luna_1.4_vocab_map.pt next to app.py."
        )
    used_tokens = torch.load(vocab_map_path, map_location='cpu')
    if len(used_tokens) != vocab_size:
        raise RuntimeError(
            f"{vocab_map_path} has {len(used_tokens)} tokens but the weights expect {vocab_size}. "
            f"Upload the map and weights from the same training run."
        )
    remap = {int(t): i for i, t in enumerate(used_tokens.tolist())}

EOT_GPT2 = 50256
eos_id = EOT_GPT2 if remap is None else remap.get(EOT_GPT2)


def encode(s):
    ids = tokenizer.encode(s, allowed_special={"<|endoftext|>"})
    if remap is not None:
        ids = [remap[i] for i in ids if i in remap]   # tokens never seen in training are dropped
    return ids or [0]


def decode(ids):
    if used_tokens is not None:
        ids = used_tokens[torch.tensor(ids)].tolist()
    return tokenizer.decode(ids)


# --- Model (identical to train.py, minus dropout which is off at inference) ---
class CausalSelfAttention(nn.Module):
    def __init__(self):
        super().__init__()
        self.qkv = nn.Linear(n_embd, 3 * n_embd, bias=False)
        self.proj = nn.Linear(n_embd, n_embd)

    def forward(self, x):
        B, T, C = x.shape
        q, k, v = self.qkv(x).split(n_embd, dim=2)
        q = q.view(B, T, n_head, C // n_head).transpose(1, 2)
        k = k.view(B, T, n_head, C // n_head).transpose(1, 2)
        v = v.view(B, T, n_head, C // n_head).transpose(1, 2)
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        return self.proj(y.transpose(1, 2).contiguous().view(B, T, C))


class FeedForward(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(n_embd, 4 * n_embd), nn.GELU(), nn.Linear(4 * n_embd, n_embd))

    def forward(self, x):
        return self.net(x)


class Block(nn.Module):
    def __init__(self):
        super().__init__()
        self.sa = CausalSelfAttention()
        self.ffwd = FeedForward()
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)

    def forward(self, x):
        # Same as training. (Your old backend had x + ffwd(ln2(x + sa(...))), which drops a residual
        # and does NOT match how the model was trained.)
        x = x + self.sa(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))
        return x


class LunaGPT(nn.Module):
    def __init__(self):
        super().__init__()
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
        self.position_embedding_table = nn.Embedding(block_size, n_embd)
        self.blocks = nn.Sequential(*[Block() for _ in range(n_layer)])
        self.ln_f = nn.LayerNorm(n_embd)
        self.lm_head = nn.Linear(n_embd, vocab_size, bias=False)
        self.lm_head.weight = self.token_embedding_table.weight   # tied, like training

    def forward(self, idx):
        B, T = idx.shape
        x = self.token_embedding_table(idx) + self.position_embedding_table(torch.arange(T, device=idx.device))
        x = self.ln_f(self.blocks(x))
        return self.lm_head(x[:, [-1], :])   # only the last position is needed

    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=0.7, top_k=40):
        for _ in range(max_new_tokens):
            logits = self(idx[:, -block_size:])[:, -1, :] / max(temperature, 1e-3)
            if top_k:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float('inf')
            nxt = torch.multinomial(F.softmax(logits, dim=-1), num_samples=1)
            idx = torch.cat((idx, nxt), dim=1)
            if eos_id is not None and nxt.item() == eos_id:
                break
        return idx


model = LunaGPT().to(device)
if state_dict is not None:
    try:
        model.load_state_dict(state_dict)
    except RuntimeError as e:
        raise RuntimeError(
            "Weights don't match this architecture. Upload the weights produced by the NEW train.py "
            f"(old Luna 1.3/1.4 weights will not load). Details: {e}"
        )
model.eval()

# --- API ---
app = FastAPI()
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"])


class PromptRequest(BaseModel):
    prompt: str = ""
    message: str = ""
    text: str = ""
    input: str = ""
    max_tokens: int = 50
    language: str = "english"
    model: str = "luna-1.4"
    thinking_level: str = "medium"


@app.get("/")
def root():
    return {"status": "Optimized Luna 1.4 backend is live!"}


@app.get("/generate")
@app.get("/generate/")
def generate_get_fallback():
    return {"status": "Endpoint is live. Send a POST request to generate text."}


@app.post("/generate")
@app.post("/generate/")
def generate_text(req: PromptRequest):
    tokens_to_generate = max(1, min(req.max_tokens, MAX_NEW_TOKENS))
    user_prompt = req.prompt or req.message or req.text or req.input or "Hello"
    formatted_prompt = f"User: {user_prompt.strip()}\nOre:"
    context = torch.tensor([encode(formatted_prompt)], dtype=torch.long, device=device)

    with torch.inference_mode():
        out = model.generate(context, max_new_tokens=tokens_to_generate, temperature=0.7)[0].tolist()

    full_output = decode(out)
    response = full_output.split("Ore:")[-1].strip() if "Ore:" in full_output else full_output[len(formatted_prompt):].strip()
    if "\nUser:" in response:
        response = response.split("\nUser:")[0].strip()
    return {"response": response.replace("<|endoftext|>", "").strip()}