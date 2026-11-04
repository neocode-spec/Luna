"""
luna_model.py  -  the Luna 1.4 model, shared by train.py and app.py

Keeping the model in ONE file means training and the server can never
disagree about the architecture. If you change a number below, delete the
old weights and train again.
"""

import json

import tiktoken
import torch
import torch.nn as nn
from torch.nn import functional as F

# ---------------------------------------------------------------
# Model size  (about 5M parameters with the compact vocabulary)
# ---------------------------------------------------------------

block_size = 128     # how many tokens the model can look back at
n_embd = 256
n_head = 8
n_layer = 6
dropout = 0.1

END_TOKEN = "<|endoftext|>"
VOCAB_FILE = "luna_vocab.json"
WEIGHTS_FILE = "luna_1.4_weights.pt"

# ---------------------------------------------------------------
# Compact tokenizer
#
# GPT-2 has 50,257 tokens, but your chat data only uses about a thousand
# of them. Keeping just those makes the model smaller, faster to train and
# much faster to answer on Render. Any word the model has not seen is
# split into single bytes, so nothing a user types can break it.
# ---------------------------------------------------------------

class LunaTokenizer:

    def __init__(self, kept_ids):
        self.enc = tiktoken.get_encoding("gpt2")
        self.kept_ids = list(kept_ids)
        self.to_small = {t: i for i, t in enumerate(self.kept_ids)}
        self.byte_tokens = [self.enc.encode_single_token(bytes([b])) for b in range(256)]
        self.end_id = self.to_small[self.enc.eot_token]
        self.vocab_size = len(self.kept_ids)

    @classmethod
    def build(cls, text):
        enc = tiktoken.get_encoding("gpt2")
        used = set(enc.encode(text, allowed_special={END_TOKEN}))
        used |= {enc.encode_single_token(bytes([b])) for b in range(256)}
        used.add(enc.eot_token)
        return cls(sorted(used))

    def encode(self, text):
        out = []
        for t in self.enc.encode(text, allowed_special={END_TOKEN}):
            if t in self.to_small:
                out.append(self.to_small[t])
            else:
                for b in self.enc.decode_single_token_bytes(t):
                    out.append(self.to_small[self.byte_tokens[b]])
        return out

    def decode(self, ids):
        return self.enc.decode([self.kept_ids[i] for i in ids])

    def save(self, path=VOCAB_FILE):
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.kept_ids, f)

    @classmethod
    def load(cls, path=VOCAB_FILE):
        with open(path, "r", encoding="utf-8") as f:
            return cls(json.load(f))

# ---------------------------------------------------------------
# Transformer
# ---------------------------------------------------------------

class CausalSelfAttention(nn.Module):

    def __init__(self):
        super().__init__()
        self.qkv = nn.Linear(n_embd, 3 * n_embd, bias=False)
        self.proj = nn.Linear(n_embd, n_embd)
        self.attn_drop = nn.Dropout(dropout)
        self.resid_drop = nn.Dropout(dropout)
        self.register_buffer(
            "mask",
            torch.tril(torch.ones(block_size, block_size)).view(1, 1, block_size, block_size),
        )

    def forward(self, x):
        B, T, C = x.shape
        q, k, v = self.qkv(x).split(n_embd, dim=2)
        q = q.view(B, T, n_head, C // n_head).transpose(1, 2)
        k = k.view(B, T, n_head, C // n_head).transpose(1, 2)
        v = v.view(B, T, n_head, C // n_head).transpose(1, 2)

        att = (q @ k.transpose(-2, -1)) * (k.size(-1) ** -0.5)
        att = att.masked_fill(self.mask[:, :, :T, :T] == 0, float("-inf"))
        att = self.attn_drop(F.softmax(att, dim=-1))

        y = (att @ v).transpose(1, 2).contiguous().view(B, T, C)
        return self.resid_drop(self.proj(y))


class Block(nn.Module):

    def __init__(self):
        super().__init__()
        self.ln1 = nn.LayerNorm(n_embd)
        self.attn = CausalSelfAttention()
        self.ln2 = nn.LayerNorm(n_embd)
        self.ffwd = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.GELU(),
            nn.Linear(4 * n_embd, n_embd),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        return x + self.ffwd(self.ln2(x))


class LunaGPT(nn.Module):

    def __init__(self, vocab_size):
        super().__init__()
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
        self.position_embedding_table = nn.Embedding(block_size, n_embd)
        self.blocks = nn.Sequential(*[Block() for _ in range(n_layer)])
        self.ln_f = nn.LayerNorm(n_embd)
        self.lm_head = nn.Linear(n_embd, vocab_size)
        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets=None, last_only=False):
        B, T = idx.shape
        pos = torch.arange(T, device=idx.device)
        x = self.token_embedding_table(idx) + self.position_embedding_table(pos)
        x = self.ln_f(self.blocks(x))

        if targets is None:
            # when answering we only need the last position, which saves a lot of work
            if last_only:
                x = x[:, -1:, :]
            return self.lm_head(x), None

        logits = self.lm_head(x)
        loss = F.cross_entropy(logits.view(B * T, -1), targets.view(B * T))
        return logits, loss

    @torch.no_grad()
    def generate(self, idx, max_new_tokens, stop_ids=(), temperature=0.7, top_k=40):
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -block_size:]
            logits, _ = self(idx_cond, last_only=True)
            logits = logits[:, -1, :] / max(temperature, 1e-5)

            if top_k:
                values, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < values[:, [-1]]] = -float("inf")

            probs = F.softmax(logits, dim=-1)
            next_id = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, next_id), dim=1)

            if next_id.item() in stop_ids:
                break

        return idx
