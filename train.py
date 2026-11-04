import os
import math
import time
import torch
import torch.nn as nn
from torch.nn import functional as F
import tiktoken

# ======================= Luna 1.4 Hyperparameters =======================
batch_size = 16
block_size = 128
max_iters = 1500          # TOTAL steps for the whole run. Raise this to keep training an existing run.
warmup_iters = 100
eval_interval = 100
eval_iters = 20
learning_rate = 6e-4
min_lr = 6e-5
weight_decay = 0.1
grad_clip = 1.0
device = 'cpu'

n_embd = 128
n_head = 4                # 128 / 4 = 32 dim per head
n_layer = 4
dropout = 0.1

# BIGGEST SPEED LEVER: the GPT-2 vocab has 50,257 tokens, and the output layer
# (128 x 50,257) is ~85% of your compute. If True, only the tokens that actually
# appear in input.txt are kept (usually a few thousand), which makes every step much
# faster and the model smaller. BUT your Render backend must then map ids back using
# luna_1.4_vocab_map.pt before decoding. Leave False until you update the backend.
COMPACT_VOCAB = False

weights_path = 'luna_1.4_weights.pt'    # model weights only (best val loss) -> upload THIS to Render
ckpt_path = 'luna_1.4_ckpt.pt'          # full resume state (model + optimizer + step) -> keep local
vocab_map_path = 'luna_1.4_vocab_map.pt'
token_cache_path = 'input_tokens.pt'    # tokenized dataset cache, so you don't re-tokenize every run

torch.manual_seed(1337)
# More threads is not always faster on small workloads. Starting at half the logical
# cores (~physical cores). Try a few values (e.g. 2, 4, 6) and keep the fastest.
torch.set_num_threads(max(1, (os.cpu_count() or 2) // 2))

# ======================= Tokenizer =======================
tokenizer = tiktoken.get_encoding("gpt2")
encode = lambda s: tokenizer.encode(s, allowed_special={"<|endoftext|>"})

# ======================= Dataset =======================
if not os.path.exists('input.txt'):
    raise FileNotFoundError("input.txt not found! Run generate_data.py first.")

if os.path.exists(token_cache_path) and os.path.getmtime(token_cache_path) >= os.path.getmtime('input.txt'):
    data = torch.load(token_cache_path)
    print("Loaded cached tokens.")
else:
    print("Tokenizing dataset with tiktoken (BPE)...")
    with open('input.txt', 'r', encoding='utf-8') as f:
        text = f.read()
    data = torch.tensor(encode(text), dtype=torch.long)
    torch.save(data, token_cache_path)
print(f"Dataset: {len(data):,} total subword tokens.")

used_tokens = None
if COMPACT_VOCAB:
    used_tokens = torch.unique(data)
    remap = torch.full((tokenizer.n_vocab,), -1, dtype=torch.long)
    remap[used_tokens] = torch.arange(len(used_tokens))
    data = remap[data]
    vocab_size = len(used_tokens)
    torch.save(used_tokens, vocab_map_path)
    print(f"Compact vocab: {vocab_size:,} tokens (down from {tokenizer.n_vocab:,}). Saved {vocab_map_path}")
else:
    vocab_size = tokenizer.n_vocab


def decode_ids(ids):
    """ids: list[int] in model space -> text."""
    if used_tokens is not None:
        ids = used_tokens[torch.tensor(ids)].tolist()
    return tokenizer.decode(ids)


n = int(0.9 * len(data))
train_data = data[:n]
val_data = data[n:]
assert len(val_data) > block_size + 1, "input.txt is too small: need much more text than block_size."


def get_batch(split):
    d = train_data if split == 'train' else val_data
    ix = torch.randint(len(d) - block_size, (batch_size,))
    x = torch.stack([d[i:i + block_size] for i in ix])
    y = torch.stack([d[i + 1:i + block_size + 1] for i in ix])
    return x.to(device), y.to(device)


@torch.no_grad()
def estimate_loss(model):
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(split)
            _, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean().item()
    model.train()
    return out


# ======================= Model =======================
class CausalSelfAttention(nn.Module):
    """All heads in one batched op + fused attention kernel (replaces the per-head Head modules)."""

    def __init__(self):
        super().__init__()
        self.qkv = nn.Linear(n_embd, 3 * n_embd, bias=False)
        self.proj = nn.Linear(n_embd, n_embd)
        self.resid_dropout = nn.Dropout(dropout)

    def forward(self, x):
        B, T, C = x.shape
        q, k, v = self.qkv(x).split(n_embd, dim=2)
        q = q.view(B, T, n_head, C // n_head).transpose(1, 2)
        k = k.view(B, T, n_head, C // n_head).transpose(1, 2)
        v = v.view(B, T, n_head, C // n_head).transpose(1, 2)
        y = F.scaled_dot_product_attention(
            q, k, v, attn_mask=None,
            dropout_p=dropout if self.training else 0.0,
            is_causal=True,
        )
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        return self.resid_dropout(self.proj(y))


class FeedForward(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.GELU(),
            nn.Linear(4 * n_embd, n_embd),
            nn.Dropout(dropout),
        )

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
        # Weight tying: output layer shares the embedding matrix (saves ~6.4M params at full vocab)
        self.lm_head.weight = self.token_embedding_table.weight

        self.apply(self._init_weights)
        for name, p in self.named_parameters():
            if name.endswith('proj.weight') or name.endswith('ffwd.net.2.weight'):
                nn.init.normal_(p, mean=0.0, std=0.02 / math.sqrt(2 * n_layer))

    @staticmethod
    def _init_weights(m):
        if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, mean=0.0, std=0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Embedding):
            nn.init.normal_(m.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets=None):
        B, T = idx.shape
        pos = torch.arange(T, device=idx.device)
        x = self.token_embedding_table(idx) + self.position_embedding_table(pos)
        x = self.ln_f(self.blocks(x))
        if targets is None:
            # inference: only the last position is needed -> skips the huge output matmul for the rest
            return self.lm_head(x[:, [-1], :]), None
        logits = self.lm_head(x)
        loss = F.cross_entropy(logits.view(B * T, -1), targets.view(B * T))
        return logits, loss


@torch.no_grad()
def generate(model, idx, max_new_tokens, temperature=0.8, top_k=40):
    model.eval()
    for _ in range(max_new_tokens):
        logits, _ = model(idx[:, -block_size:])
        logits = logits[:, -1, :] / temperature
        if top_k:
            v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
            logits[logits < v[:, [-1]]] = -float('inf')
        probs = F.softmax(logits, dim=-1)
        idx = torch.cat((idx, torch.multinomial(probs, 1)), dim=1)
    model.train()
    return idx


def atomic_save(obj, path):
    """Write to a temp file then swap, so Ctrl+C mid-save can't corrupt your checkpoint."""
    tmp = path + '.tmp'
    torch.save(obj, tmp)
    os.replace(tmp, path)


def get_lr(it):
    if it <= warmup_iters:
        return learning_rate * it / warmup_iters
    if it >= max_iters:
        return min_lr
    ratio = (it - warmup_iters) / (max_iters - warmup_iters)
    return min_lr + 0.5 * (1.0 + math.cos(math.pi * ratio)) * (learning_rate - min_lr)


# ======================= Build / Resume =======================
model = LunaGPT().to(device)
total_params = sum(p.numel() for p in model.parameters()) / 1e6
print(f"Ore AI: Luna 1.4 parameter count: {total_params:.2f}M (vocab {vocab_size:,})")

decay_params = [p for p in model.parameters() if p.dim() >= 2]
nodecay_params = [p for p in model.parameters() if p.dim() < 2]
optimizer = torch.optim.AdamW(
    [{'params': decay_params, 'weight_decay': weight_decay},
     {'params': nodecay_params, 'weight_decay': 0.0}],
    lr=learning_rate, betas=(0.9, 0.95),
)

start_iter = 0
best_val = float('inf')
if os.path.exists(ckpt_path):
    try:
        ckpt = torch.load(ckpt_path, map_location=device)
        model.load_state_dict(ckpt['model'])
        optimizer.load_state_dict(ckpt['optimizer'])
        start_iter = ckpt['iter']
        best_val = ckpt['best_val']
        print(f"Resumed from {ckpt_path} at step {start_iter} (best val {best_val:.4f})")
    except (RuntimeError, KeyError) as e:
        raise SystemExit(
            f"Could not resume from {ckpt_path}: {e}\n"
            f"The model shape probably changed (vocab / layers / embd). Delete {ckpt_path} to start fresh."
        )
elif os.path.exists(weights_path):
    # Old-architecture weights from earlier runs can't load into this model. Don't overwrite them.
    os.replace(weights_path, weights_path + '.old')
    print(f"Fresh run: backed up old {weights_path} -> {weights_path}.old")

tokens_per_iter = batch_size * block_size
print(f"Plan: {max_iters} steps x {tokens_per_iter:,} tokens = "
      f"~{max_iters * tokens_per_iter / len(train_data):.1f} passes over the training data")

# ======================= Training Loop =======================
t_start = time.time()
for it in range(start_iter + 1, max_iters + 1):
    lr = get_lr(it)
    for g in optimizer.param_groups:
        g['lr'] = lr

    xb, yb = get_batch('train')
    _, loss = model(xb, yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
    optimizer.step()

    if it % eval_interval == 0 or it == 1 or it == max_iters:
        losses = estimate_loss(model)
        elapsed = time.time() - t_start
        done = it - start_iter
        eta_min = elapsed / done * (max_iters - it) / 60
        print(f"Step {it}/{max_iters} | Train {losses['train']:.4f} | Val {losses['val']:.4f} "
              f"| lr {lr:.2e} | {elapsed / done:.2f}s/step | ETA {eta_min:.1f} min")

        if losses['val'] < best_val:
            best_val = losses['val']
            atomic_save(model.state_dict(), weights_path)
            print(f"--> New best val loss, weights saved to {weights_path}")

        atomic_save({'model': model.state_dict(), 'optimizer': optimizer.state_dict(),
                     'iter': it, 'best_val': best_val}, ckpt_path)

if start_iter >= max_iters:
    print(f"Already trained to step {start_iter}. Raise max_iters to train further.")
else:
    print("Training finished successfully!")

# ======================= Quick sample from the best weights =======================
if os.path.exists(weights_path):
    model.load_state_dict(torch.load(weights_path, map_location=device))
    prompt = train_data[:8].unsqueeze(0).to(device)
    out = generate(model, prompt, max_new_tokens=60)
    print("\n--- Sample ---")
    print(decode_ids(out[0].tolist()))