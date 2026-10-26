"""The default model (a GPT-style Transformer) and the sizes you can ask for by name.

    python train.py --model small          # built-in size
    python train.py --model my_model.py    # your own: see build() below

To test a different architecture, make a .py file with a `build(vocab_size, block_size)` function that
returns a torch module. Its forward must be  forward(idx, targets=None) -> (logits, loss)
where idx/targets are (batch, seq) token ids, logits are (batch, seq, vocab_size) and loss is the
mean cross-entropy.  (For example: copy GPT below and change it.)
"""
import importlib.util
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

# name: (layers, heads, width).  Each step up roughly doubles the compute per token.
SIZES = {
    "tiny": (2, 2, 64),
    "mini": (4, 4, 128),
    "small": (4, 4, 256),
    "medium": (6, 6, 384),
    "large": (8, 8, 512),
}


class Block(nn.Module):
    def __init__(self, width, heads):
        super().__init__()
        self.heads = heads
        self.ln_1, self.ln_2 = nn.LayerNorm(width), nn.LayerNorm(width)
        self.qkv, self.proj = nn.Linear(width, 3 * width), nn.Linear(width, width)
        self.fc, self.out = nn.Linear(width, 4 * width), nn.Linear(4 * width, width)

    def forward(self, x):
        B, T, C = x.shape
        q, k, v = (t.view(B, T, self.heads, -1).transpose(1, 2) for t in self.qkv(self.ln_1(x)).split(C, dim=2))
        x = x + self.proj(F.scaled_dot_product_attention(q, k, v, is_causal=True).transpose(1, 2).reshape(B, T, C))
        return x + self.out(F.gelu(self.fc(self.ln_2(x))))


class GPT(nn.Module):
    def __init__(self, vocab_size, block_size, layers, heads, width):
        super().__init__()
        self.vocab_size = vocab_size
        padded = math.ceil(vocab_size / 64) * 64        # round up: faster matmuls
        self.wte, self.wpe = nn.Embedding(padded, width), nn.Embedding(block_size, width)
        self.blocks = nn.ModuleList(Block(width, heads) for _ in range(layers))
        self.ln_f = nn.LayerNorm(width)
        self.head = nn.Linear(width, padded, bias=False)
        self.head.weight = self.wte.weight              # input/output embeddings are shared
        for m in self.modules():
            if isinstance(m, (nn.Linear, nn.Embedding)):
                nn.init.normal_(m.weight, 0.0, 0.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.zeros_(m.bias)
        for b in self.blocks:                           # GPT-2 trick: small residual outputs
            for lin in (b.proj, b.out):
                nn.init.normal_(lin.weight, 0.0, 0.02 / math.sqrt(2 * layers))

    def forward(self, idx, targets=None):
        x = self.wte(idx) + self.wpe(torch.arange(idx.size(1), device=idx.device))
        for b in self.blocks:
            x = b(x)
        logits = self.head(self.ln_f(x))[..., :self.vocab_size]
        loss = None if targets is None else F.cross_entropy(logits.float().reshape(-1, self.vocab_size), targets.reshape(-1))
        return logits, loss


def build(spec, vocab_size, block_size):
    """spec is a size name from SIZES, or the path to a .py file defining build(vocab_size, block_size)."""
    if spec in SIZES:
        return GPT(vocab_size, block_size, *SIZES[spec])
    if spec.endswith(".py"):
        s = importlib.util.spec_from_file_location("user_model", spec)
        mod = importlib.util.module_from_spec(s)
        s.loader.exec_module(mod)
        return mod.build(vocab_size, block_size)
    raise SystemExit(f"Unknown model '{spec}'. Use one of {list(SIZES)} or a path to a .py file.")
