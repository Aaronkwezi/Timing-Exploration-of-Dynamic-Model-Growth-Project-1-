import torch
import torch.nn as nn
import torch.nn.functional as f
   
class SelfAttention(nn.Module):
    def __init__(self, n_embd, n_heads):
        super().__init__()
        self.n_heads = n_heads
        assert n_embd % n_heads == 0
        self.head_size = n_embd // n_heads
        self.qkv = nn.Linear(n_embd, 3*n_embd)
        self.proj = nn.Linear(n_embd,n_embd)

    def forward(self, x):
        B, T, C = x.shape
        q, k, v = self.qkv(x).split(C, dim=-1)
        q = q.view(B,T, self.n_heads, self.head_size)
        k = k.view(B,T, self.n_heads, self.head_size)
        v = v.view(B,T, self.n_heads, self.head_size)

        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        out = f.scaled_dot_product_attention(q,k,v, is_causal=True)
        out = out.transpose(1,2).contiguous()
        out = out.view(B,T, C)

        return self.proj(out)
class FeedFoward(nn.Module):
    def __init__(self, n_embd):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.GELU(),
            nn.Linear(4*n_embd,n_embd)
        )
    def forward(self,x):
        return self.net(x)
class TransformerBlock(nn.Module):
    def __init__(self, n_embd, n_heads):
        super().__init__()
        self.ln1 = nn.LayerNorm(n_embd)
        self.attention = SelfAttention(n_embd, n_heads)

        self.ln2 = nn.LayerNorm(n_embd)
        self.ffn = FeedFoward(n_embd)
    def forward(self, x):
        x = x + self.attention(self.ln1(x))
        x = x + self.ffn(self.ln2(x))

        return x
class Transformer(nn.Module):
    def __init__(self, vocab_size, n_embd,block_size,n_heads,n_layers):
        super().__init__()
        self.block_size = block_size
        self.token_embedding = nn.Embedding(vocab_size,n_embd)
        self.position_embedding = nn.Embedding(block_size,n_embd)

        self.blocks = nn.Sequential(
            *[
                TransformerBlock(n_embd,n_heads)
                for _ in range(n_layers)
            ]
        )
        self.ln_f = nn.LayerNorm(n_embd)
        self.lm_head = nn.Linear(n_embd, vocab_size)
    def forward(self, idx):
        B, T = idx.shape
        if T > self.block_size:
            raise ValueError("Sequence exceeds block size")
        
        tok_emb = self.token_embedding(idx)
        positons = torch.arange(T, device=idx.device)
        pos_emb = self.position_embedding(positons)
        x = tok_emb + pos_emb
        x = self.blocks(x)
        x = self.ln_f(x)
        logits = self.lm_head(x)
        return logits
  #testing of large and small  
if __name__ == "__main__":
    small_model = Transformer(
        vocab_size=50257,
        n_embd=64,
        block_size=64,
        n_heads=2,
        n_layers=2
    )
    large_model = Transformer(
            vocab_size=50257,
            n_embd=128,
            block_size=64,
            n_heads=4,
            n_layers=4
    )

    tokens = torch.randint(0,50257,(16,64))
    small_output = small_model(tokens)
    large_output = large_model(tokens)

    print("Small output:", small_output.shape)
    print("Large output:", large_output.shape)
    print("Small parameters:" , sum(p.numel() for p in small_model.parameters()))
    print("Large parameters:" , sum(p.numel() for p in large_model.parameters()))