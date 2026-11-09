import sys
import torch
import torch.nn.functional as f
from model import Transformer
from data import train_data, val_data, get_batch
from growth import grow

# Run from the repo folder:
#   python src/train.py small          small model the whole time
#   python src/train.py large          large model the whole time
#   python src/train.py grow 1000      small model, grown into the large one at step 1000

steps = 2000
mode = sys.argv[1]
if mode not in ["small", "large", "grow"]:
    raise ValueError("mode must be small, large, or grow")
if mode == 'grow':
    switch_at = int(sys.argv[2])
else:
    switch_at = -1

def small_model():
    return Transformer(vocab_size=50257, n_embd=64, block_size=64, n_heads=2, n_layers=2)

def large_model():
    return Transformer(vocab_size=50257, n_embd=128, block_size=64, n_heads=4, n_layers=4)

def get_loss(model, x, y):
    logits = model(x)
    return f.cross_entropy(logits.view(-1, logits.shape[-1]), y.view(-1))

def val_loss(model):
    model.eval()
    with torch.no_grad():
        total = sum(get_loss(model, x, y).item() for x, y in val_batches)
    model.train()
    return total / len(val_batches)

# The same 20 validation batches for every run, so runs can be compared.
torch.manual_seed(0)
val_batches = []
for i in range(20):
    batch = get_batch(val_data)
    val_batches.append(batch)

if mode == "large" :
    model = large_model()
else:
    model = small_model()

optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)

# Cost: training one token costs about 6 FLOPs per parameter (2 forward, 4 backward).
tokens_per_step = 16 * 64
flops = 0

name = mode if mode != "grow" else f"grow{switch_at}"
out = open(f"results_{name}.csv", "w")
out.write("step,tflops,val_loss\n")

for step in range(steps + 1):
    if step == switch_at:
        before_loss = val_loss(model)
        print(f"step {step}: growing from small to large model, val loss before growth {before_loss:.3f}")
        model = grow(model, large_model())
        optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)
        print(f"step {step}: grew into the large model")

    if step % 100 == 0:
        v = val_loss(model)
        print(f"step {step}  val loss {v:.3f}  cost {flops / 1e12:.2f} TFLOPs")
        out.write(f"{step},{flops / 1e12:.4f},{v:.4f}\n")
    if step == steps:
        break

    x, y = get_batch(train_data)
    loss = get_loss(model, x, y)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    params = sum(p.numel() for p in model.parameters())
    flops += 6 * params * tokens_per_step

out.close()
