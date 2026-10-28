"""Train a model and record its loss against compute cost (TFLOPs).

    python train.py                                                      # model.py, 2000 steps
    python train.py --model small.py                                     # any model file (see model.py)
    python train.py --model small.py --then large.py --switch-at 1000    # grow small -> large at step 1000 (see grow.py)
    python train.py --steps 200                                  # quick test

Results land in runs/<name>/. Compare runs with:  python plot.py
"""
import argparse
import csv
import importlib.util
import json
import logging
import math
import time
from pathlib import Path

import torch
from torch.nn.attention import SDPBackend, sdpa_kernel

logging.getLogger("torch.utils.flop_counter").setLevel(logging.ERROR)  # harmless "triton not found" notice
from torch.utils.flop_counter import FlopCounterMode  # noqa: E402

from data import VOCAB_SIZE, Data  # noqa: E402
from grow import grow  # noqa: E402

# Same for every run so results are comparable. Change here if you must.
SEQ_LEN, BATCH, ACCUM = 256, 16, 2        # 8,192 tokens per step
LR, MIN_LR, WARMUP, REWARM = 6e-4, 6e-5, 200, 100   # REWARM: LR warm-up steps after a switch
EVAL_EVERY, EVAL_BATCHES = 100, 40

ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
ap.add_argument("--model", default="model.py", help="path to a .py file defining build(vocab_size, block_size)")
ap.add_argument("--then", help="grow into this model file at --switch-at")
ap.add_argument("--switch-at", type=int, help="step at which to grow (needs --then)")
ap.add_argument("--steps", type=int, default=2000)
ap.add_argument("--name", help="run name (default: made from the options)")
args = ap.parse_args()
if bool(args.then) != (args.switch_at is not None):
    ap.error("--then and --switch-at go together")

name = args.name or (f"{Path(args.model).stem}" + (f"-to-{Path(args.then).stem}@{args.switch_at}" if args.then else "")
                     + f"_{args.steps}")
run_dir = Path(__file__).parent / "runs" / name
run_dir.mkdir(parents=True, exist_ok=True)

device = "cuda" if torch.cuda.is_available() else "cpu"
amp = torch.autocast(device, dtype=torch.bfloat16, enabled=device == "cuda")
torch.manual_seed(1337)
gen = torch.Generator().manual_seed(1337)
data = Data()


def build(path, vocab_size, block_size):
    spec = importlib.util.spec_from_file_location(Path(path).stem, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.build(vocab_size, block_size)


def flops_per_token(model):
    """Measured (not estimated): FLOPs of one forward+backward pass, per token. Works for any model."""
    x = torch.randint(0, VOCAB_SIZE, (1, SEQ_LEN), device=device)
    with sdpa_kernel(SDPBackend.MATH), FlopCounterMode(display=False) as fc:
        model(x, x)[1].backward()
    model.zero_grad(set_to_none=True)
    return fc.get_total_flops() / SEQ_LEN


def make_optimizer(model):
    # (weight decay only on matrices, not on biases/LayerNorm)
    mats = [p for p in model.parameters() if p.dim() >= 2]
    rest = [p for p in model.parameters() if p.dim() < 2]
    return torch.optim.AdamW([{"params": mats, "weight_decay": 0.1}, {"params": rest, "weight_decay": 0.0}],
                             lr=LR, betas=(0.9, 0.95), fused=device == "cuda")


@torch.no_grad()
def evaluate(model, split):
    model.eval()
    tot = n = 0
    for x, y in data.eval_batches(split, BATCH, SEQ_LEN, device, EVAL_BATCHES):
        with amp:
            tot += model(x, y)[1].item() * x.numel()
        n += x.numel()
    model.train()
    return tot / n


def lr_at(step, last_switch):
    if step < WARMUP:
        lr = LR * (step + 1) / WARMUP
    else:
        t = min(1.0, (step - WARMUP) / max(1, args.steps - WARMUP))
        lr = MIN_LR + 0.5 * (1 + math.cos(math.pi * t)) * (LR - MIN_LR)
    if last_switch is not None:
        lr *= min(1.0, (step - last_switch + 1) / REWARM)
    return lr


model = build(args.model, VOCAB_SIZE, SEQ_LEN).to(device)
opt, fpt = make_optimizer(model), flops_per_token(model)
tokens = tflops = 0.0
last_switch, recent, t0 = None, [], time.time()
log = csv.writer(open(run_dir / "log.csv", "w", newline=""))
log.writerow(["step", "tokens", "tflops", "train_loss", "val_loss", "note"])
print(f"{name}: {sum(p.numel() for p in model.parameters()) / 1e6:.1f}M params, {fpt / 1e6:.0f} MFLOPs/token, {device}")

for step in range(args.steps + 1):
    if step % EVAL_EVERY == 0 or step == args.steps:
        v = evaluate(model, "val")
        t = sum(recent) / len(recent) if recent else float("nan")
        recent.clear()
        log.writerow([step, int(tokens), round(tflops, 4), round(t, 4), round(v, 4), ""])
        print(f"step {step:5d}  val {v:.3f}  train {t:.3f}  {tflops:8.1f} TFLOPs  {time.time() - t0:5.0f}s")
    if step == args.steps:
        break

    if args.then and step == args.switch_at:
        before = evaluate(model, "val")
        model = grow(model, build(args.then, VOCAB_SIZE, SEQ_LEN).to(device)).to(device)
        opt, fpt, last_switch = make_optimizer(model), flops_per_token(model), step
        after = evaluate(model, "val")
        log.writerow([step, int(tokens), round(tflops, 4), "", round(after, 4), f"grown (val before: {before:.4f})"])
        print(f"step {step:5d}  GREW to {args.then}: val {before:.3f} -> {after:.3f}  ({fpt / 1e6:.0f} MFLOPs/token now)")

    for g in opt.param_groups:
        g["lr"] = lr_at(step, last_switch)
    loss_sum = 0.0
    for _ in range(ACCUM):
        x, y = data.train_batch(BATCH, SEQ_LEN, device, gen)
        with amp:
            loss = model(x, y)[1]
        (loss / ACCUM).backward()
        loss_sum += loss.item() / ACCUM
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    opt.step()
    opt.zero_grad(set_to_none=True)
    recent.append(loss_sum)
    tokens += BATCH * ACCUM * SEQ_LEN
    tflops += fpt * BATCH * ACCUM * SEQ_LEN / 1e12

summary = {"name": name, "steps": args.steps, "tflops": round(tflops, 3), "val_loss": round(v, 4),
           "test_loss": round(evaluate(model, "test"), 4), "seconds": round(time.time() - t0)}
(run_dir / "summary.json").write_text(json.dumps(summary, indent=2))
torch.save(model.state_dict(), run_dir / "model.pt")
print(summary)
