import argparse
import json
import math
import time
from pathlib import Path

import torch
import torch.nn.functional as f

import data as D
from flops import train_flops_per_token, eval_flops_per_token
from growth import grow
from model import Transformer

# Run from anywhere:
#   python src/train.py small                      small model the whole time
#   python src/train.py large                      large model the whole time
#   python src/train.py grow 1000                  small model, grown into the large one at step 1000
# Useful options (all defaults reproduce the original setup):
#   --seed 3  --steps 20000  --log_every 25  --out_dir runs/main
#   --schedule cosine --warmup 200 --post_grow_warmup 200      learning-rate schedule
#   --small 64,2,2 --large 256,8,6                              width,heads,layers
# For many seeds / configs at once see src/sweep.py; to aggregate see src/analyze.py.

VOCAB = 50257


def dims(s):
    w, h, l = (int(v) for v in s.split(","))
    return w, h, l


def parse_args(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("mode", choices=["small", "large", "grow"])
    p.add_argument("switch_at", type=int, nargs="?", help="step at which to grow (mode=grow)")
    p.add_argument("--steps", type=int, default=2000)
    p.add_argument("--seed", type=int, default=0, help="model init + data order")
    p.add_argument("--small", type=dims, default=(64, 2, 2), help="width,heads,layers")
    p.add_argument("--large", type=dims, default=(128, 4, 4), help="width,heads,layers")
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--schedule", choices=["constant", "cosine"], default="constant")
    p.add_argument("--warmup", type=int, default=0, help="linear warm-up steps at the start of training")
    p.add_argument("--post_grow_warmup", type=int, default=0,
                   help="linear warm-up steps after growth (fresh optimizer), multiplies the schedule")
    p.add_argument("--min_lr_ratio", type=float, default=0.1, help="cosine floor as a fraction of --lr")
    p.add_argument("--log_every", type=int, default=100)
    p.add_argument("--val_batches", type=int, default=50,
                   help="fixed validation batches (32 x 64 tokens each) used for the curve; 0 = whole split")
    p.add_argument("--no_final_eval", action="store_true", help="skip full-validation and test loss at the end")
    p.add_argument("--out_dir", default=".")
    p.add_argument("--device", default="auto")
    p.add_argument("--threads", type=int, default=None)
    a = p.parse_args(argv)
    if a.mode == "grow":
        if a.switch_at is None or not 0 < a.switch_at < a.steps:
            p.error("grow needs a switch step with 0 < step < --steps")
    elif a.switch_at is not None:
        p.error("switch step only applies to mode grow")
    return a


def lr_at(a, step, last_reset):
    lr = a.lr
    if a.schedule == "cosine":
        progress = min(1.0, max(0, step - a.warmup) / max(1, a.steps - a.warmup))
        lr *= a.min_lr_ratio + (1 - a.min_lr_ratio) * 0.5 * (1 + math.cos(math.pi * progress))
    if step < a.warmup:
        lr *= (step + 1) / a.warmup
    if last_reset is not None and step - last_reset < a.post_grow_warmup:
        lr *= (step - last_reset + 1) / a.post_grow_warmup
    return lr


def main(argv=None):
    a = parse_args(argv)
    device = ("cuda" if torch.cuda.is_available() else "cpu") if a.device == "auto" else a.device
    if a.threads:
        torch.set_num_threads(a.threads)

    def build(d):
        return Transformer(vocab_size=VOCAB, n_embd=d[0], block_size=D.block_size,
                           n_heads=d[1], n_layers=d[2]).to(device)

    name = a.mode if a.mode != "grow" else f"grow{a.switch_at}"
    out_dir = Path(a.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = out_dir / f"results_{name}_s{a.seed}"

    torch.manual_seed(a.seed)                       # model init only; data has its own generator
    stream = D.BatchStream(D.train_data, a.seed, device)
    val_set = D.eval_batches(D.val_data, a.val_batches or None, device=device)

    def val_loss(model, batches=val_set):
        model.eval()
        tot, n = 0.0, 0
        with torch.no_grad():
            for x, y in batches:
                tot += f.cross_entropy(model(x).view(-1, VOCAB), y.view(-1), reduction="sum").item()
                n += y.numel()
        model.train()
        return tot / n, n

    model = build(a.large if a.mode == "large" else a.small)
    which = "large" if a.mode == "large" else "small"
    optimizer = torch.optim.AdamW(model.parameters(), lr=a.lr)
    per_token = train_flops_per_token(model, D.block_size)
    eval_per_token = eval_flops_per_token(model, D.block_size)

    tokens_per_step = D.batch_size * D.block_size
    cost = {k: 0.0 for k in per_token}              # cumulative training FLOPs per proxy
    eval_cost = 0.0                                 # cumulative evaluation FLOPs (not in `cost`)
    last_reset = None
    window_loss, window_n = 0.0, 0

    cols = ["step", "tflops", "tflops_matmul", "tflops_full", "tflops_eval",
            "train_loss", "val_loss", "model", "lr"]
    out = open(f"{stem}.csv", "w")
    out.write(",".join(cols) + "\n")

    def log(step, train_loss, lr):
        nonlocal eval_cost
        v, n = val_loss(model)
        eval_cost += eval_per_token * n
        row = [step, cost["6pbt"] / 1e12, cost["matmul"] / 1e12, cost["full"] / 1e12, eval_cost / 1e12,
               train_loss, v, which, lr]
        out.write(",".join(f"{r:.6g}" if isinstance(r, float) else str(r) for r in row) + "\n")
        out.flush()
        print(f"[{name} s{a.seed}] step {step:6d}  {which:5s}  val {v:.3f}  "
              f"train {train_loss:.3f}  cost {cost['6pbt'] / 1e12:.2f} TFLOPs", flush=True)

    t0 = time.time()
    for step in range(a.steps + 1):
        lr = lr_at(a, step, last_reset)
        mean_train = window_loss / window_n if window_n else float("nan")
        if step == a.switch_at:
            # Two rows at the same step and cost: the small model just before growth and the
            # grown model just after, so the jump is measured directly, not inferred.
            log(step, mean_train, lr)
            window_loss, window_n = 0.0, 0
            model = grow(model, build(a.large))
            which = "large"
            optimizer = torch.optim.AdamW(model.parameters(), lr=a.lr)
            per_token = train_flops_per_token(model, D.block_size)
            eval_per_token = eval_flops_per_token(model, D.block_size)
            last_reset = step
            lr = lr_at(a, step, last_reset)
            log(step, float("nan"), lr)
        elif step % a.log_every == 0:
            log(step, mean_train, lr)
            window_loss, window_n = 0.0, 0
        if step == a.steps:
            break

        for g in optimizer.param_groups:
            g["lr"] = lr
        x, y = stream.next()
        loss = f.cross_entropy(model(x).view(-1, VOCAB), y.view(-1))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        window_loss += loss.item()
        window_n += 1
        for k in cost:
            cost[k] += per_token[k] * tokens_per_step
    out.close()

    summary = {"args": vars(a), "name": name, "wall_seconds": time.time() - t0,
               "tflops": {k: v / 1e12 for k, v in cost.items()}}
    if not a.no_final_eval:
        summary["final_val_full"] = val_loss(model, D.eval_batches(D.val_data, None, device=device))[0]
        summary["final_test"] = val_loss(model, D.eval_batches(D.test_data, None, device=device))[0]
    Path(f"{stem}.json").write_text(json.dumps(summary, indent=2))   # written last: marks the run complete


if __name__ == "__main__":
    main()
