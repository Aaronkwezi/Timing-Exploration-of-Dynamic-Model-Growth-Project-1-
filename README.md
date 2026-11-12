# Timing Exploration of Dynamic Model Growth

Train a small Transformer, grow it into a large one partway through, and see when that saves
training compute. Runs are compared by validation loss at the same cost (TFLOPs).

## Setup (once)

```
pip install -r requirements.txt     # install PyTorch for your GPU first: pytorch.org/get-started/locally
python src/prepare.py               # only if train.pt / val.pt / test.pt are missing
```

`prepare.py` tokenizes WikiText-2 with the GPT-2 tokenizer and writes `train.pt`, `val.pt` and
`test.pt` to the current folder. These files are already in the repo.

## Run things

```
python src/train.py small          # small model the whole time (2000 steps)
python src/train.py large          # large model the whole time
python src/train.py grow 1000      # small model, grown into the large one at step 1000
```

Options: `--seed`, `--steps`, `--log_every`, `--small/--large width,heads,layers`, `--schedule cosine`,
`--warmup`, `--post_grow_warmup`, `--val_batches`, `--out_dir`, `--device`.

Many seeds and all configurations, in parallel and resumable, then the paired analysis:

```
python src/sweep.py --seeds 0 1 2 3 4 --grow 250 500 1000 1500 --out_dir runs/main --jobs 2 --log_every 25
python src/analyze.py runs/main --plot runs/main/results.png
```

Each run writes `results_<name>_s<seed>.csv` with columns `step, tflops, tflops_matmul, tflops_full,
tflops_eval, train_loss, val_loss, model, lr`, plus a `.json` summary (full-validation and test loss).
`tflops` is the original 6PBT proxy; the others are the matmul-only and matmul + attention proxies
(`src/flops.py`). At the growth step two rows are logged (small just before, large just after) so the jump is
measured directly. Training batches depend only on `--seed` and the step, so all runs with one seed see
identical data; validation uses a fixed, seed-independent set of tokens.

## Files

- `src/model.py`: the Transformer. Small = 64 wide, 2 heads, 2 layers; large = 128 wide, 4 heads, 4 layers.
- `src/growth.py`: `grow(small, large)` copies each small weight into the top-left corner of the
  matching large weight; extra width and extra layers keep their random start.
- `src/train.py`: training (AdamW, fresh optimizer after growing, optional warm-up / cosine schedule).
- `src/sweep.py`, `src/analyze.py`: multi-seed runner and paired matched-cost analysis.
- `src/flops.py`: the three cost proxies.
- `src/data.py`: loads the token files and draws random batches (16 sequences of 64 tokens).
- `src/prepare.py`: builds the token files.

## Notes

- Cost defaults to 6 x parameters x tokens per step (`tflops`); `tflops_matmul` and `tflops_full` are alternatives.
- The corner copy does not keep the model's output the same: the loss jumps right after growing.

Do each experiment on its own git branch.
