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

Run from the repo folder, so `data.py` can find the `.pt` files:

```
python src/train.py small          # small model the whole time (2000 steps)
python src/train.py large          # large model the whole time
python src/train.py grow 1000      # small model, grown into the large one at step 1000
```

Each run writes `results_<name>.csv` (`step, tflops, val_loss`), e.g. `results_grow1000.csv`.
Validation loss is measured every 100 steps on the same 20 validation batches in every run.

## Files

- `src/model.py`: the Transformer. Small = 64 wide, 2 heads, 2 layers; large = 128 wide, 4 heads, 4 layers.
- `src/growth.py`: `grow(small, large)` copies each small weight into the top-left corner of the
  matching large weight; extra width and extra layers keep their random start.
- `src/train.py`: training (AdamW, lr 3e-4, fresh optimizer after growing).
- `src/data.py`: loads the token files and draws random batches (16 sequences of 64 tokens).
- `src/prepare.py`: builds the token files.

## Notes

- Cost is estimated as 6 x parameters x tokens per step, so it includes the jump when the model grows.
- The corner copy does not keep the model's output the same: the loss jumps right after growing.

Do each experiment on its own git branch.
