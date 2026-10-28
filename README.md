# Timing Exploration of Dynamic Model Growth

Train a small Transformer, grow it into a large one partway through, and see when that saves
training compute. Data, training, cost measurement and plotting are set up; the models and the growth method are yours to write.

## Setup (once)

```
pip install -r requirements.txt     # install PyTorch for your GPU first: pytorch.org/get-started/locally
python prepare.py                   # downloads + tokenizes WikiText-103 (~230 MB)
```

## Run things

```
python train.py                                               # model.py, 2000 steps
python train.py --model small.py --then large.py --switch-at 1000   # grow at step 1000
python plot.py                                                # compare all runs -> results/compare.png
```

Add `--steps 200` to any run for a quick test. Other options: `--name`. Run `python train.py -h` for the list.

## Where you work

- **`grow.py`**: write how the small model initializes the large one. This is the research part.
- **`model.py`**: write the model. Any `.py` file with `build(vocab_size, block_size)` returning a torch
  module works (contract at the top of `model.py`); write a second one for the large model.

## What you get

Every run is saved in `runs/<name>/` (`log.csv`, `summary.json`). Cost is the TFLOPs spent so far,
measured automatically (including the jump in cost when the model grows), so any two runs can be
compared with `python plot.py`. Its table has a "val loss @ same cost" column: the fair comparison.

Dataset: WikiText-103 (details and citations in [docs/DATASETS.md](docs/DATASETS.md)).

`data.py`, `prepare.py`, `train.py` and `plot.py` are plumbing; you shouldn't need to edit them.
Do each experiment on its own git branch.
