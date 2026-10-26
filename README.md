# Timing Exploration of Dynamic Model Growth

Train a small Transformer, grow it into a large one partway through, and see when that saves
training compute. Everything except the model idea is already set up.

## Setup (once)

```
pip install -r requirements.txt     # install PyTorch for your GPU first: pytorch.org/get-started/locally
python prepare.py                   # downloads + tokenizes the datasets (~1 GB)
```

## Run things

```
python train.py                                               # small model, 2000 steps
python train.py --model large                                 # sizes: tiny mini small medium large
python train.py --model small --then large --switch-at 1000   # grow small -> large at step 1000
python train.py --model my_model.py                           # test your own architecture
python plot.py                                                # compare all runs -> results/compare.png
```

Add `--data wikitext2 --steps 200` to any run for a quick test (about 40 seconds).
Other options: `--data tinystories`, `--steps N`, `--name`. Run `python train.py -h` for the list.

## Where you work

- **`grow.py`**: write how the small model initializes the large one. This is the research part.
- **`model.py`**: the default model and the sizes. To try a new architecture, make a `.py` file with
  `build(vocab_size, block_size)` returning a torch module (instructions at the top of `model.py`).

## What you get

Every run is saved in `runs/<name>/` (`log.csv`, `summary.json`). Cost is the TFLOPs spent so far,
measured automatically (including the jump in cost when the model grows), so any two runs can be
compared with `python plot.py`. Its table has a "val loss @ same cost" column: the fair comparison.

Datasets (details and citations in [docs/DATASETS.md](docs/DATASETS.md)): `wikitext103` (default),
`tinystories`, `wikitext2` (tiny, for tests).

`data.py`, `prepare.py`, `train.py` and `plot.py` are plumbing; you shouldn't need to edit them.
Do each experiment on its own git branch.
