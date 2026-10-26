# Timing Exploration of Dynamic Model Growth (Project 1)

Datasets for the project: English text corpora downloaded from the Hugging Face Hub and tokenized
with the GPT-2 BPE into flat `uint16` token files, plus a small loader that serves `(x, y)` batches
to any PyTorch language model. See [docs/DATASETS.md](docs/DATASETS.md) for sizes, the reasons
these corpora were picked, and citations.

## The project in brief

Train a small Transformer M0 (dimension d0) and a large one M1 (d1 > d0) from scratch. Then
initialize a large model M1' from a partly trained M0 after some number of epochs and keep
training it, and find out when that switch saves training cost. Compare training cost (TFLOPs)
against accuracy (validation loss / test error) across different growth timings and deepening
methods.

Deliverables: an IEEE-proceedings-format report (at least 5 two-column letter pages, at least 10
references; abstract, introduction, related work, method, implementation, testbed, metrics,
results), plus the source code and dataset.

## Repository layout

```
dataloader.py            TokenDataset: batches from the prepared token files
scripts/prepare_data.py  download + tokenize a dataset into data/<name>/
scripts/check_setup.py   verify PyTorch/CUDA and every prepared dataset
docs/DATASETS.md         dataset choices, sizes, preprocessing, citations
data/                    generated token files (git-ignored)
```

## Working on it

`main` is only the shared starting point. Do each area of work on its own branch (for example
`git checkout -b small-model` or `git checkout -b growth-timing`) and merge back through a pull
request.

## Setup

```bash
python -m venv .venv            # add --system-site-packages to reuse an existing CUDA PyTorch
.venv/Scripts/activate          # Windows; use `source .venv/bin/activate` on Linux/macOS
pip install -r requirements.txt
```

## Build the datasets

`data/` is git-ignored, so each person generates it locally:

```bash
python scripts/prepare_data.py wikitext2      # ~2.4M train tokens, fast; for quick checks
python scripts/prepare_data.py wikitext103    # ~118M train tokens, ~226 MB; main dataset
python scripts/prepare_data.py tinystories    # ~474M train tokens, ~914 MB; simple English
```

Each produces `data/<name>/{train,val,test}.bin` and `meta.json` (token counts, tokenizer, vocab
size 50,257). Pass `--max-train-tokens N` to write a truncated train split. Then check everything with:

```bash
python scripts/check_setup.py
```

## Use with a model

```python
from dataloader import TokenDataset

ds = TokenDataset("wikitext103")
vocab_size = ds.meta["vocab_size"]                     # 50257 (GPT-2 BPE)
x, y = ds.get_batch("train", 16, 256, "cuda")          # random windows, y = x shifted by 1
for x, y in ds.eval_batches("val", 16, 256, "cuda", max_batches=40):
    ...                                                # fixed windows: same tokens every run
```
