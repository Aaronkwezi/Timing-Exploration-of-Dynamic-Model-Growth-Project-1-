"""Download the datasets and turn them into token files. Run once:  python prepare.py

Builds wikitext2 (tiny, for quick tests), wikitext103 (main) and tinystories (simple English)
into data/<name>/. Already-built datasets are skipped. To build just one: python prepare.py wikitext2
"""
import json
import sys
from pathlib import Path

import numpy as np
import tiktoken
from datasets import load_dataset

DATA = Path(__file__).parent / "data"

# name -> (huggingface dataset, config, rows are separate documents?)
DATASETS = {
    "wikitext2": ("Salesforce/wikitext", "wikitext-2-raw-v1", False),
    "wikitext103": ("Salesforce/wikitext", "wikitext-103-raw-v1", False),
    "tinystories": ("roneneldan/TinyStories", None, True),
}


def build(name):
    out = DATA / name
    if (out / "meta.json").exists():
        print(f"{name}: already built")
        return
    path, config, docs = DATASETS[name]
    enc = tiktoken.get_encoding("gpt2")
    raw = load_dataset(path, config)
    if "test" not in raw:  # TinyStories has no test split: halve its validation split
        n = len(raw["validation"])
        raw["test"] = raw["validation"].select(range(n // 2, n))
        raw["validation"] = raw["validation"].select(range(n // 2))
    out.mkdir(parents=True, exist_ok=True)
    counts = {}
    for split, hf_split in {"train": "train", "val": "validation", "test": "test"}.items():
        rows, total = raw[hf_split], 0
        with open(out / f"{split}.bin", "wb") as f:
            for i in range(0, len(rows), 20_000):
                texts = rows[i:i + 20_000]["text"]
                if not docs:  # WikiText rows are lines of one long text: keep them contiguous
                    texts = ["".join(texts)]
                ids = []
                for toks in enc.encode_ordinary_batch(texts):
                    ids += toks + ([enc.eot_token] if docs else [])
                np.asarray(ids, dtype=np.uint16).tofile(f)
                total += len(ids)
        counts[split] = total
        print(f"{name}/{split}: {total:,} tokens")
    (out / "meta.json").write_text(json.dumps({"vocab_size": enc.n_vocab, "tokens": counts}, indent=2))


if __name__ == "__main__":
    for n in sys.argv[1:] or DATASETS:
        build(n)
