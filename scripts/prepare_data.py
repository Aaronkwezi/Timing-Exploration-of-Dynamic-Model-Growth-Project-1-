"""Download an English text dataset from the Hugging Face Hub and tokenize it with
the GPT-2 BPE (tiktoken) into data/<name>/{train,val,test}.bin (uint16) + meta.json.

    python scripts/prepare_data.py wikitext2      # ~2M tokens, smoke tests
    python scripts/prepare_data.py wikitext103    # ~117M tokens, main experiments
    python scripts/prepare_data.py tinystories    # ~470M tokens, simple-English corpus

See docs/DATASETS.md for why these were chosen and how to cite them.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import tiktoken
from datasets import load_dataset

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from dataloader import DATA_ROOT  # noqa: E402

# name -> (hf path, hf config, {our split: (hf split, slice)}, documents are rows?)
DATASETS = {
    "wikitext2": ("Salesforce/wikitext", "wikitext-2-raw-v1",
                  {"train": ("train", None), "val": ("validation", None), "test": ("test", None)}, False),
    "wikitext103": ("Salesforce/wikitext", "wikitext-103-raw-v1",
                    {"train": ("train", None), "val": ("validation", None), "test": ("test", None)}, False),
    # TinyStories has no test split: the official validation split is halved into val/test.
    "tinystories": ("roneneldan/TinyStories", None,
                    {"train": ("train", None), "val": ("validation", (0, 0.5)), "test": ("validation", (0.5, 1))}, True),
}

CHUNK_ROWS = 20_000


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("name", choices=sorted(DATASETS))
    ap.add_argument("--max-train-tokens", type=int, default=0,
                    help="stop writing train.bin after this many tokens (0 = full split)")
    args = ap.parse_args()

    hf_path, hf_cfg, splits, row_is_doc = DATASETS[args.name]
    enc = tiktoken.get_encoding("gpt2")
    eot = enc.eot_token
    out_dir = DATA_ROOT / args.name
    out_dir.mkdir(parents=True, exist_ok=True)
    counts = {}

    for split, (hf_split, frac) in splits.items():
        ds = load_dataset(hf_path, hf_cfg, split=hf_split)
        if frac:
            n = len(ds)
            ds = ds.select(range(int(frac[0] * n), int(frac[1] * n)))
        limit = args.max_train_tokens if split == "train" and args.max_train_tokens else None
        total = 0
        with open(out_dir / f"{split}.bin", "wb") as f:
            for start in range(0, len(ds), CHUNK_ROWS):
                rows = ds[start:start + CHUNK_ROWS]["text"]
                if not row_is_doc:
                    # WikiText rows are lines of one long article stream; keep them contiguous.
                    rows = ["".join(rows)]
                ids = []
                for toks in enc.encode_ordinary_batch(rows):
                    ids.extend(toks)
                    if row_is_doc:
                        ids.append(eot)
                arr = np.asarray(ids, dtype=np.uint16)
                if limit is not None and total + len(arr) > limit:
                    arr = arr[:limit - total]
                arr.tofile(f)
                total += len(arr)
                if limit is not None and total >= limit:
                    break
        counts[split] = total
        print(f"{args.name}/{split}: {total:,} tokens")

    meta = {"name": args.name, "source": hf_path, "config": hf_cfg, "tokenizer": "gpt2 (tiktoken)",
            "vocab_size": enc.n_vocab, "dtype": "uint16", "tokens": counts}
    (out_dir / "meta.json").write_text(json.dumps(meta, indent=2))


if __name__ == "__main__":
    main()
