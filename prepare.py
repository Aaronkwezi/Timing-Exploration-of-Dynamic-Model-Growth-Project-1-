"""Download WikiText-103 and turn it into GPT-2 token files in data/wikitext103/. Run once:  python prepare.py"""
from pathlib import Path

import numpy as np
import tiktoken
from datasets import load_dataset

out = Path(__file__).parent / "data" / "wikitext103"
if (out / "test.bin").exists():
    raise SystemExit("already built")
out.mkdir(parents=True, exist_ok=True)
enc = tiktoken.get_encoding("gpt2")
raw = load_dataset("Salesforce/wikitext", "wikitext-103-raw-v1")
for split, hf_split in {"train": "train", "val": "validation", "test": "test"}.items():
    rows, total = raw[hf_split], 0
    with open(out / f"{split}.bin", "wb") as f:
        for i in range(0, len(rows), 20_000):  # rows are lines of one long article stream: keep them contiguous
            ids = enc.encode_ordinary("".join(rows[i:i + 20_000]["text"]))
            np.asarray(ids, dtype=np.uint16).tofile(f)
            total += len(ids)
    print(f"{split}: {total:,} tokens")
