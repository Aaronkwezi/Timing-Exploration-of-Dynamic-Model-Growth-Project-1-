"""Batches of text for training. train.py uses this; you shouldn't need to touch it."""
import json
from pathlib import Path

import numpy as np
import torch


class Data:
    def __init__(self, name):
        d = Path(__file__).parent / "data" / name
        if not (d / "meta.json").exists():
            raise SystemExit(f"Dataset '{name}' isn't built yet. Run:  python prepare.py {name}")
        self.vocab_size = json.loads((d / "meta.json").read_text())["vocab_size"]
        self.split = {s: np.memmap(d / f"{s}.bin", dtype=np.uint16, mode="r") for s in ("train", "val", "test")}
        self.train_tokens = len(self.split["train"])

    def _xy(self, data, starts, T, device):
        w = torch.stack([torch.from_numpy(data[i:i + T + 1].astype(np.int64)) for i in starts])
        return w[:, :-1].to(device), w[:, 1:].to(device)   # target = input shifted by one token

    def train_batch(self, batch_size, T, device, gen):
        data = self.split["train"]
        starts = torch.randint(len(data) - T - 1, (batch_size,), generator=gen).tolist()
        return self._xy(data, starts, T, device)

    def eval_batches(self, split, batch_size, T, device, max_batches):
        """Fixed windows from the start of the split, so every run is scored on the same text."""
        data = self.split[split]
        n = min((len(data) - 1) // T, batch_size * max_batches)
        for b in range(0, n, batch_size):
            yield self._xy(data, [i * T for i in range(b, min(b + batch_size, n))], T, device)
