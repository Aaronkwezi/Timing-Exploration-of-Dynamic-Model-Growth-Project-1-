"""Load the uint16 token files written by scripts/prepare_data.py as (x, y) batches
of shape (batch, seq_len) for next-token prediction (y is x shifted by one).

    from dataloader import TokenDataset
    ds = TokenDataset("wikitext103")
    x, y = ds.get_batch("train", 16, 256, "cuda")
    for x, y in ds.eval_batches("val", 16, 256, "cuda", max_batches=40): ...
"""
import json
from pathlib import Path

import numpy as np
import torch

DATA_ROOT = Path(__file__).resolve().parent / "data"


class TokenDataset:
    def __init__(self, name: str, root: Path = DATA_ROOT):
        self.dir = Path(root) / name
        meta_path = self.dir / "meta.json"
        if not meta_path.exists():
            raise FileNotFoundError(
                f"{meta_path} missing - run: python scripts/prepare_data.py {name}")
        self.meta = json.loads(meta_path.read_text())
        self._splits = {}

    def split(self, split: str) -> np.memmap:
        if split not in self._splits:
            self._splits[split] = np.memmap(self.dir / f"{split}.bin", dtype=np.uint16, mode="r")
        return self._splits[split]

    def get_batch(self, split, batch_size, seq_len, device, generator=None):
        """Random contiguous windows (training)."""
        data = self.split(split)
        ix = torch.randint(len(data) - seq_len - 1, (batch_size,), generator=generator)
        x = torch.stack([torch.from_numpy(data[i:i + seq_len].astype(np.int64)) for i in ix.tolist()])
        y = torch.stack([torch.from_numpy(data[i + 1:i + 1 + seq_len].astype(np.int64)) for i in ix.tolist()])
        return _to(x, device), _to(y, device)

    def eval_batches(self, split, batch_size, seq_len, device, max_batches):
        """Deterministic non-overlapping windows from the start of a split, so every
        run is scored on exactly the same tokens."""
        data = self.split(split)
        n_windows = min((len(data) - 1) // seq_len, batch_size * max_batches)
        for b in range(0, n_windows, batch_size):
            starts = range(b * seq_len, min(b + batch_size, n_windows) * seq_len, seq_len)
            x = torch.stack([torch.from_numpy(data[s:s + seq_len].astype(np.int64)) for s in starts])
            y = torch.stack([torch.from_numpy(data[s + 1:s + 1 + seq_len].astype(np.int64)) for s in starts])
            yield _to(x, device), _to(y, device)


def _to(t, device):
    if torch.device(device).type == "cuda":
        return t.pin_memory().to(device, non_blocking=True)
    return t.to(device)
