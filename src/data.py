import torch
from pathlib import Path

# Token files live next to src/, so scripts work from any directory.
ROOT = Path(__file__).resolve().parent.parent
train_data = torch.load(ROOT / "train.pt", weights_only=True)
val_data = torch.load(ROOT / "val.pt", weights_only=True)
test_data = torch.load(ROOT / "test.pt", weights_only=True)

block_size = 64
batch_size = 16


class BatchStream:

    def __init__(self, data, seed, device="cpu"):
        self.data = data
        self.device = device
        self.gen = torch.Generator().manual_seed(seed)

    def next(self):
        d = self.data
        starts = torch.randint(0, len(d) - block_size, (batch_size,), generator=self.gen)
        x = torch.stack([d[i:i + block_size] for i in starts])
        y = torch.stack([d[i + 1:i + block_size + 1] for i in starts])
        return x.to(self.device), y.to(self.device)


def eval_batches(data, n_batches=None, eval_batch_size=32, device="cpu"):

    n_seq = (len(data) - 1) // block_size
    x = data[:n_seq * block_size].view(n_seq, block_size)
    y = data[1:n_seq * block_size + 1].view(n_seq, block_size)
    if n_batches is not None:
        n_seq = min(n_seq, n_batches * eval_batch_size)
    return [(x[i:i + eval_batch_size].to(device), y[i:i + eval_batch_size].to(device))
            for i in range(0, n_seq, eval_batch_size)]
