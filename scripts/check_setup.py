"""Check that PyTorch/CUDA work and every prepared dataset loads correctly.

    python scripts/check_setup.py
"""
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from dataloader import DATA_ROOT, TokenDataset  # noqa: E402

print(f"torch {torch.__version__}, CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    p = torch.cuda.get_device_properties(0)
    print(f"GPU: {p.name}, {p.total_memory / 2**30:.1f} GiB, bf16: {torch.cuda.is_bf16_supported()}")
device = "cuda" if torch.cuda.is_available() else "cpu"

names = sorted(d.name for d in DATA_ROOT.glob("*") if (d / "meta.json").exists()) if DATA_ROOT.exists() else []
if not names:
    sys.exit("no datasets found - run: python scripts/prepare_data.py wikitext2")
for name in names:
    ds = TokenDataset(name)
    x, y = ds.get_batch("train", 4, 64, device)
    assert x.shape == (4, 64) and torch.equal(x[:, 1:], y[:, :-1]), "targets must be inputs shifted by one"
    assert int(x.max()) < ds.meta["vocab_size"]
    for split, n in ds.meta["tokens"].items():
        assert len(ds.split(split)) == n, f"{name}/{split} size mismatch - re-run prepare_data.py"
    counts = ", ".join(f"{s} {n:,}" for s, n in ds.meta["tokens"].items())
    print(f"{name}: OK ({counts} tokens)")
