"""Compare every run in runs/ :  python plot.py

Writes results/compare.png (validation loss vs compute cost) and prints a table. The key column is
"val loss @ same cost": every run's loss at the most TFLOPs ALL runs reached, i.e. a fair comparison.
"""
import csv
import json
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

root = Path(__file__).parent
runs = {}
for f in sorted((root / "runs").glob("*/log.csv")):
    rows = [r for r in csv.DictReader(open(f)) if r["val_loss"]]
    if rows:
        runs[f.parent.name] = rows
if not runs:
    raise SystemExit("No runs yet. Try:  python train.py --steps 200")

fig, ax = plt.subplots(figsize=(8, 5))
curves = {}
for name, rows in runs.items():
    x = [float(r["tflops"]) for r in rows]
    y = [float(r["val_loss"]) for r in rows]
    curves[name] = (x, y)
    line, = ax.plot(x, y, label=name, lw=1.5)
    for r in rows:
        if r["note"].startswith("grown"):
            ax.plot(float(r["tflops"]), float(r["val_loss"]), "o", color=line.get_color())  # dot = growth moment
ax.set_xlabel("training compute so far (TFLOPs)")
ax.set_ylabel("validation loss")
ax.grid(alpha=0.3)
ax.legend(fontsize=7, frameon=False)
fig.tight_layout()
(root / "results").mkdir(exist_ok=True)
fig.savefig(root / "results" / "compare.png", dpi=150)

same = min(x[-1] for x, _ in curves.values())
print(f"{'run':<44}{'TFLOPs':>9}{'val loss':>10}{'test loss':>11}{'val loss @ ' + format(same, '.0f') + ' TFLOPs':>24}")
for name, (x, y) in curves.items():
    s = json.loads((root / "runs" / name / "summary.json").read_text()) if (root / "runs" / name / "summary.json").exists() else {}
    print(f"{name:<44}{x[-1]:>9.1f}{y[-1]:>10.3f}{s.get('test_loss', float('nan')):>11.3f}{np.interp(same, x, y):>24.3f}")
print("\nplot: results/compare.png")
