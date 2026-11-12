"""Aggregate a sweep: paired, multi-seed comparison at matched training cost.

  python src/analyze.py runs/main                       # tables for all three cost proxies
  python src/analyze.py runs/main --cost tflops --plot results.png

Every grown run is compared with the large-only run of the *same seed* (same data order,
same evaluation tokens), at the same cost, so the reported gap is a paired difference with a
95% t-interval over seeds. Positive = the grown run is worse than large-only.
"""
import argparse
import csv
import re
from collections import defaultdict
from pathlib import Path

import warnings

import numpy as np

warnings.filterwarnings("ignore", category=RuntimeWarning)   # all-NaN slices beyond a run's cost range

COSTS = {"tflops": "6PBT (all params)", "tflops_matmul": "matmul params only",
         "tflops_full": "matmul + attention scores"}
T95 = [12.706, 4.303, 3.182, 2.776, 2.571, 2.447, 2.365, 2.306, 2.262, 2.228, 2.201, 2.179, 2.160,
       2.145, 2.131, 2.120, 2.110, 2.101, 2.093, 2.086, 2.080, 2.074, 2.069, 2.064, 2.060, 2.056,
       2.052, 2.048, 2.045, 2.042]


def load(folder):
    runs = defaultdict(dict)                       # runs[name][seed] = {column: array}
    for path in sorted(Path(folder).glob("results_*_s*.csv")):
        m = re.fullmatch(r"results_(.+)_s(\d+)\.csv", path.name)
        with open(path) as fh:
            rows = list(csv.DictReader(fh))
        cols = {k: [r[k] for r in rows] for k in rows[0]}
        runs[m[1]][int(m[2])] = {k: (np.array(v) if k == "model" else np.array(v, float))
                                 for k, v in cols.items()}
    return runs


def curve(run, cost, budgets):
    """Val loss at the given cost budgets (linear interpolation; NaN outside the run's range)."""
    step = run["step"]
    last = {s: i for i, s in enumerate(step)}      # at a growth step keep the post-growth row
    idx = sorted(last.values())
    return np.interp(budgets, run[cost][idx], run["val_loss"][idx], left=np.nan, right=np.nan)


def mean_ci(x):
    x = np.asarray(x, float)
    x = x[~np.isnan(x)]
    n = len(x)
    if n == 0:
        return np.nan, np.nan, 0
    if n == 1:
        return x[0], np.nan, 1
    half = (T95[n - 2] if n - 1 <= len(T95) else 1.96) * x.std(ddof=1) / np.sqrt(n)
    return x.mean(), half, n


def order(names):
    g = sorted((n for n in names if n.startswith("grow")), key=lambda n: int(n[4:]))
    return [n for n in ("small",) if n in names] + g + [n for n in ("large",) if n in names]


def total_cost(runs, name, cost):
    return float(np.median([r[cost][-1] for r in runs[name].values()]))


def table(runs, cost):
    names = order(runs)
    budgets = sorted({round(total_cost(runs, n, cost), 1) for n in names})
    print(f"\n### Paired gap to large-only at equal cost, cost = {COSTS[cost]}\n")
    print("mean ± 95% CI over seeds (k/n = seeds where large-only was better). Blank = run never spent that much.\n")
    others = [n for n in names if n != "large"]
    print("| TFLOPs | large val | " + " | ".join(others) + " |")
    print("|---:|---:|" + "---:|" * len(others))
    for b in budgets:
        lv = [curve(r, cost, [b])[0] for r in runs["large"].values()]
        m, h, n = mean_ci(lv)
        cells = []
        for name in others:
            d = [curve(r, cost, [b])[0] - curve(runs["large"][s], cost, [b])[0]
                 for s, r in runs[name].items() if s in runs["large"]]
            d = np.array(d)
            d = d[~np.isnan(d)]
            if len(d) == 0:
                cells.append("")
                continue
            dm, dh, dn = mean_ci(d)
            ci = "" if np.isnan(dh) else f" ±{dh:.3f}"
            cells.append(f"{dm:+.3f}{ci} ({int((d > 0).sum())}/{dn})")
        lci = "" if np.isnan(h) else f" ±{h:.3f}"
        print(f"| {b:.1f} | {m:.3f}{lci} | " + " | ".join(cells) + " |")


def noise(runs, cost):
    if "large" not in runs:
        return
    top = total_cost(runs, "large", cost)
    v = [curve(r, cost, [top])[0] for r in runs["large"].values()]
    print(f"\nSeed-to-seed SD of large-only val loss at {top:.1f} TFLOPs: {np.nanstd(v, ddof=1) if len(v) > 1 else float('nan'):.3f} "
          f"(n={len(v)}). Gaps smaller than about twice this need more seeds.")


def jumps(runs):
    print("\n### Growth jump (same step, same cost: val after growth minus val just before)\n")
    print("| run | jump (nats) | steps to recover pre-growth loss | steps to match small-only |")
    print("|---|---:|---:|---:|")
    for name in order(runs):
        if not name.startswith("grow"):
            continue
        j, rec, beat = [], [], []
        for s, r in runs[name].items():
            sw = int(name[4:])
            at = np.where(r["step"] == sw)[0]
            pre, post = r["val_loss"][at[0]], r["val_loss"][at[-1]]
            j.append(post - pre)
            later = np.where((r["step"] > sw) & (r["val_loss"] <= pre))[0]
            rec.append(r["step"][later[0]] - sw if len(later) else np.nan)
            if "small" in runs and s in runs["small"]:
                sm = runs["small"][s]
                ref = dict(zip(sm["step"], sm["val_loss"]))
                later = [i for i in np.where(r["step"] > sw)[0]
                         if r["step"][i] in ref and r["val_loss"][i] <= ref[r["step"][i]]]
                beat.append(r["step"][later[0]] - sw if later else np.nan)
        f = lambda x: (lambda m, h, n: f"{m:.3f}" + ("" if np.isnan(h) else f" ±{h:.3f}") + f" (n={n})")(*mean_ci(x))
        print(f"| {name} | {f(j)} | {f(rec)} | {f(beat) if beat else ''} |")


def crossover(runs, cost):
    if "small" not in runs or "large" not in runs:
        return
    hi = total_cost(runs, "small", cost)
    grid = np.linspace(0, hi, 4000)
    xs = []
    for s, r in runs["small"].items():
        if s not in runs["large"]:
            continue
        diff = curve(runs["large"][s], cost, grid) - curve(r, cost, grid)
        ok = np.where(~np.isnan(diff) & (grid > 0) & (diff <= 0))[0]
        xs.append(grid[ok[0]] if len(ok) else np.nan)
    m, h, n = mean_ci(xs)
    ci = "" if np.isnan(h) else f" ±{h:.1f}"
    print(f"\nLarge-only first matches small-only at {m:.1f}{ci} TFLOPs (n={n}; resolution limited by --log_every).")


def plot(runs, cost, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    names = order(runs)
    hi = max(total_cost(runs, n, cost) for n in names)
    grid = np.linspace(0, hi, 600)
    fig, ax = plt.subplots(1, 2, figsize=(13, 4.8))
    for n in names:
        Y = np.array([curve(r, cost, grid) for r in runs[n].values()])
        m = np.nanmean(Y, 0)
        se = np.nanstd(Y, 0, ddof=1) / np.sqrt((~np.isnan(Y)).sum(0)) if len(Y) > 1 else 0 * m
        ax[0].plot(grid, m, label=f"{n} (n={len(Y)})")
        ax[0].fill_between(grid, m - 1.96 * se, m + 1.96 * se, alpha=0.15)
        if n != "large" and "large" in runs:
            D = np.array([curve(r, cost, grid) - curve(runs["large"][s], cost, grid)
                          for s, r in runs[n].items() if s in runs["large"]])
            dm = np.nanmean(D, 0)
            ds = np.nanstd(D, 0, ddof=1) / np.sqrt((~np.isnan(D)).sum(0)) if len(D) > 1 else 0 * dm
            ax[1].plot(grid, dm, label=n)
            ax[1].fill_between(grid, dm - 1.96 * ds, dm + 1.96 * ds, alpha=0.15)
    ax[0].set(xlabel=f"cost, {COSTS[cost]} (TFLOPs)", ylabel="validation loss (nats)", ylim=(None, 8.5))
    ax[1].axhline(0, color="k", lw=0.8)
    ax[1].set(xlabel="cost (TFLOPs)", ylabel="loss minus large-only (paired, 95% CI)", ylim=(-0.3, 0.6))
    for x in ax:
        x.legend(); x.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    print(f"\nplot written to {path}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("folder")
    p.add_argument("--cost", choices=list(COSTS), nargs="*", default=list(COSTS))
    p.add_argument("--plot", default=None, help="png for the first chosen cost")
    a = p.parse_args()
    runs = load(a.folder)
    print(f"{sum(len(v) for v in runs.values())} runs; seeds per config: "
          + ", ".join(f"{n}={len(runs[n])}" for n in order(runs)))
    for c in a.cost:
        table(runs, c)
        noise(runs, c)
        crossover(runs, c)
    jumps(runs)
    if a.plot:
        plot(runs, a.cost[0], a.plot)
