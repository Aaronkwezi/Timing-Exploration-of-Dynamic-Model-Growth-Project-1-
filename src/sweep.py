"""Run small / large / grow@t over several seeds, in parallel, resumable.

  python src/sweep.py --seeds 0 1 2 3 4 --grow 250 500 1000 1500 --out_dir runs/main --jobs 2
Anything this script does not recognise is passed to train.py, e.g.
  python src/sweep.py --seeds 0 1 2 --grow 500 --steps 8000 --log_every 25 --out_dir runs/long
"""
import argparse
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

p = argparse.ArgumentParser()
p.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
p.add_argument("--grow", type=int, nargs="*", default=[250, 500, 1000, 1500])
p.add_argument("--out_dir", default="runs/main")
p.add_argument("--jobs", type=int, default=1, help="runs in parallel")
p.add_argument("--threads", type=int, default=1, help="torch threads per run")
a, extra = p.parse_known_args()

out = Path(a.out_dir)
(out / "logs").mkdir(parents=True, exist_ok=True)
train = str(Path(__file__).with_name("train.py"))

jobs = []
for seed in a.seeds:
    for mode in [["small"], ["large"]] + [["grow", str(t)] for t in a.grow]:
        name = mode[0] if mode[0] != "grow" else f"grow{mode[1]}"
        if (out / f"results_{name}_s{seed}.json").exists():      # finished earlier
            continue
        cmd = [sys.executable, train, *mode, "--seed", str(seed), "--out_dir", str(out),
               "--threads", str(a.threads), *extra]
        jobs.append((f"{name}_s{seed}", cmd))


def run(job):
    name, cmd = job
    with open(out / "logs" / f"{name}.log", "w") as log:
        rc = subprocess.run(cmd, stdout=log, stderr=subprocess.STDOUT).returncode
    print(f"{'ok  ' if rc == 0 else 'FAIL'} {name}", flush=True)
    return rc


print(f"{len(jobs)} runs to do")
with ThreadPoolExecutor(a.jobs) as ex:
    sys.exit(1 if any(ex.map(run, jobs)) else 0)
