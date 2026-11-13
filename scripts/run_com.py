#!/usr/bin/env python3
"""Centre of mass: per-level profile of --com level over distributions, sizes and thread counts.

    make bench
    python3 scripts/run_com.py --config experiments/com-local.json --output results/com-local

Runs build/float/com_levels for every combination and writes levels.csv (one
row per tree level, plus a `repeat` column) and meta.json. Standard library only.
"""
import argparse
import csv
import io
import itertools
import json
import random
from pathlib import Path

import experiment


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--binary", default="build/float/com_levels")
    parser.add_argument("--config", required=True)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()

    cfg = json.loads(Path(args.config).read_text(encoding="utf-8"))
    runs = list(itertools.product(cfg["distributions"], cfg["sizes"], cfg["threads"], range(cfg["repeats"])))
    random.Random(cfg["run_order_seed"]).shuffle(runs)

    env, meta = experiment.begin(cfg, args.binary, args.output)
    writer = None
    with (args.output / "levels.csv").open("w", newline="", encoding="utf-8") as out:
        for i, (dist, n, threads, repeat) in enumerate(runs, 1):
            print(f"[{i}/{len(runs)}] {dist} n={n} threads={threads} repeat={repeat}", flush=True)
            text = experiment.run([args.binary, dist, str(n), str(threads), cfg["build"], str(cfg["reps"])], env)
            rows = list(csv.DictReader(io.StringIO(text)))
            if writer is None:
                writer = csv.DictWriter(out, fieldnames=["repeat"] + list(rows[0]))
                writer.writeheader()
            for row in rows:
                writer.writerow(dict(row, repeat=repeat))
            out.flush()
    experiment.finish(meta, args.output)
    print(f"saved {len(runs)} runs to {args.output}")


if __name__ == "__main__":
    main()
