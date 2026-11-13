#!/usr/bin/env python3
"""Q1: time every tree build strategy over distributions, sizes and thread counts.

    python3 scripts/run_q1.py --config experiments/q1-local.json --output results/q1-local

Writes steps.csv (one row per time step, plus a `repeat` column) and meta.json
to the output directory. Standard library only.
"""
import argparse
import csv
import itertools
import json
import random
from pathlib import Path

import experiment


def cases(cfg):
    """The sequential baseline and every build / thread count, for each distribution and size."""
    for dist, n in itertools.product(cfg["distributions"], cfg["sizes"]):
        yield dict(dist=dist, n=n, backend="seq", build="seq", com="seq", force="seq", threads=1)
        for build, threads in itertools.product(cfg["builds"], cfg["threads"]):
            yield dict(dist=dist, n=n, backend="omp", build=build, com=cfg["com"],
                       force=cfg["force"], threads=threads)


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--binary", default="./bh")
    parser.add_argument("--config", required=True)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--dry-run", action="store_true", help="list the runs and exit")
    args = parser.parse_args()

    cfg = json.loads(Path(args.config).read_text(encoding="utf-8"))
    runs = [dict(case, repeat=r) for case in cases(cfg) for r in range(cfg["repeats"])]
    random.Random(cfg["run_order_seed"]).shuffle(runs)  # spreads slow drift over all configurations
    if args.dry_run:
        for run in runs:
            print(run)
        print(len(runs), "runs")
        return

    env, meta = experiment.begin(cfg, args.binary, args.output)
    one_run = args.output / "run.csv"
    writer = None
    with (args.output / "steps.csv").open("w", newline="", encoding="utf-8") as out:
        for i, run in enumerate(runs, 1):
            print(f"[{i}/{len(runs)}] {run}", flush=True)
            experiment.run([args.binary, "--quiet", "--csv", str(one_run), "--seed", str(cfg["seed"]),
                            "--theta", str(cfg["theta"]), "--steps", str(cfg["steps"]),
                            "--warmup", str(cfg["warmup"]), "--dist", run["dist"], "--n", str(run["n"]),
                            "--backend", run["backend"], "--build", run["build"], "--com", run["com"],
                            "--force", run["force"], "--threads", str(run["threads"])], env)
            with one_run.open(newline="", encoding="utf-8") as f:
                rows = list(csv.DictReader(f))
            one_run.unlink()
            if writer is None:
                writer = csv.DictWriter(out, fieldnames=["repeat"] + list(rows[0]))
                writer.writeheader()
            for row in rows:
                writer.writerow(dict(row, repeat=run["repeat"]))
            out.flush()
    experiment.finish(meta, args.output)
    print(f"saved {len(runs)} runs to {args.output}")


if __name__ == "__main__":
    main()
