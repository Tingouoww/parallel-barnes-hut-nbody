#!/usr/bin/env python3
"""Q4: accuracy, work and parallel efficiency of the force calculation as theta changes.

    python3 scripts/run_q4.py --config experiments/q4-local.json --output results/q4-local

Writes steps.csv (one row per time step, plus `phase` and `repeat` columns) and
meta.json to the output directory. Standard library only. The phases:

  error   one validated step per distribution, size and theta
  energy  a long validated run at a small size, for the energy drift
  timing  the sequential baseline and every thread count, without validation
"""
import argparse
import csv
import itertools
import json
import random
from pathlib import Path

import experiment


def cases(cfg):
    """(phase, repeat, bh arguments) of every run."""
    seq = ["--backend", "seq"]
    omp = ["--backend", "omp", "--build", cfg["build"], "--com", cfg["com"], "--force", cfg["force"],
           *cfg["force_args"]]
    energy = cfg["energy"]
    for dist, theta in itertools.product(cfg["distributions"], cfg["thetas"]):
        shape = ["--dist", dist, "--theta", str(theta), "--samples", str(cfg["samples"])]
        yield "energy", 0, shape + omp + ["--n", str(energy["n"]), "--steps", str(energy["steps"]),
                                          "--validate-every", str(energy["every"])]
        for n in cfg["sizes"]:
            yield "error", 0, shape + omp + ["--n", str(n), "--steps", "1", "--validate-every", "1"]
            timed = shape + ["--n", str(n), "--steps", str(cfg["steps"]), "--warmup", str(cfg["warmup"])]
            for repeat in range(cfg["repeats"]):
                yield "timing", repeat, timed + seq
                for threads in cfg["threads"]:
                    yield "timing", repeat, timed + omp + ["--threads", str(threads)]


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--binary", default="./bh")
    parser.add_argument("--config", required=True)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--dry-run", action="store_true", help="list the runs and exit")
    args = parser.parse_args()

    cfg = json.loads(Path(args.config).read_text(encoding="utf-8"))
    runs = list(cases(cfg))
    random.Random(cfg["run_order_seed"]).shuffle(runs)  # spreads slow drift over all configurations
    if args.dry_run:
        for phase, repeat, options in runs:
            print(phase, repeat, " ".join(options))
        print(len(runs), "runs")
        return

    env, meta = experiment.begin(cfg, args.binary, args.output)
    one_run = args.output / "run.csv"
    writer = None
    with (args.output / "steps.csv").open("w", newline="", encoding="utf-8") as out:
        for i, (phase, repeat, options) in enumerate(runs, 1):
            print(f"[{i}/{len(runs)}] {phase} {repeat} {' '.join(options)}", flush=True)
            experiment.run([args.binary, "--quiet", "--csv", str(one_run), "--seed", str(cfg["seed"])] + options,
                           env)
            with one_run.open(newline="", encoding="utf-8") as f:
                rows = list(csv.DictReader(f))
            one_run.unlink()
            if writer is None:
                writer = csv.DictWriter(out, fieldnames=["phase", "repeat"] + list(rows[0]))
                writer.writeheader()
            for row in rows:
                writer.writerow(dict(row, phase=phase, repeat=repeat))
            out.flush()
    experiment.finish(meta, args.output)
    print(f"saved {len(runs)} runs to {args.output}")


if __name__ == "__main__":
    main()
