#!/usr/bin/env python3
"""Q1: time every tree build strategy over distributions, sizes and thread counts.

    python3 scripts/run_q1.py --config experiments/q1-local.json --output results/q1-local

Writes steps.csv (one row per time step, plus a `repeat` column) and meta.json
to the output directory. Standard library only.
"""
import argparse
import csv
import datetime
import itertools
import json
import os
import platform
import random
import subprocess
from pathlib import Path


def cases(cfg):
    """The sequential baseline and every build / thread count, for each distribution and size."""
    for dist, n in itertools.product(cfg["distributions"], cfg["sizes"]):
        yield dict(dist=dist, n=n, backend="seq", build="seq", com="seq", force="seq", threads=1)
        for build, threads in itertools.product(cfg["builds"], cfg["threads"]):
            yield dict(dist=dist, n=n, backend="omp", build=build, com=cfg["com"],
                       force=cfg["force"], threads=threads)


def capture(command):
    try:
        return subprocess.run(command, capture_output=True, text=True).stdout.strip()
    except OSError:
        return ""


def cpu_model():
    try:
        for line in Path("/proc/cpuinfo").read_text().splitlines():
            if line.startswith("model name"):
                return line.partition(":")[2].strip()
    except OSError:
        pass
    return platform.processor()


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

    args.output.mkdir(parents=True)  # refuses to mix with the results of an earlier run
    env = dict(os.environ, **cfg["environment"])
    head = capture(["git", "rev-parse", "--short", "HEAD"])
    meta = dict(config=cfg, started=datetime.datetime.now().isoformat(timespec="seconds"),
                binary=args.binary, git_head=head,
                git_dirty=bool(capture(["git", "status", "--porcelain"])) if head else None,
                compiler=capture(["g++", "--version"]).split("\n")[0], cpu=cpu_model(),
                logical_cpus=os.cpu_count(), platform=platform.platform(),
                omp_environment={k: v for k, v in env.items() if k.startswith(("OMP_", "GOMP_"))})
    meta_file = args.output / "meta.json"
    meta_file.write_text(json.dumps(meta, indent=2), encoding="utf-8")

    one_run = args.output / "run.csv"
    writer = None
    with (args.output / "steps.csv").open("w", newline="", encoding="utf-8") as out:
        for i, run in enumerate(runs, 1):
            print(f"[{i}/{len(runs)}] {run}", flush=True)
            command = [args.binary, "--quiet", "--csv", str(one_run), "--seed", str(cfg["seed"]),
                       "--theta", str(cfg["theta"]), "--steps", str(cfg["steps"]),
                       "--warmup", str(cfg["warmup"]), "--dist", run["dist"], "--n", str(run["n"]),
                       "--backend", run["backend"], "--build", run["build"], "--com", run["com"],
                       "--force", run["force"], "--threads", str(run["threads"])]
            done = subprocess.run(command, env=env, capture_output=True, text=True)
            if done.returncode != 0:
                raise SystemExit(f"run failed: {' '.join(command)}\n{done.stdout}{done.stderr}")
            with one_run.open(newline="", encoding="utf-8") as f:
                rows = list(csv.DictReader(f))
            one_run.unlink()
            if writer is None:
                writer = csv.DictWriter(out, fieldnames=["repeat"] + list(rows[0]))
                writer.writeheader()
            for row in rows:
                writer.writerow(dict(row, repeat=run["repeat"]))
            out.flush()

    meta["finished"] = datetime.datetime.now().isoformat(timespec="seconds")
    meta_file.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(f"saved {len(runs)} runs to {args.output}")


if __name__ == "__main__":
    main()
