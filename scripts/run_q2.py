#!/usr/bin/env python3
"""Run Q2 experiments sequentially, preserving commands, logs and raw CSV."""
import argparse
import csv
import datetime as dt
import hashlib
import itertools
import json
import os
from pathlib import Path
import platform
import random
import subprocess
import sys
import zipfile


def binary_path(name):
    path = Path(name).resolve()
    if not path.is_file() and Path(str(path) + ".exe").is_file():
        path = Path(str(path) + ".exe")
    if not path.is_file():
        raise ValueError(f"binary does not exist: {name}")
    return path


def cases(cfg):
    result = []
    for kind, spec in cfg["experiments"].items():
        ns = spec.get("sizes", [spec.get("n", spec.get("base_n", 0))])
        for dist, threads, n in itertools.product(spec["distributions"], spec["threads"], ns):
            if kind == "weak":
                n = spec["base_n"] * threads
            for variant in spec.get("variants", cfg["variants"]):
                force, order, chunk = variant
                result.append(dict(kind=kind, dist=dist, threads=threads, n=n,
                                   force=force, order=order, chunk=chunk,
                                   steps=spec.get("steps", cfg["steps"]),
                                   warmup=spec.get("warmup", cfg["warmup"])))
        # Matching-N sequential baselines, not the p=1 time at a different N.
        baseline_shapes = {(c["dist"], c["n"], c["steps"], c["warmup"])
                           for c in result if c["kind"] == kind}
        for dist, n, steps, warmup in sorted(baseline_shapes):
            result.append(dict(kind=kind, dist=dist, threads=1, n=n, force="seq",
                               order="original", chunk=0, steps=steps, warmup=warmup))
    for case, repeat in itertools.product(result, range(cfg["repeats"])):
        yield dict(case, repeat=repeat)


def capture(command):
    try:
        p = subprocess.run(command, capture_output=True, text=True, encoding="utf-8",
                           errors="replace", timeout=20)
        return (p.stdout or "").strip() if p.returncode == 0 else (p.stderr or "").strip()
    except (OSError, subprocess.TimeoutExpired) as e:
        return str(e)


def hardware():
    info = dict(platform=platform.platform(), machine=platform.machine(),
                processor=platform.processor(), logical_cpus=os.cpu_count())
    if os.name == "nt":
        import winreg
        with winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE,
                           r"HARDWARE\DESCRIPTION\System\CentralProcessor\0") as key:
            info["processor"] = winreg.QueryValueEx(key, "ProcessorNameString")[0].strip()
    elif Path("/proc/cpuinfo").exists():
        for line in Path("/proc/cpuinfo").read_text().splitlines():
            if line.startswith("model name"):
                info["processor"] = line.partition(":")[2].strip()
                break
    return info


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--binary", default="./bh")
    parser.add_argument("--config", default="experiments/q2-local.json")
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--cxx", default="g++")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    cfg = json.loads(Path(args.config).read_text(encoding="utf-8-sig"))
    binary = binary_path(args.binary)
    runs = list(cases(cfg))
    if not runs or any(r["steps"] <= r["warmup"] or r["warmup"] < 0 for r in runs):
        raise ValueError("the profile needs positive repeats and steps > warmup >= 0")
    random.Random(cfg.get("run_order_seed", 123)).shuffle(runs)
    if args.dry_run:
        print(json.dumps(runs, indent=2))
        print(f"{len(runs)} runs", file=sys.stderr)
        return
    # Fail if the directory exists: do not silently mix prior experiment runs.
    args.output.mkdir(parents=True, exist_ok=False)
    (args.output / "runs").mkdir()
    environment = os.environ.copy()
    for key, value in cfg.get("environment", {}).items():
        environment[key] = str(value)
    metadata = dict(started_utc=dt.datetime.now(dt.timezone.utc).isoformat(),
                    config=cfg, hardware=hardware(), compiler=capture([args.cxx, "--version"]),
                    python=sys.version, binary=str(binary),
                    binary_sha256=hashlib.sha256(binary.read_bytes()).hexdigest(),
                    git_head=capture(["git", "rev-parse", "HEAD"]),
                    git_status=capture(["git", "status", "--short"]),
                    git_diff_sha256=hashlib.sha256(capture(["git", "diff"]).encode()).hexdigest(),
                    openmp_environment={k: v for k, v in environment.items()
                                        if k.startswith(("OMP_", "GOMP_"))},
                    build_command=cfg.get("build_command", "not recorded"), runs=[])
    manifest = args.output / "manifest.json"
    manifest.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    # Snapshot tracked changes; binary hash also identifies untracked sources.
    (args.output / "source.patch").write_text(capture(["git", "diff"]), encoding="utf-8")
    with zipfile.ZipFile(args.output / "source.zip", "w", zipfile.ZIP_DEFLATED) as snapshot:
        for filename in capture(["git", "ls-files", "--cached", "--others", "--exclude-standard"]).splitlines():
            path = Path(filename)
            if path.is_file() and ("docs/results" not in path.as_posix()):
                snapshot.write(path, path.as_posix())
    combined = None
    with (args.output / "steps.csv").open("w", newline="", encoding="utf-8") as table:
        for run_id, case in enumerate(runs):
            run_csv = args.output / "runs" / f"{run_id:04d}.csv"
            command = [str(binary), "--backend", "seq" if case["force"] == "seq" else "omp",
                       "--build", "seq" if case["force"] == "seq" else cfg.get("build", "seq"),
                       "--force", case["force"], "--force-order", case["order"],
                       "--chunk", str(case["chunk"]), "--threads", str(case["threads"]),
                       "--dist", case["dist"], "--n", str(case["n"]),
                       "--steps", str(case["steps"]), "--warmup", str(case["warmup"]),
                       "--seed", str(cfg["seed"]), "--theta", str(cfg["theta"]),
                       "--dt", str(cfg["dt"]), "--eps", str(cfg["eps"]),
                       "--csv", str(run_csv), "--quiet"]
            print(f"[{run_id+1}/{len(runs)}] {case}", flush=True)
            entry = dict(case, run_id=run_id, command=command)
            metadata["runs"].append(entry)
            # A timeout or failed process preserves a manifest for diagnosis.
            manifest.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
            try:
                run = subprocess.run(command, env=environment, capture_output=True, text=True,
                                     encoding="utf-8", errors="replace",
                                     timeout=cfg.get("timeout_seconds", 1800))
            except subprocess.TimeoutExpired:
                entry["error"] = "timeout"
                manifest.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
                raise
            (args.output / "runs" / f"{run_id:04d}.log").write_text(
                run.stdout + run.stderr, encoding="utf-8")
            entry["returncode"] = run.returncode
            if run.returncode:
                manifest.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
                raise RuntimeError(f"run {run_id} failed; see its log")
            with run_csv.open(newline="", encoding="utf-8") as raw:
                rows = list(csv.DictReader(raw))
            if len(rows) != case["steps"]:
                raise RuntimeError(f"run {run_id}: incomplete CSV")
            if combined is None:
                combined = csv.DictWriter(table, fieldnames=["kind", "run_id", "repeat"] + list(rows[0]))
                combined.writeheader()
            for row in rows:
                if case["force"] != "seq" and int(row["actual_threads"]) != case["threads"]:
                    raise RuntimeError(f"run {run_id}: actual team differs from requested scaling point")
                combined.writerow(dict(kind=case["kind"], run_id=run_id, repeat=case["repeat"], **row))
            table.flush()
    metadata["completed_utc"] = dt.datetime.now(dt.timezone.utc).isoformat()
    manifest.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(f"Saved {len(runs)} runs to {args.output}", flush=True)


if __name__ == "__main__":
    main()
