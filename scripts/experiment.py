"""Shared by the run_*.py scripts: output directory, environment and meta.json."""
import datetime
import json
import os
import platform
import subprocess
from pathlib import Path


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


def now():
    return datetime.datetime.now().isoformat(timespec="seconds")


def begin(cfg, binary, output):
    """Creates the output directory and meta.json. Returns the environment for the runs and the metadata."""
    output.mkdir(parents=True)  # refuses to mix with the results of an earlier run
    env = dict(os.environ, **cfg["environment"])
    head = capture(["git", "rev-parse", "--short", "HEAD"])
    meta = dict(config=cfg, started=now(), binary=binary, git_head=head,
                git_dirty=bool(capture(["git", "status", "--porcelain"])) if head else None,
                compiler=capture(["g++", "--version"]).split("\n")[0], cpu=cpu_model(),
                logical_cpus=os.cpu_count(), platform=platform.platform(),
                omp_environment={k: v for k, v in env.items() if k.startswith(("OMP_", "GOMP_"))})
    (output / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    return env, meta


def finish(meta, output):
    meta["finished"] = now()
    (output / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")


def run(command, env):
    done = subprocess.run(command, env=env, capture_output=True, text=True)
    if done.returncode != 0:
        raise SystemExit(f"run failed: {' '.join(command)}\n{done.stdout}{done.stderr}")
    return done.stdout
