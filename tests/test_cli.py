#!/usr/bin/env python3
"""CLI integration checks (stdlib only). Run after building bh."""
import csv
import math
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from run_q2 import binary_path
from analyze_q2 import summarize

BINARY = binary_path(os.environ.get("BH", "./bh"))


class CliTests(unittest.TestCase):
    def run_bh(self, *args, ok=True):
        env = dict(os.environ, OMP_NUM_THREADS="4", OMP_THREAD_LIMIT="2")
        p = subprocess.run([str(BINARY), *map(str, args)], env=env, capture_output=True, text=True, timeout=30)
        self.assertEqual(p.returncode == 0, ok, p.stdout + p.stderr)
        return p

    def test_csv_metrics_and_physics(self):
        with tempfile.TemporaryDirectory() as folder:
            baseline = Path(folder) / "seq.csv"
            common = ["--n", "71", "--steps", "4", "--warmup", "1", "--dist", "collision",
                      "--validate-every", "1", "--check-tree", "--quiet"]
            self.run_bh(*common, "--csv", baseline)
            with baseline.open() as f:
                want = list(csv.DictReader(f))
            for force in ("static", "dynamic", "guided", "costzones"):
                for order in ("morton", "original"):
                    if force == "costzones" and order == "original":
                        continue
                    path = Path(folder) / f"{force}-{order}.csv"
                    self.run_bh(*common, "--backend", "omp", "--force", force,
                                "--force-order", order, "--threads", "8", "--chunk", "3", "--csv", path)
                    with path.open() as f:
                        rows = list(csv.DictReader(f))
                    self.assertEqual(len(rows), 4)
                    for step, row in enumerate(rows):
                        work = list(map(int, row["thread_interactions"].split(";")))
                        self.assertEqual(len(work), int(row["actual_threads"]))
                        self.assertLessEqual(len(work), 2)
                        total = int(row["interactions"])
                        self.assertEqual(sum(work), total)
                        self.assertAlmostEqual(float(row["load_imbalance"]),
                                               max(work) * len(work) / total if total else 1)
                        self.assertEqual(row["is_warmup"], "1" if step == 0 else "0")
                        self.assertEqual(row["force_order"], order)
                        for metric in ("nodes", "interactions", "err_median", "err_p99", "err_rms",
                                       "energy", "energy_drift"):
                            if metric in ("nodes", "interactions"):
                                self.assertEqual(row[metric], want[step][metric], metric)
                            else:
                                # Validation reductions may add double values in a
                                # different order; per-particle force equality is
                                # checked bit-for-bit in test_parallel.cpp.
                                self.assertTrue(math.isclose(float(row[metric]), float(want[step][metric]),
                                                             rel_tol=1e-12, abs_tol=1e-14), metric)
                        components = sum(float(row[name]) for name in
                                         ("force_order_s", "force_partition_s", "force_walk_s"))
                        self.assertLessEqual(components, float(row["force_s"]) + 1e-5)
            self.run_bh("--n", "1", "--backend", "omp", "--force", "costzones", "--threads", "8",
                        "--steps", "2", "--csv", Path(folder) / "zero.csv")
            with (Path(folder) / "zero.csv").open() as f:
                for row in csv.DictReader(f):
                    self.assertEqual(float(row["load_imbalance"]), 1)
                    self.assertEqual(int(row["interactions"]), 0)

    def test_csv_append_and_old_header(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "test.csv"
            self.run_bh("--n", "2", "--steps", "1", "--csv", path)
            self.run_bh("--n", "2", "--steps", "1", "--csv", path)
            with path.open() as f:
                self.assertEqual(len(list(csv.DictReader(f))), 2)
            path.write_text("old,header\n", encoding="utf-8")
            p = self.run_bh("--n", "2", "--steps", "1", "--csv", path, ok=False)
            self.assertIn("CSV header mismatch", p.stderr)
            self.assertEqual(path.read_text(), "old,header\n")

    def test_invalid_options(self):
        for args in (["--chunk", "-1"], ["--threads", "-1"], ["--backend", "omp", "--force-order", "bad"],
                     ["--backend", "omp", "--force", "costzones", "--force-order", "original"],
                     ["--force", "dynamic"]):
            self.run_bh("--n", "2", "--steps", "1", *args, ok=False)

    def test_summary_excludes_warmup_and_uses_matching_n_baseline(self):
        rows = []
        for kind, n, threads, force, seconds in (
                ("strong", "100", "1", "seq", 8), ("strong", "100", "2", "static", 2),
                ("weak", "100", "1", "seq", 8), ("weak", "100", "1", "static", 4),
                ("weak", "200", "1", "seq", 20), ("weak", "200", "2", "static", 5)):
            for repeat in range(3):
                for warmup in ("1", "0", "0"):
                    row = dict(kind=kind, n=n, threads=threads, force=force, dist="uniform",
                               force_order="original" if force == "seq" else "morton", chunk="0",
                               run_id=f"{kind}-{n}-{threads}-{force}-{repeat}", is_warmup=warmup)
                    for metric in ("bbox_s", "build_s", "com_s", "force_s", "integrate_s", "total_s",
                                   "load_imbalance", "force_order_s", "force_partition_s", "force_walk_s"):
                        row[metric] = str(999 if warmup == "1" else seconds)
                    rows.append(row)
        summaries = summarize(rows)
        strong = next(s for s in summaries if s["kind"] == "strong" and s["force"] == "static")
        self.assertEqual(strong["force_speedup"], 4)
        self.assertEqual(strong["force_efficiency"], 2)
        weak = next(s for s in summaries if s["kind"] == "weak" and s["threads"] == "2")
        self.assertEqual(weak["force_speedup"], 4)  # 20 / 5 at N=200
        self.assertEqual(weak["weak_time_efficiency"], 0.8)  # 4 / 5 at N=100 -> 200
        self.assertEqual(weak["repeats"], 3)
        weak_seq = next(s for s in summaries if s["kind"] == "weak" and s["force"] == "seq" and s["n"] == "200")
        self.assertEqual(weak_seq["weak_time_efficiency"], 0.4)  # 8 / 20, always smallest-N reference


if __name__ == "__main__":
    unittest.main()
