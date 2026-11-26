#!/usr/bin/env python3
"""Q1: summarise the build times recorded by run_q1.py.

    python3 scripts/analyze_q1.py results/q1-local

Writes summary.csv, summary.md and, when matplotlib is installed, one
build-n<N>.png per problem size to the same directory.

A run's time is the median over its measured steps; a configuration's time is
the median over repeats (min and max over repeats are kept as the spread).
speedup compares with the sequential build at the same distribution and size,
scaling with the same strategy on one thread.
"""
import csv
import json
import statistics
import sys
from collections import defaultdict
from pathlib import Path


def summarise(folder):
    warmup = json.loads((folder / "meta.json").read_text(encoding="utf-8"))["config"]["warmup"]
    per_run = defaultdict(lambda: defaultdict(list))
    with (folder / "steps.csv").open(newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if int(row["step"]) >= warmup:
                build = row["build"] + ("+reorder" if row.get("reorder") == "1" else "")
                key = (row["dist"], int(row["n"]), build, int(row["threads"]))
                per_run[key][row["repeat"]].append((float(row["build_s"]), float(row["com_s"])))

    table = {}
    for key, repeats in per_run.items():
        build = [statistics.median(b for b, _ in steps) for steps in repeats.values()]
        com = [statistics.median(c for _, c in steps) for steps in repeats.values()]
        table[key] = dict(repeats=len(repeats), build_s=statistics.median(build), build_min=min(build),
                          build_max=max(build), com_s=statistics.median(com))
    for (dist, n, build, threads), row in table.items():
        row["speedup"] = table[dist, n, "seq", 1]["build_s"] / row["build_s"]
        one = table.get((dist, n, build, 1))
        row["scaling"] = one["build_s"] / row["build_s"] if one else ""
    return table


def write_csv(table, path):
    fields = ["dist", "n", "build", "threads", "repeats", "build_s", "build_min", "build_max",
              "com_s", "speedup", "scaling"]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for (dist, n, build, threads), row in sorted(table.items()):
            writer.writerow(dict(row, dist=dist, n=n, build=build, threads=threads))


def markdown(table):
    """One table per distribution and size: build time in ms (speedup over the sequential build)."""
    shapes = sorted({(dist, n) for dist, n, _, _ in table})
    builds = sorted({build for _, _, build, _ in table} - {"seq"})
    threads = sorted({t for _, _, build, t in table if build != "seq"})
    lines = []
    for dist, n in shapes:
        lines += [f"**{dist}，N = {n:,}**（循序建樹 {1e3 * table[dist, n, 'seq', 1]['build_s']:.2f} ms）", "",
                  "| 執行緒 | " + " | ".join(builds) + " | 較快 |", "|---|" + "---|" * (len(builds) + 1)]
        for t in threads:
            rows = {b: table[dist, n, b, t] for b in builds if (dist, n, b, t) in table}
            cells = [f"{1e3 * rows[b]['build_s']:.2f}（{rows[b]['speedup']:.1f}x）" if b in rows else "" for b in builds]
            fastest = min(rows, key=lambda b: rows[b]["build_s"])
            lines.append(f"| {t} | " + " | ".join(cells) + f" | {fastest} |")
        lines.append("")
    return "\n".join(lines)


def plot(table, folder):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    dists = sorted({dist for dist, _, _, _ in table})
    builds = sorted({build for _, _, build, _ in table} - {"seq"})
    for n in sorted({n for _, n, _, _ in table}):
        fig, axes = plt.subplots(2, len(dists), figsize=(4.2 * len(dists), 7), squeeze=False)
        for col, dist in enumerate(dists):
            time_ax, speed_ax = axes[0, col], axes[1, col]
            time_ax.axhline(1e3 * table[dist, n, "seq", 1]["build_s"], color="gray", linestyle="--",
                            label="sequential")
            for build in builds:
                rows = sorted((t, row) for (d, m, b, t), row in table.items() if (d, m, b) == (dist, n, build))
                threads = [t for t, _ in rows]
                ms = [1e3 * row["build_s"] for _, row in rows]
                spread = [[1e3 * (row["build_s"] - row["build_min"]) for _, row in rows],
                          [1e3 * (row["build_max"] - row["build_s"]) for _, row in rows]]
                time_ax.errorbar(threads, ms, yerr=spread, marker="o", capsize=3, label=build)
                speed_ax.plot(threads, [row["speedup"] for _, row in rows], marker="o", label=build)
            speed_ax.plot(threads, threads, color="gray", linestyle=":", label="ideal")
            time_ax.set(title=dist, xlabel="threads", ylabel="build time per step (ms)", xscale="log",
                        yscale="log")
            speed_ax.set(xlabel="threads", ylabel="speedup over the sequential build", xscale="log",
                         yscale="log")
            for ax in (time_ax, speed_ax):
                ax.set_xticks(threads, [str(t) for t in threads])
                ax.xaxis.set_minor_locator(matplotlib.ticker.NullLocator())
                ax.yaxis.set_major_formatter("{x:g}")
                ax.yaxis.set_minor_formatter("{x:g}")
                ax.grid(alpha=0.3)
                ax.legend()
        fig.suptitle(f"Tree build, N = {n:,} (median of repeats, bars: min and max)")
        fig.tight_layout()
        fig.savefig(folder / f"build-n{n}.png", dpi=150)
        plt.close(fig)


def main():
    if len(sys.argv) != 2:
        sys.exit(__doc__)
    folder = Path(sys.argv[1])
    table = summarise(folder)
    write_csv(table, folder / "summary.csv")
    text = markdown(table)
    (folder / "summary.md").write_text(text, encoding="utf-8")
    print(text)
    try:
        plot(table, folder)
    except ImportError:
        print("matplotlib is not installed: no figures written")


if __name__ == "__main__":
    main()
