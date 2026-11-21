#!/usr/bin/env python3
"""Centre of mass: summarise the per-level profile recorded by run_com.py.

    python3 scripts/analyze_com.py results/com-local [THREADS = 8]

Writes summary.csv, summary.md and, when matplotlib is installed,
levels-n<N>.png (the profile of every level at THREADS threads) and
wait-share.png. Every value is the median over repeats. Per configuration:

  busy     sum over levels of the mean per-thread time spent on the level's nodes
  wait     sum over levels of the mean per-thread time spent at the level's barrier
  barrier  sum over levels of the smallest per-thread wait: the barriers themselves
  other    total - busy - wait: sorting the nodes by depth, starting the threads
"""
import csv
import statistics
import sys
from collections import defaultdict
from pathlib import Path

FIELDS = ("busy_mean_s", "busy_max_s", "wait_mean_s", "wait_min_s", "total_s", "seq_s")


def load(folder):
    """profile[dist, n, threads][level]: node count and the median of every field."""
    raw = defaultdict(lambda: defaultdict(list))
    with (folder / "levels.csv").open(newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            raw[row["dist"], int(row["n"]), int(row["threads"])][int(row["level"])].append(row)
    return {config: {level: dict({f: statistics.median(float(r[f]) for r in rows) for f in FIELDS},
                                 nodes=int(rows[0]["nodes"]))
                     for level, rows in sorted(levels.items())}
            for config, levels in raw.items()}


def summarise(profile):
    table = {}
    for config, levels in profile.items():
        rows = list(levels.values())
        total, seq = rows[0]["total_s"], rows[0]["seq_s"]
        busy = sum(r["busy_mean_s"] for r in rows)
        wait = sum(r["wait_mean_s"] for r in rows)
        table[config] = dict(levels=len(rows), nodes=sum(r["nodes"] for r in rows), seq_s=seq, total_s=total,
                             speedup=seq / total, busy_s=busy, wait_s=wait,
                             barrier_s=sum(r["wait_min_s"] for r in rows), other_s=total - busy - wait,
                             wait_share=wait / total)
    return table


def write_csv(table, path):
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["dist", "n", "threads"] + list(next(iter(table.values()))))
        writer.writeheader()
        for (dist, n, threads), row in sorted(table.items()):
            writer.writerow(dict(row, dist=dist, n=n, threads=threads))


def markdown(table):
    """One table per distribution, times in ms."""
    lines = []
    for dist in sorted({dist for dist, _, _ in table}):
        lines += [f"**{dist}**", "",
                  "| N | 層數 | 執行緒 | 循序 | 逐層 | 加速比 | 計算 | 等待 | 其中 barrier | 其他 | 等待占比 |",
                  "|---|---|---|---|---|---|---|---|---|---|---|"]
        for (d, n, threads), r in sorted(table.items()):
            if d != dist:
                continue
            ms = [f"{1e3 * r[k]:.3f}" for k in ("seq_s", "total_s", "busy_s", "wait_s", "barrier_s", "other_s")]
            lines.append(f"| {n:,} | {r['levels']} | {threads} | {ms[0]} | {ms[1]} | {r['speedup']:.2f}x | "
                         f"{ms[2]} | {ms[3]} | {ms[4]} | {ms[5]} | {100 * r['wait_share']:.0f}% |")
        lines.append("")
    return "\n".join(lines)


def plot(profile, table, folder, threads):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    dists = sorted({dist for dist, _, _ in table})
    sizes = sorted({n for _, n, _ in table})
    for n in sizes:
        fig, axes = plt.subplots(1, len(dists), figsize=(4.5 * len(dists), 4), squeeze=False)
        for ax, dist in zip(axes[0], dists):
            levels = profile[dist, n, threads]
            depth = list(levels)
            busy = [1e3 * r["busy_mean_s"] for r in levels.values()]
            wait = [1e3 * r["wait_mean_s"] for r in levels.values()]
            ax.bar(depth, busy, label="busy")
            ax.bar(depth, wait, bottom=busy, label="wait at the barrier")
            ax.set(title=f"{dist}: {len(depth)} levels", xlabel="depth", ylabel="ms per thread")
            ax.xaxis.set_major_locator(matplotlib.ticker.MaxNLocator(integer=True))
            nodes = ax.twinx()
            nodes.plot(depth, [r["nodes"] for r in levels.values()], color="gray", marker=".", label="nodes")
            nodes.set(yscale="log", ylabel="nodes")
            ax.legend(loc="upper left")
        fig.suptitle(f"Centre of mass by level, N = {n:,}, {threads} threads")
        fig.tight_layout()
        fig.savefig(folder / f"levels-n{n}.png", dpi=150)
        plt.close(fig)

    fig, axes = plt.subplots(1, len(dists), figsize=(4.5 * len(dists), 4), squeeze=False, sharey=True)
    for ax, dist in zip(axes[0], dists):
        for n in sizes:
            rows = sorted((t, r) for (d, m, t), r in table.items() if (d, m) == (dist, n))
            ax.plot([t for t, _ in rows], [100 * r["wait_share"] for _, r in rows], marker="o",
                    label=f"N = {n:,} ({rows[0][1]['levels']} levels)")
        ax.set(title=dist, xlabel="threads", ylabel="time at barriers (% of the stage)", xscale="log")
        ax.set_xticks([t for t, _ in rows], [str(t) for t, _ in rows])
        ax.xaxis.set_minor_locator(matplotlib.ticker.NullLocator())
        ax.grid(alpha=0.3)
        ax.legend()
    fig.tight_layout()
    fig.savefig(folder / "wait-share.png", dpi=150)
    plt.close(fig)


def main():
    if len(sys.argv) not in (2, 3):
        sys.exit(__doc__)
    folder = Path(sys.argv[1])
    profile = load(folder)
    table = summarise(profile)
    write_csv(table, folder / "summary.csv")
    text = markdown(table)
    (folder / "summary.md").write_text(text, encoding="utf-8")
    print(text)
    try:
        plot(profile, table, folder, int(sys.argv[2]) if len(sys.argv) == 3 else 8)
    except ImportError:
        print("matplotlib is not installed: no figures written")


if __name__ == "__main__":
    main()
