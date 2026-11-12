#!/usr/bin/env python3
"""Summarize repeated Q2 runs and render publication-friendly PNG/PDF plots."""
import argparse
from collections import defaultdict
import csv
import gzip
import json
from pathlib import Path
import statistics as st

KEYS = ("kind", "dist", "n", "force", "force_order", "chunk", "threads")
METRICS = ("bbox_s", "build_s", "com_s", "force_s", "integrate_s", "total_s",
           "load_imbalance", "force_order_s", "force_partition_s", "force_walk_s")
PRIMARY = ("static", "dynamic", "guided", "costzones")


def config_key(row):
    return tuple(row[k] for k in KEYS)


def summarize(rows):
    per_run = defaultdict(list)
    for row in rows:
        if row["is_warmup"] == "0":
            per_run[(config_key(row), row["run_id"])].append(row)
    per_config = defaultdict(list)
    for (key, _), steps in per_run.items():
        measured = {name: st.median(float(r[name]) for r in steps) for name in METRICS}
        per_config[key].append(measured)
    summaries = []
    for key, repeats in sorted(per_config.items()):
        row = dict(zip(KEYS, key))
        row["repeats"] = len(repeats)
        for name in METRICS:
            row[name] = st.median(r[name] for r in repeats)
        row["force_s_min"] = min(r["force_s"] for r in repeats)
        row["force_s_max"] = max(r["force_s"] for r in repeats)
        summaries.append(row)
    baselines = {(s["kind"], s["dist"], s["n"]): s for s in summaries if s["force"] == "seq"}
    single = {(s["kind"], s["dist"], s["n"], s["force"], s["force_order"], s["chunk"]): s
              for s in summaries if s["threads"] == "1"}
    weak_single = {}
    for s in summaries:
        if s["kind"] == "weak" and s["threads"] == "1":
            key = (s["dist"], s["force"], s["force_order"], s["chunk"])
            if key not in weak_single or int(s["n"]) < int(weak_single[key]["n"]):
                weak_single[key] = s
    for row in summaries:
        base = baselines[(row["kind"], row["dist"], row["n"])]
        p = int(row["threads"])
        for scope in ("force", "total"):
            row[scope + "_speedup"] = base[scope + "_s"] / row[scope + "_s"]
            row[scope + "_efficiency"] = row[scope + "_speedup"] / p
        one = single.get((row["kind"], row["dist"], row["n"], row["force"],
                          row["force_order"], row["chunk"]))
        row["strategy_speedup"] = one["force_s"] / row["force_s"] if one else ""
        weak = weak_single.get((row["dist"], row["force"], row["force_order"], row["chunk"]))
        row["weak_time_efficiency"] = weak["force_s"] / row["force_s"] if weak and row["kind"] == "weak" else ""
    return summaries


def primary(row):
    return row["force"] in PRIMARY and row["force_order"] == "morton" and row["chunk"] == "0"


def figures(rows, summaries, out):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})
    colors = dict(zip(PRIMARY, ("#2864b4", "#d08020", "#35915d", "#9854ab")))

    def save(fig, name):
        fig.tight_layout()
        fig.savefig(out / (name + ".png"), dpi=160)
        fig.savefig(out / (name + ".pdf"))
        plt.close(fig)

    strong_shapes = sorted({(s["dist"], s["n"]) for s in summaries if s["kind"] == "strong"})
    for dist, n in strong_shapes:
        selected = [s for s in summaries if s["kind"] == "strong" and s["dist"] == dist and s["n"] == n]
        fig, axes = plt.subplots(2, 2, figsize=(10, 7))
        for method in PRIMARY:
            line = sorted((s for s in selected if primary(s) and s["force"] == method),
                          key=lambda s: int(s["threads"]))
            if not line:
                continue
            threads = [int(s["threads"]) for s in line]
            runtime = [1000 * s["force_s"] for s in line]
            error = [[1000 * (s["force_s"] - s["force_s_min"]) for s in line],
                     [1000 * (s["force_s_max"] - s["force_s"]) for s in line]]
            axes[0, 0].errorbar(threads, runtime, yerr=error, label=method, marker="o", color=colors[method])
            for ax, metric in ((axes[0, 1], "load_imbalance"), (axes[1, 0], "force_speedup"),
                               (axes[1, 1], "force_efficiency")):
                ax.plot(threads, [s[metric] for s in line], label=method, marker="o", color=colors[method])
        for ax, label in zip(axes.flat, ("Force time incl. overhead (ms)", "Load imbalance lambda",
                                        "Speedup vs matching-N sequential force", "Force efficiency vs sequential")):
            ax.set(xlabel="Requested threads", ylabel=label)
            ax.grid(alpha=0.2)
        axes[0, 0].legend()
        fig.suptitle(f"{dist}, N={int(n):,}; median of repeat medians (bars: min/max repeats)")
        save(fig, f"strong-{dist}-{n}")

        fig, axes = plt.subplots(1, 2, figsize=(10, 4))
        for method in ("static", "dynamic"):
            for order in ("morton", "original"):
                line = sorted((s for s in selected if s["force"] == method and s["force_order"] == order
                               and s["chunk"] == "0"), key=lambda s: int(s["threads"]))
                axes[0].plot([int(s["threads"]) for s in line], [1000 * s["force_s"] for s in line],
                             marker="o", linestyle="-" if order == "morton" else "--", label=f"{method}/{order}")
        max_threads = max(int(s["threads"]) for s in selected)
        chunk_rows = sorted((s for s in selected if s["force"] in ("dynamic", "guided") and
                             s["force_order"] == "morton" and int(s["threads"]) == max_threads),
                            key=lambda s: (s["force"], int(s["chunk"])))
        axes[1].bar(range(len(chunk_rows)), [1000 * s["force_s"] for s in chunk_rows])
        axes[1].set_xticks(range(len(chunk_rows)), [s["force"] + "\n" + s["chunk"] for s in chunk_rows])
        axes[0].set(xlabel="Requested threads", ylabel="Force time incl. ordering (ms)", title="Locality control")
        axes[1].set(ylabel="Force time (ms)", xlabel="Strategy / chunk (0 = runtime default)",
                    title=f"Chunk sensitivity, {max_threads} threads")
        axes[0].legend(fontsize=8)
        fig.suptitle(f"{dist}, N={int(n):,}")
        save(fig, f"locality-{dist}-{n}")

        chosen = [s for s in selected if primary(s) and int(s["threads"]) == max_threads]
        chosen += [s for s in selected if s["force"] == "seq"]
        fig, axes = plt.subplots(1, 2, figsize=(10, 4))
        labels = [s["force"] for s in chosen]
        bottom = [0.0] * len(chosen)
        for stage in ("bbox", "build", "com", "force", "integrate"):
            values = [1000 * s[stage + "_s"] for s in chosen]
            axes[0].bar(labels, values, bottom=bottom, label=stage)
            bottom = [a + b for a, b in zip(bottom, values)]
        bottom = [0.0] * len(chosen)
        for component in ("order", "partition", "walk"):
            values = [1000 * s["force_" + component + "_s"] for s in chosen]
            axes[1].bar(labels, values, bottom=bottom, label=component)
            bottom = [a + b for a, b in zip(bottom, values)]
        axes[0].set(ylabel="Median stage time (ms)", title="Stage medians (not median total)")
        axes[1].set(ylabel="Median force component time (ms)", title="Ordering/partition/walk diagnostics")
        for ax in axes:
            ax.legend(fontsize=8)
            ax.tick_params(axis="x", rotation=20)
        fig.suptitle(f"{dist}, N={int(n):,}; {max_threads} requested threads (seq: 1)")
        save(fig, f"stages-{dist}-{n}")

    weak_dists = sorted({s["dist"] for s in summaries if s["kind"] == "weak"})
    if weak_dists:
        fig, axes = plt.subplots(1, len(weak_dists), figsize=(5 * len(weak_dists), 4), squeeze=False)
        for ax, dist in zip(axes[0], weak_dists):
            for method in PRIMARY:
                line = sorted((s for s in summaries if s["kind"] == "weak" and s["dist"] == dist and
                               primary(s) and s["force"] == method), key=lambda s: int(s["threads"]))
                ax.plot([int(s["threads"]) for s in line], [s["weak_time_efficiency"] for s in line],
                        marker="o", label=method, color=colors[method])
            ax.set(title=dist, xlabel="Requested threads (fixed N/threads)", ylabel="T1(Nbase) / Tp(p*Nbase)")
            ax.grid(alpha=0.2)
        axes[0, 0].legend()
        fig.suptitle("Weak time efficiency; Barnes-Hut work is not exactly linear in N")
        save(fig, "weak-time-efficiency")

    collision_teams = sorted({int(r["threads"]) for r in rows if r["kind"] == "collision" and primary(r)})
    for threads in collision_teams:
        fig, axes = plt.subplots(1, 3, figsize=(14, 4))
        for method in PRIMARY:
            bins = defaultdict(list)
            for r in rows:
                if r["kind"] == "collision" and primary(r) and r["force"] == method and int(r["threads"]) == threads:
                    bins[int(r["step"]) // 25].append(r)
            times = [st.median((int(r["step"]) + 1) * float(r["dt"]) for r in bins[b]) for b in sorted(bins)]
            for ax, metric, scale in ((axes[0], "load_imbalance", 1), (axes[1], "force_s", 1000)):
                values = [st.median(float(r[metric]) for r in bins[b]) * scale for b in sorted(bins)]
                ax.plot(times, values, label=method, color=colors[method])
            if method == "costzones":
                axes[2].plot(times, [st.median(int(r["interactions"]) / int(r["n"]) for r in bins[b])
                                     for b in sorted(bins)], color=colors[method])
        for ax, label in zip(axes, ("Load imbalance lambda", "Force time incl. overhead (ms)", "Interactions / particle")):
            ax.set(xlabel="Simulation time after drift", ylabel=label)
            ax.grid(alpha=0.2)
        axes[0].legend(fontsize=8)
        fig.suptitle(f"Evolving collision workload, {threads} threads; 25-step bins across repeats")
        save(fig, f"collision-{threads}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    manifest = args.input / "manifest.json"
    if manifest.exists() and "completed_utc" not in json.loads(manifest.read_text(encoding="utf-8")):
        raise ValueError("experiment is incomplete; inspect the manifest and logs before analysis")
    path = args.input / "steps.csv"
    opener = path.open
    if not path.exists():
        path = args.input / "steps.csv.gz"
        opener = lambda **kwargs: gzip.open(path, "rt", **kwargs)
    with opener(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    summaries = summarize(rows)
    if not summaries:
        raise ValueError("no measured rows")
    args.output.mkdir(parents=True, exist_ok=True)
    with (args.output / "summary.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(summaries[0]))
        writer.writeheader()
        writer.writerows(summaries)
    figures(rows, summaries, args.output)
    print(f"Saved {len(summaries)} configurations and figures to {args.output}")


if __name__ == "__main__":
    main()
